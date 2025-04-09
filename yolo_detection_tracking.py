import cv2
import time
import os
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "models/yolov10m.pt"
VIDEO_PATH = "assets/testing.mp4"
OUTPUT_DIR = "screenshots"  # Directory to save screenshots

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()

    cap = cv2.VideoCapture("rtsp:172.24.110.245:8080/h264_ulaw.sdp")

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Dictionary to store the start time for each tracking ID
    start_times = {}
    # Set to keep track of which IDs have already triggered a screenshot
    screenshot_taken = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Time for FPS calculation
        start_time = time.perf_counter()
        
        # Perform detection and tracking
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        # Current time for timer calculation
        current_time = time.perf_counter()

        # Draw bounding boxes and timers
        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            # If this is a new tracking ID, record its start time
            if tracking_id not in start_times:
                start_times[tracking_id] = current_time

            # Calculate elapsed time in seconds for this object
            elapsed_time = current_time - start_times[tracking_id]

            # Determine bounding box color: red if > 5s, otherwise default (blue)
            box_color = (0, 0, 255) if elapsed_time > 2 else (255, 0, 0)

            # Draw the bounding box
            cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), 
                         (int(bounding_box[2]), int(bounding_box[3])), 
                         box_color, 2)
            
            # Display the timer above the bounding box
            cv2.putText(frame, f"{elapsed_time:.1f}s", 
                       (int(bounding_box[0]), int(bounding_box[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save screenshot if timer exceeds 5 seconds and hasn't been saved yet
            if elapsed_time > 2 and tracking_id not in screenshot_taken:
                screenshot_frame = frame.copy()  # Copy the frame to avoid modifying the display
                # Redraw the bounding box in red for the screenshot (ensuring it's clear)
                cv2.rectangle(screenshot_frame, (int(bounding_box[0]), int(bounding_box[1])), 
                             (int(bounding_box[2]), int(bounding_box[3])), 
                             (0, 0, 255), 2)
                cv2.putText(screenshot_frame, f"{elapsed_time:.1f}s", 
                           (int(bounding_box[0]), int(bounding_box[1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Generate filename with tracking ID and timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{OUTPUT_DIR}/object_{tracking_id}_{timestamp}.jpg"
                cv2.imwrite(filename, screenshot_frame)
                print(f"Screenshot saved: {filename}")
                
                # Mark this tracking ID as having a screenshot taken
                screenshot_taken.add(tracking_id)

        # Calculate and display FPS
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps}")

        cv2.imshow("Frame", frame)

        # Break the loop if 'q' or ESC is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()