from collections import defaultdict
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import imageio

# Load the YOLOv8 model
model = YOLO('best_final.pt')

# Open the video file (replace with the correct path)
video_path = "./test_video/final_tl_vid.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the counting polygon
video_width = 300
video_height = 200

# Define the rectangle points
polygon_pts = np.array([
    (0, 0),                     # Top-left corner
    (video_width, 0),           # Top-right corner
    (video_width, video_height), # Bottom-right corner
    (0, video_height)            # Bottom-left corner
], dtype=np.int32)

# Store the track history
track_history = defaultdict(lambda: {"track": None, "counted": False})  # Initialize with None and False

# Initialize counters
in_count = 0
out_count = 0

# Define the output video writer using imageio
output_path = "./results/output_video_rect6.mp4"
output_writer = imageio.get_writer(output_path)

# Parameters for optimization
new_width = 320  # Adjust the resolution as needed
new_height = 240
skip_frames = 2  # Process every frame

# Counter to track frames processed
frame_counter = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Skip frames based on skip_frames
        if frame_counter % skip_frames == 0:
            # Resize the frame to a smaller resolution
            frame = cv2.resize(frame, (new_width, new_height))

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker='bytetrack.yaml')

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                current_position = Point(float(x), float(y))

                # Check if the object is within the counting polygon
                if Polygon(polygon_pts).contains(current_position):
                    if not track_history[track_id]["counted"]:
                        if track_history[track_id]["track"] is None:  # If it's the first time inside
                            track_history[track_id]["track"] = []  # Initialize an empty list for the track
                        # Draw the tracking lines
                        track = track_history[track_id]["track"]
                        track.append((int(x + w / 2), int(y + h / 2)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)
                        points = np.array(track, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                        in_count += 1
                        track_history[track_id]["counted"] = True
                else:
                    if track_history[track_id]["counted"]:  # If it was previously inside
                        track_history[track_id]["counted"] = False
                        out_count += 1

            # Display the annotated frame
            cv2.putText(annotated_frame, f'In Count: {in_count}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f'Out Count: {out_count}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(annotated_frame, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Write the frame to the output video using imageio
            output_writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_counter += 1
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, the output video writer, and close the display window
cap.release()
output_writer.close()
cv2.destroyAllWindows()