import cv2
import torch

# Load YOLOv5 pre-trained model (supports PyTorch Hub)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s' for a small, fast model

# Define a list of vehicle classes (based on YOLO model's labels)
VEHICLE_CLASSES = ['car', 'motorbike', 'bus', 'truck']

# Initialize video capture
video_path = 'input_video.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the output video
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize vehicle count
total_vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frame is returned

    # Convert frame (BGR) to RGB for YOLO model
    results = model(frame)

    # Extract detections
    detections = results.pandas().xyxy[0]  # Bounding boxes and labels

    # Filter vehicle detections
    vehicle_detections = detections[detections['name'].isin(VEHICLE_CLASSES)]

    # Count vehicles in the current frame
    current_frame_count = len(vehicle_detections)
    total_vehicle_count += current_frame_count

    # Annotate the frame with detections and counts
    for _, detection in vehicle_detections.iterrows():
        # Bounding box coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        label = detection['name']  # Class name
        confidence = detection['confidence']  # Confidence score

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display vehicle count on the frame
    cv2.putText(frame, f'Frame Vehicle Count: {current_frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Vehicle Count: {total_vehicle_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional, for debugging)
    cv2.imshow('Vehicle Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f'Total Vehicles Counted: {total_vehicle_count}')
