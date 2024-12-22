import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace with your YOLO model weight file

# Input and output paths
input_video_path = "Videos\\video.mp4"
output_video_path = "output_video.avi"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model.predict(source=frame, save=False, conf=0.5, verbose=False)

    # Get the detection results
    detections = results[0].boxes.data.cpu().numpy()

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        label = f"{model.names[int(class_id)]} {confidence * 100:.2f}%"  # Add confidence percentage

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('YOLO Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
