import torch
from PIL import Image
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the target width and height for resizing
target_width = 1000
target_height = 500

line_start = (100, target_height - target_height // 3)
line_end = (target_width - 100, target_height - target_height // 3)

# Initialize car counter
car_counter = 0

# Define a function to process individual video frames and detect vehicles
def detect_vehicles(frame):
    global car_counter
    
    # Resize frame to target size
    frame_resized = cv2.resize(frame, (target_width, target_height))

    # Convert frame to PIL Image format
    frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

    # Perform inference
    results = model(frame_pil)

    # Get the first frame with detections
    img_with_detections = results.render()[0]

    # Convert PIL image to NumPy array
    img_with_detections_np = np.array(img_with_detections)

    # Draw the imaginary line on the frame
    cv2.line(img_with_detections_np, line_start, line_end, (0, 255, 0), 2)

    # Check if any detected object crosses the imaginary line
    for det in results.xyxy[0]:
        if det[0] < min(line_start[0], line_end[0]) and det[2] > max(line_start[0], line_end[0]):
            car_counter += 1

    return img_with_detections_np

# Open the input video
cap = cv2.VideoCapture('veh6.mp4')  # Replace with your video filename

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Process video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Detect vehicles in the frame
    frame_with_detections = detect_vehicles(frame)

    # Display the frame with detected vehicles and car counter
    cv2.putText(frame_with_detections, f'Car Counter: {car_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Vehicle detection', frame_with_detections)

    # Exit on 'q' key press 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):  # Increment car counter on 'r' key press
        car_counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
