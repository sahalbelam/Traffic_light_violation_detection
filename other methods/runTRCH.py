import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Open video capture object
cap = cv2.VideoCapture("veh6.mp4")

# Define target frame size
target_width = 1000  # Width of the target frame
target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (target_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (target_width, target_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PyTorch tensor
    input_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    
    # Perform object detection
    with torch.no_grad():
        predictions = model([input_tensor])[0]
    
    # Draw bounding boxes around detected vehicles
    for box in predictions['boxes']:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Resize frame to target size
    frame = cv2.resize(frame, (target_width, target_height))

    # Write the modified frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Check for key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Stop running if "q" key is pressed
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
