import torch
from PIL import Image
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Perform inference
    results = model(frame_pil)
    
    # Convert results to OpenCV format and display using OpenCV
    results.render()  # Updates results.imgs with boxes and labels
    for img in results.imgs:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('YOLOv5 Detection', img)
        
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()