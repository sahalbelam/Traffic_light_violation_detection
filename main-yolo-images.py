import torch
from PIL import Image
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load image
image = Image.open('car.jpg')

# Perform inference
results = model(image)

# Display results
results.show()

# Convert results to OpenCV format and display using OpenCV
results.render()  # Updates results.imgs with boxes and labels
for img in results.imgs:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Detection', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
