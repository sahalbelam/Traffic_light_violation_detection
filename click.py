import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import Tracker
import time

model = YOLO('models/yolov5su.pt')

target_width = 1000
target_height = 600

line_start = (100, target_height - target_height // 3)
line_end = (target_width - 100, target_height - target_height // 3)

# Define default colors for traffic light
DEFAULT_RED = (0, 0, 255)
DEFAULT_GREEN = (0, 255, 0)

# Initialize traffic light color and timer
light_color = DEFAULT_GREEN
light_timer = time.time()

def select_color(event, x, y, flags, param):
    global light_color
    if event == cv2.EVENT_LBUTTONDOWN:
        light_color = DEFAULT_RED
    elif event == cv2.EVENT_RBUTTONDOWN:
        light_color = DEFAULT_GREEN

cap = cv2.VideoCapture('videos/veh6.mp4')

cv2.namedWindow('Traffic Control')
cv2.setMouseCallback('Traffic Control', select_color)

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

cy1 = 427
offset = 6

count = 0
tracker = Tracker()

vehicle_count = 0

while True:    
    ret, frame = cap.read()
    if not ret:
       break

    count += 1
    if count % 3 != 0:
        continue

    # Change traffic light color every 5 seconds
    if time.time() - light_timer > 5:
        light_timer = time.time()

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    bbox_idx = []
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
       
        if 'bus' in c or 'car' in c or 'truck' in c:
            bbox_idx.append([x1, y1, x2, y2])

    for bbox in bbox_idx:
        x3, y3, x4, y4 = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            if light_color == DEFAULT_RED:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {vehicle_count}', (x3, y3), 1, 1)
                vehicle_count += 1
               

    cvzone.putTextRect(frame, f'vehicle count: {vehicle_count}', (50, 60), 2, 2)

    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # Draw traffic light
    cv2.circle(frame, (500, 50), 30, light_color, -1)

    cv2.imshow("Traffic Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()    
cv2.destroyAllWindows()