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

# Define colors for traffic light
RED = (0, 0, 255)
GREEN = (0, 255, 0)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('videos/veh4.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

cy1 = 427
offset = 6

count = 0
tracker = Tracker()

vehicle_count = 0

# Initialize traffic light color and timer
light_color = GREEN
light_timer = time.time()

while True:    
    ret, frame = cap.read()
    if not ret:
       break

    count += 1
    if count % 3 != 0:
        continue

    # Change traffic light color every 5 seconds
    if time.time() - light_timer > 5:
        if light_color == GREEN:
            light_color = RED
        else:
            light_color = GREEN
        light_timer = time.time()

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
       
        if 'bus' in c or 'car' in c or 'truck' in c:
            list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            if light_color == RED:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                vehicle_count += 1

    cvzone.putTextRect(frame, f'vehicle count: {vehicle_count}', (50, 60), 2, 2)

    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # Draw traffic light
    cv2.circle(frame, (500, 50), 30, light_color, -1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()    
cv2.destroyAllWindows()
