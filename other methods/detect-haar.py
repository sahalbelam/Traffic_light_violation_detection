import cv2
import numpy as np

# Path to input video
video_path = "veh6.mp4"

# Load pre-trained Haar cascade classifier for vehicle detection
haar_cascade_path = "cars.xml"  # Change this to the path of your Haar cascade XML file
car_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Open video capture object
cap = cv2.VideoCapture(video_path)

# Define minimum size for vehicle detection
min_vehicle_size = (80, 80)

# Define target frame size
target_width = 1000  # Width of the target frame
target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (target_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# Define imaginary line coordinates
line_start = (100, target_height - target_height // 3)
line_end = (800, target_height - target_height // 3)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (target_width, target_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect vehicles using Haar cascade
    vehicles = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_vehicle_size)

    # Draw bounding boxes around detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize frame to target size
    frame = cv2.resize(frame, (target_width, target_height))

    # Draw the line on the frame
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
    
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
