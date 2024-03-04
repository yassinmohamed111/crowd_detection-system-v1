from ultralytics import YOLO
import cv2
import numpy as np
import os 

cap = cv2.VideoCapture(r"C:\Users\yassi\Desktop\crowd detection system\crowd videos\nn.mp4")

ret, frame = cap.read()
model = YOLO(r"C:\Users\yassi\Desktop\crowd detection system\crowd model\crowd59rp.pt")

while ret:
    results = model(frame , imgsz = 814 , conf = 0.3 , save = True)
    
    # Initialize count of people
    people_count = 0
    
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, r[:4])  # Extract only the first four values

            # Calculate center coordinates of the box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw circle around the detected object
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
            
            # Increment count if the detected object is a person
            if int(r[5]) == 0:  # Assuming person class is index 0
                people_count += 1
    
    # Check if the people count exceeds the limit
    if people_count > 30:
        # Print "Overlimit" in red
        cv2.putText(frame, "Overlimit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
    
    # Overlay count on the frame
    cv2.putText(frame, f"People: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()





"""
from ultralytics import YOLO
import cv2
import numpy as np


image_path = "103137062-a5132700-4700-11eb-9fb6-9ba56a846d96 (1).jpg"
frame = cv2.imread(image_path)


model = YOLO("best (1).pt")


results = model(frame, imgsz=1200)


for result in results:
    detections = []
    for r in result.boxes.data.tolist():
       
        x1, y1, x2, y2 = map(int, r[:4])  

        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

       
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  


cv2.imshow('image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""