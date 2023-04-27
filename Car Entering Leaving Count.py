import numpy as np
import math
import cv2
import cvzone
from ultralytics import YOLO
from sort import *

# Read the video.
cap = cv2.VideoCapture("videos/videoplayback (1).mp4")

# Make the model.
model = YOLO("yolo_weights/yolov8l.pt")

# Tracker
tracker = Sort(max_age=20, min_hits=3 , iou_threshold=0.3)

# Limits 
limits_up =  [970,500,1700,500]
limits_down = [40,640,919,640]

# Counts
total_counts_up = []
total_counts_down = []

# mask
mask = cv2.imread("mask.png")

# Class Names
class_names = ['person','bicycle','car','motorcycle','airplane','bus',
                'train',    'truck',    'boat',    'traffic light','fire hydrant',
                'stop sign',    'parking meter',    'bench',    'bird',    'cat',    
                'dog',    'horse',    'sheep',    'cow',    'elephant',    'bear',    
                'zebra',    'giraffe',    'backpack',    'umbrella',    'handbag',    
                'tie',    'suitcase',    'frisbee',    'skis',    'snowboard',    
                'sports ball',    'kite',    'baseball bat',    'baseball glove',    
                'skateboard',    'surfboard',    'tennis racket',    'bottle',    
                'wine glass',    'cup',    'fork',    'knife',    'spoon',    
                'bowl',    'banana',    'apple',    'sandwich',    'orange',    
                'broccoli',    'carrot',    'hot dog',    'pizza',    'donut',    
                'cake',    'chair',    'couch',    'potted plant',    'bed',    
                'dining table',    'toilet',    'tv',    'laptop',    'mouse',    
                'remote',    'keyboard',    'cell phone',    'microwave',    'oven',    
                'toaster',    'sink',    'refrigerator',    'book',    'clock',    
                'vase',    'scissors',    'teddy bear',    'hair drier',    
                'toothbrush']

# Iterate through each and every frame.
while True:
    success,img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    
    results = model(imgRegion,stream=True)

    detections = np.empty((0,5))
    cv2.line(img,(limits_up[0],limits_up[1]),(limits_up[2],limits_up[3]),color=(0,0,255),thickness=5)
    cv2.line(img,(limits_down[0],limits_down[1]),(limits_down[2],limits_down[3]),color=(0,0,255),thickness=5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # BBOX
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            classes = int(box.cls[0])
            currentClass = class_names[classes]

            if currentClass == "car" or currentClass=="truck" or currentClass=="bus" and conf > 0.5:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
            
    results_tracker = tracker.update(detections)

    for result in results_tracker:
            x1,y1,x2,y2,ID = result
            x1,y1,x2,y2,ID = int(x1),int(y1),int(x2),int(y2),int(ID)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=8,rt=1,t=1,colorR=(255,0,255))
            cvzone.putTextRect(img,f"{currentClass}{ID}",(max(0,x1),max(35,y1)),scale=1,thickness=1,colorT=(255,0,0))

            # Center
            cx,cy = x1+w//2 , y1+h//2
            cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)

            if limits_up[0]<cx<limits_up[2] and limits_up[1]-15<cy<limits_up[3]+15:
                 if total_counts_up.count(ID)==0:
                      total_counts_up.append(ID)
                      cv2.line(img,(limits_up[0],limits_up[1]),(limits_up[2],limits_up[3]),color=(0,255,0),thickness=5)

            if limits_down[0]<cx<limits_down[2] and limits_down[1]-15<cy<limits_down[3]+15:
                 if total_counts_down.count(ID)==0:
                      total_counts_down.append(ID)
                      cv2.line(img,(limits_down[0],limits_down[1]),(limits_down[2],limits_down[3]),color=(0,255,0),thickness=5)

    cv2.putText(img,f"CARS UP : {str(len(total_counts_up))}",(1200,100),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
    cv2.putText(img,f"CARS DOWN : {str(len(total_counts_down))}",(500,100),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
    #cvzone.putTextRect(img,f"CARS___UP : {len(str(total_counts_up))}",((10,50).(500,150)))

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
         break
    
cap.release()
cv2.destroyAllWindows()

