import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
   
    cv2.circle(frame,(600,300),50,(0,255,0),5)  
    cv2.rectangle(frame, (100, 100), (200, 200), color=(255,0,255),thickness= 5)
    # normal color
    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print ('width',w)
print ('height',h)

