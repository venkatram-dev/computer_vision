import numpy as np
import cv2 
cap = cv2.VideoCapture('output.mp4')
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    print ('frame',frame)
 
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)

    if cv2.waitKey(100) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
