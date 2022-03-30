import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    """
    face = frame.copy()
    cv2.imshow('face', face)
    face_box = haar_cascade.detectMultiScale(face)
    if len(face_box) >0:
        print ('face_box',face_box)
        left_top_x=face_box[0][0]
        left_top_y=face_box[0][1]
        width = face_box[0][2]
        height = face_box[0][3]
   
        box = cv2.rectangle(face, (left_top_x,left_top_y), (left_top_x+width,left_top_y+height), (0,255,0), 20)

        cv2.imshow('box', box)

    """

    face_box = haar_cascade.detectMultiScale(frame)
    if len(face_box) >0:
        print ('face_box',face_box)
        left_top_x=face_box[0][0]
        left_top_y=face_box[0][1]
        width = face_box[0][2]
        height = face_box[0][3]

        box = cv2.rectangle(frame, (left_top_x,left_top_y), (left_top_x+width,left_top_y+height), (0,255,0), 20)

        cv2.imshow('box', box)
 
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
