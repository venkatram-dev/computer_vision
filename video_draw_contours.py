import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

ret,frame = cap.read()
desired_roi=cv2.selectROI(frame)

x = desired_roi[0]
y = desired_roi[1]
w = desired_roi[2]
h =  desired_roi[3]

while True:
    ret, frame = cap.read()

    frame = frame[y:y+h,x:x+w]

    cv2.imshow('frame',frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray',gray)

    blur = cv2.GaussianBlur(gray,(7,7),0)

    cv2.imshow('blur',blur)

    ret2,img2=cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)

    cv2.imshow('binary thresholded image',img2)

    contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2, contours, -1, 255, 5)
    cv2.imshow("contours", img2)

    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(img2, [hull], -1, (0, 0, 255), 1)  

    cv2.imshow("convex hull", img2)
 
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    c_hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, c_hull)

    print ('defects',defects)
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        farthest = contours[f][0]
        cv2.circle(img2,(farthest),5,[0,0,255],-1)

    cv2.imshow('defects',img2)
 
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
