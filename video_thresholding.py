import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray',gray)

    blur = cv2.GaussianBlur(gray,(7,7),0)

    cv2.imshow('blur',blur)

    # very light color becomes black, others are white
    ret1,img1=cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)

    cv2.imshow('inverse thresholded image',img1)

    #very light color like white remains white, all others become black

    ret2,img2=cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

    cv2.imshow('binary thresholded image',img2)

    #similar to above but uses otsu which is like adaptive thresholding

    ret,img=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('otsu thresholded image',img)

 
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
