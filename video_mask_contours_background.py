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

    #cv2.imshow('gray',gray)

    #blur = cv2.GaussianBlur(gray,(7,7),0)
    blur = gray
    #cv2.imshow('blur',blur)

    ret2,img2=cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)
    #ret2,img2=cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

    #cv2.imshow('binary thresholded image',img2)

    #contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(frame.shape, np.uint8)
    mask.fill(255)

    background = cv2.imread('flower.jpeg')
    background = cv2.resize(background,(frame.shape[1],frame.shape[0]))

    print ('contour leng',len(contours))
    for c in range(len(contours)):
        cv2.drawContours(img2, contours, -1, 255, 5)
        cv2.fillConvexPoly(mask, contours[c], (0, 0, 0))

    ret,rev_mask=cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)

    print ('bg shape',background.shape)
    print ('frame shape',frame.shape)
    modified_img1 = cv2.bitwise_and(rev_mask, frame)
    modified_img2 = cv2.bitwise_and(mask, background)

    modified_img3 = modified_img1

    black_pixels = np.where(
    (modified_img3[:, :, 0] == 0) &
    (modified_img3[:, :, 1] == 0) &
    (modified_img3[:, :, 2] == 0)
    )

    # set those pixels to white
    modified_img3[black_pixels] = [255, 255, 255]


    modified_img4 = modified_img2

    black_pixels = np.where(
    (modified_img4[:, :, 0] == 0) &
    (modified_img4[:, :, 1] == 0) &
    (modified_img4[:, :, 2] == 0)
    )

    # set those pixels to white
    modified_img4[black_pixels] = [255, 255, 255]

    modified_img5 = cv2.bitwise_and(modified_img3, modified_img4)

    cv2.imshow("contours", img2)
    cv2.imshow("mask", mask)
    cv2.imshow("rev_mask", rev_mask)
    cv2.imshow("background", background)
    cv2.imshow("modified_img1", modified_img1)
    cv2.imshow("modified_img2", modified_img2)
    cv2.imshow("modified_img3", modified_img3)
    cv2.imshow("modified_img4", modified_img4)
    cv2.imshow("final output", modified_img5)

 
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
