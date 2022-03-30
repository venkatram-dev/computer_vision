import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    blur = cv2.GaussianBlur(frame,(7,7),0)

    edge=cv2.Canny(image=blur, threshold1=40 , threshold2=100)

    # Display the resulting frame
    #cv2.imshow('blur', blur)

    # normal color
    # Display the resulting frame
    #cv2.imshow('frame', frame)
    #cv2.imshow('edge', edge)

    #h,w,c = frame.shape;
    #h1,w1,c1 = blur.shape;

    #print(edge.shape)
    #edge is 2d array with out color
    #just do below to make it look like 3d array

    stacked_edge = np.stack((edge,)*3, axis=-1)
    print (stacked_edge.shape)

    #cv2.imshow('stacked_edge',stacked_edge)

    both_frames= cv2.hconcat([frame, stacked_edge])
    cv2.imshow('both_frames', both_frames)
    
 
    if cv2.waitKey(1) == ord('q'):
        break

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#print ('width',w)
#print ('height',h)



cap.release()
cv2.destroyAllWindows()
