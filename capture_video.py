import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    #cv2.imshow('frame', gray)

    # does not work, looks blue
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame
    #cv2.imshow('frame', rgb)

    # normal color
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    #cv2.imshow('second_frame', frame)
    
    frame_two=cv2.resize(frame,(240,160))

    cv2.imshow('second_frame', frame_two)
 
    if cv2.waitKey(1) == ord('q'):
        break

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print ('width',w)
print ('height',h)



cap.release()
cv2.destroyAllWindows()
