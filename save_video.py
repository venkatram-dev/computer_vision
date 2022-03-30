import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out_flip = cv2.VideoWriter('output_flip.mp4', fourcc, 20.0, (1280,  720))
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280,  720))
while cap.isOpened():
    ret, frame = cap.read()

    #write normal video
    out.write(frame)

    frame = cv2.flip(frame, 0)
    # write the flipped frame
    out_flip.write(frame)


    cv2.imshow('frame', frame)

 
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
out_flip.release()
cv2.destroyAllWindows()
