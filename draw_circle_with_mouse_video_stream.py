import cv2

def my_draw(event,x,y,flags,param):

    global x1,y1,draw_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        x1=x
        y1=y
        print ('x1',x1)
        print ('y1,y1')
        draw_flag=1
        # circle here does not work
        #cv2.circle(frame, center=(x1,y1), radius=35, color=(0,0,255), thickness=10)

        
draw_flag=0

cap = cv2.VideoCapture(0) 

cv2.namedWindow('my_window')

cv2.setMouseCallback('my_window', my_draw) 


while True:
    ret, frame = cap.read()

    if draw_flag == 1:
        pass
        cv2.circle(frame, center=(x1,y1), radius=25, color=(0,255,0), thickness=10)
        
    cv2.imshow('my_window', frame)

    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
