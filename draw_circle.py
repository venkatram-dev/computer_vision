import cv2


def my_draw(event,x,y,flags,param):
  if event==cv2.EVENT_LBUTTONDOWN:
    print ('x',x)
    #print ('y',y)
    print ('event',event)
    print ('flags',flags)
    print ('param',param)
    cv2.circle(img,(x,y),100,(255,0,0),5)

img = cv2.imread("apple.jpeg")
window_name='my_window'
cv2.namedWindow(winname=window_name)
cv2.setMouseCallback(window_name,my_draw)

while True:
    cv2.imshow(window_name,img)
    ## waitkey(0) does not draw circle
    ## waitkey(1) also works
    #key = cv2.waitKey(30) 
    #print('key',key) 
    ## draws immediately after lbutton down
    ## breaks when escape key is pressed
    #if key & 0xFF == 27:
        #break

    ##below works in a different way
    ##first captures the x,y location
    ##does not draw immediately
    ##and draws after any other key is pressed
    ## breaks when esc is pressed
    #key2 = cv2.waitKey(0)
    #print('key2',key2)
    #if key2 & 0xFF == 27:
        #break

    ##first captures the x,y location
    ##does not draw immediately
    ##and draws after any other key is pressed
    ## breaks when q is pressed
    key1 = cv2.waitKey(0) 
    print ('key1',key1)
    if key1 & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
