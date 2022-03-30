#code source courtesy https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
corner_params = dict( maxCorners = 120,
                       #qualityLevel = 0.3,
                       qualityLevel = 0.07,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lucas_params = dict( winSize  = (100,100),
                  #winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# grab first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **corner_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lucas_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        a,b = int(a), int(b)
        c,d= int(c),int(d)
        print ('a,b',a,b)
        mask = cv2.line(mask, (a,b),(c,d), (255,0,0), 10)
        frame = cv2.circle(frame,(a,b),15,(0,255,0),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

