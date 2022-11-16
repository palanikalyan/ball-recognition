# import the necessary packages

from collections import deque 
#list like data structure will keep prev positions of ball 
#can make a trail of the ball from it 

import numpy as np
import argparse
import imutils 
#this is that guys list of Opencv stuff he uses - got resizing and all - can use pip to get it 
#$ pip install --upgrade imutils

import cv2
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="C:/Object_detection/models-master/research/object_detection/test_images/multi_angle.mp4") 
#can put path to video here. That is if it is there
#if not there the program will just use the webcam

ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
# this tells max size of deque which is the list with points 

args = vars(ap.parse_args())


##Put lower & upper boundaries of colour 
#colourLow = (0, 135, 30)
#colourHigh = (19, 255, 255)


#Put lower & upper boundaries of colour 
colourLow = (0, 135, 30)
colourHigh = (19, 255, 255)
pts = deque(maxlen=args["buffer"]) #initialises our deque points

# if a video path was not supplied, grab the reference
# to the webcam
# item that tells if we using a video or webcam
if not args.get("video", False):
    cap = cv2.VideoCapture(0) #imutils.Video stream item works good with webcam 

# otherwise, grab a reference to the video file
else:
    cap = cv2.VideoCapture(args["video"]) #this is if the video is supplied


    
#Loop for video frame capturing
while True:
    #calls the read method in our capture module 
    ret, frame = cap.read()

    #if we were running a video from external source and no other frame was taken again for processing
    #it means we reached end of video so we break out of loop
    if frame is None:
        break

    frame = imutils.resize(frame, width=800) #smaller frames means faster processing so we resize
    blurred = cv2.GaussianBlur(frame, (11, 11), 0) #blur reduces picture noise to allow us to see stuff more clearly 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converting frame to HSV

    # we now masking to get the desired colour only
    # we do erosion, dilation and removal of blobs
    mask = cv2.inRange(hsv, colourLow, colourHigh) #locates our object in the frame
    mask = cv2.erode(mask, None, iterations=2) #erosion
    mask = cv2.dilate(mask, None, iterations=2) #removal of blobs

    # Will draw outline of ball and find (x, y) center of ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)[-2]   #this makes sure contour will work on all opencv items 
    center = None #make the coords of the ball 0 at first 

    
    if len(cnts) > 0:     # only proceed if at least one contour was found
        # finds largest contour mask, then uses this to get minimum enclosing circle and center
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  #this & above line get centre coords
        

        # only proceed if the radius meets a minimum size
        if (radius > 30): 
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
        
    # update list of points
    pts.appendleft(center)

    # loop over set of points 
    for i in range(1, len(pts)):
        #if we don't have tracked points we should ignore them 
        if pts[i - 1] is None or pts[i] is None:
            continue

        ickk = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

        def drawline(img,pt1,pt2,color,thickness=ickk,style='dotted',gap=20):
            dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
            pts= []
            for i in  np.arange(0,dist,gap):
                r=i/dist
                x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
                y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
                p = (x,y)
                pts.append(p)

            if style=='dotted':
                for p in pts:
                    cv2.circle(img,p,thickness,color,-1)
            else:
                s=pts[0]
                e=pts[0]
                i=0
                for p in pts:
                    s=e
                    e=p
                    if i%2==1:
                        cv2.line(img,s,e,color,thickness)
                    i+=1

        #if we do we will draw point connecting line 
        #gotta define the thickness first
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        drawline(frame,pts[i - 1], pts[i],(0, 0, 255),thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break


# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
