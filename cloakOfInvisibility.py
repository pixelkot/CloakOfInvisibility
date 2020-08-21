import cv2
import time
import numpy as np

#Prep for writing output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#Read data from camera
video_capture = cv2.VideoCapture(0)

#Give camera 3 seconds to start up!
time.sleep(3)
count = 0
background = 0

for i in range(30):
    ret, background = video_capture.read() #capturing imag

#Read every frame from the camera while alive
while (True):
    ret, img = video_capture.read()

    img = np.flip(img, axis=1)

    #Convert color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Generate masks to detect red color
    lower_red = np.array([0, 125, 100])
    upper_red = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    #lower_red = np.array([170, 120, 120])
    #upper_red = np.array([180, 255, 255])
    #mask2 = cv2.inRange(hsv, lower_red, upper_red)

    #mask1 = mask1 + mask2

    #Open and dilate the mask image
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    #Create an inverted mask to segment out the red color from the frame
    mask2 = cv2.bitwise_not(mask1)

    #Segment the red color part out of the frame using bitwise and wit hthe inverted mask
    res1 = cv2.bitwise_and(img, img, mask=mask2)

    #Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow('invisibility_cloak', final_output)
    if cv2.waitKey(1) == 27:
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
