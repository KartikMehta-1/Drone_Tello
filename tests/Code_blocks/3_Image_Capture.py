from djitellopy import tello
from time import sleep
import time
import cv2

me = tello.Tello()
me.connect()

print(me.get_battery())

me.streamon()  

#Allow some time for the stream to start
time.sleep(2)

while True:
    img = me.get_frame_read().frame #get frame from drone
    img = cv2.resize(img,(360,240)) #resize image using cv2 library
    cv2.imshow("Image",img) #display image in window
    cv2.waitKey(1) #delay between frames 1ms

