import cv2
from ultralytics import YOLO

#create a videoCapture Object (this allow to read frames one by one)
video = cv2.VideoCapture("video_test/test_2.mp4")

model = YOLO('datasets/yolov8m.pt')

#check it's ok
if video.isOpened():
    print('Video Succefully opened')
else:
    print('Something went wrong check if the video name and path is correct')

windowName = 'Video Reproducer'
cv2.namedWindow(windowName)

#let's reproduce the video
while True:
    ret,frame = video.read() #read a single frame
    if not ret: #this mean it could not read the frame
         print("Could not read the frame")
         cv2.destroyWindow(windowName)
         break

    results = model(
        source=frame,
        show=True
    )

    waitKey = (cv2.waitKey(1) & 0xFF)
    if  waitKey == ord('q'): #if Q pressed you could do something else with other keypress
         print("closing video and exiting")
         cv2.destroyWindow(windowName)
         video.release()
         break
