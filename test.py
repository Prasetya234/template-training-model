from ultralytics import YOLO
# from PIL import Image
import cv2
model = YOLO('datasets/yolov8n.pt')
SOURCE='test1.jpeg'

results = model(
   source=SOURCE,
   show=True
)
cv2.waitKey(0)