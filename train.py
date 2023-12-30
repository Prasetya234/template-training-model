from ultralytics import YOLO
MODEL = 'datasets/yolov8m.pt'
model = YOLO(MODEL)
model.train(data='data.yaml', epochs=38, imgsz=640)
# model(SOURCE)

print("Print success")