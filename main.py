from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
results = model.train(data="./yolo_dataset/qaisc.yaml", epochs=1000, imgsz=640)
