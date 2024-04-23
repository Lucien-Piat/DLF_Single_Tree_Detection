from ultralytics import YOLO
from setup import IMAGE_SIZE


if __name__ == "__main__":
    # Load the model
    model = YOLO('model/yolov8n-seg.pt', task='segment')

    # Train the model
    results = model.train(data='data/gold/config.yaml',
                          epochs=100, imgsz=IMAGE_SIZE[0])
