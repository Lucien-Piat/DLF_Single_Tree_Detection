from ultralytics import YOLO


if __name__ == "__main__":
    # Load the model
    model = YOLO('model/yolov8n-seg.pt')

    # Train the model
    results = model.train(data='data/gold/config.yaml',
                          epochs=100, imgsz=256)

    # Save the model
    model.save('model/yolov8n-seg-trained.pt')
