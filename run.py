from ultralytics import YOLO

if __name__ == "__main__":
    # Load the model
    model = YOLO('runs/segment/train/weights/best.pt')

    # Run the model on a single image
    results = model('data/gold/valid/0__  1_102_    1_    1.jpg')

    # Display the results
    results[0].show()
