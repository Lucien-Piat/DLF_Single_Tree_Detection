from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Get image
    image_dir = 'data/gold/valid'
    image_files = os.listdir(image_dir)
    image_files = [file for file in image_files if file.endswith('.jpg')]
    image_files.sort()
    image_path = os.path.join(image_dir, image_files[0])

    # Load the model
    model = YOLO('runs/segment/train/weights/best.pt')

    # Run the model on a single image
    results = model(image_path)

    # Display the results
    results[0].show()
