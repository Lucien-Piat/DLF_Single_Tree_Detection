import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from itertools import product

from sklearn.model_selection import train_test_split

from src.edge_detection import get_yolo_items
from src.augmentation import augment_image
from src.utils import parse_number_string
from src.segmentation import label_image as separate_mask_image, load_image

IMAGE_SIZE = (256, 256)

if __name__ == "__main__":
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    tiles_dir = "data/bronze/tiles"
    masks_dir = "data/bronze/masks"

    tile_paths = sorted(os.listdir(tiles_dir))
    mask_paths = sorted(os.listdir(masks_dir))

    print("\nLoading images...")
    tile_image_arrays = [load_image(tiles_dir, tile_path)
                         for tile_path in tqdm(tile_paths, desc="Tiles")]
    mask_image_arrays = [load_image(masks_dir, mask_path)
                         for mask_path in tqdm(mask_paths, desc="Masks")]

    print("\nSeparating mask images...")
    separated_mask_images = []

    os.makedirs("data/silver/masks", exist_ok=True)
    for mask_path, mask_image in tqdm(list(zip(mask_paths, mask_image_arrays))):
        idx = parse_number_string(mask_path)
        image_path = f"data/silver/masks/{idx}.png"

        if os.path.exists(image_path):
            separated_mask_images.append(
                Image.open(image_path).convert("L"))
            continue

        mask_image_separated = separate_mask_image(mask_image).astype(np.uint8)
        mask_image_separated = Image.fromarray(mask_image_separated)

        separated_mask_images.append(mask_image_separated)
        mask_image_separated.save(image_path)

    print("\nSaving tiles as [InfRed, Green, Blue]...")
    os.makedirs("data/silver/tiles", exist_ok=True)
    for tile_path, tile_image in tqdm(list(zip(tile_paths, tile_image_arrays))):
        tile_image = Image.fromarray(tile_image[:, :, [3, 1, 2]])
        idx = parse_number_string(tile_path)
        image_path = f"data/silver/tiles/{idx}.png"
        tile_image.save(image_path)

    quit()

    print("\nMaking sub-tiles...")
    os.makedirs("data/gold/tiles", exist_ok=True)
    os.makedirs("data/gold/masks", exist_ok=True)

    for idx in tqdm(range(len(tile_image_arrays)), desc="Sub-tiles"):
        tile_image = Image.fromarray(tile_image_arrays[idx][:, :, [3, 1, 2]])
        mask_image = separated_mask_images[idx]

        zoom_factors = [1, .25, .5, .75]
        rotate_angles = np.random.randint(0, 360, 4)
        flip_horizontal = [False, True]
        flip_vertical = [False, True]

        for zoom, rotation, flip_horizontal, flip_vertical in product(
                zoom_factors, rotate_angles, flip_horizontal, flip_vertical):

            tile_image_augmented = augment_image(
                tile_image, IMAGE_SIZE,
                zoom=zoom,
                rotation=rotation,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical
            )
            mask_image_augmented = augment_image(
                mask_image, IMAGE_SIZE,
                zoom=zoom,
                rotation=rotation,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical
            )

            image_name = f"{idx}__{zoom:3}_{rotation:3}_{
                flip_horizontal:5}_{flip_vertical:5}.jpg"

            tile_image_augmented.save(f"data/gold/tiles/{image_name}")
            mask_image_augmented.save(f"data/gold/masks/{image_name}")

    print("\nMaking bounding boxes...")
    os.makedirs("data/gold/segmentations", exist_ok=True)

    for img_name in tqdm(os.listdir("data/gold/masks"), desc="Bounding boxes"):
        mask_image = load_image("data/gold/masks", img_name)

        yolo_text = get_yolo_items(mask_image)

        yolo_file = img_name.replace(".jpg", ".txt")
        with open(f"data/gold/segmentations/{yolo_file}", "w+") as f:
            f.write(yolo_text)

    print("\nReorganizing dataset...")
    os.makedirs("data/gold/train", exist_ok=True)
    os.makedirs("data/gold/valid", exist_ok=True)

    train, valid = train_test_split(
        os.listdir("data/gold/tiles"), test_size=.2, random_state=RANDOM_SEED)

    for img_name in tqdm(train, desc="Train images"):
        filename = img_name.replace(".jpg", "")
        os.rename(f"data/gold/tiles/{filename}.jpg",
                  f"data/gold/train/{filename}.jpg")
        os.rename(f"data/gold/segmentations/{filename}.txt",
                  f"data/gold/train/{filename}.txt")

    for img_name in tqdm(valid, desc="Valid images"):
        filename = img_name.replace(".jpg", "")
        os.rename(f"data/gold/tiles/{filename}.jpg",
                  f"data/gold/valid/{filename}.jpg")
        os.rename(f"data/gold/segmentations/{filename}.txt",
                  f"data/gold/valid/{filename}.txt")

    print("\nMaking config.yaml for dataset...")
    with open("data/gold/config.yaml", "w+") as f:
        contents = (
            f"""
path: {os.getcwd()}/data/gold
train: ./train
val: ./valid

names:
  0: tree
"""[1:-1])
        f.write(contents)

    print("\nDeleting temporary files...")
    os.rmdir("data/gold/tiles")
    os.rmdir("data/gold/segmentations")
    for mask in os.listdir("data/gold/masks"):
        os.remove(f"data/gold/masks/{mask}")
    os.rmdir("data/gold/masks")
