import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from itertools import product

from sklearn.model_selection import train_test_split

from src.utils import parse_number_string
from src.segmentation import label_image as separate_mask_image, load_image

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

    os.makedirs("data/silver/masks", exist_ok=True)
    for mask_path, mask_image in tqdm(list(zip(mask_paths, mask_image_arrays))):
        idx = parse_number_string(mask_path)
        image_path = f"data/silver/masks/{idx}.png"

        if os.path.exists(image_path):
            continue

        mask_image_separated = separate_mask_image(mask_image).astype(np.uint8)
        mask_image_separated = Image.fromarray(mask_image_separated)

        mask_image_separated.save(image_path)

    print("\nSaving tiles as [InfRed, Green, Blue]...")
    os.makedirs("data/silver/tiles", exist_ok=True)
    for tile_path, tile_image in tqdm(list(zip(tile_paths, tile_image_arrays))):
        idx = parse_number_string(tile_path)
        image_path = f"data/silver/tiles/{idx}.png"

        if os.path.exists(image_path):
            continue

        tile_image = Image.fromarray(tile_image[:, :, [3, 1, 2]])
        tile_image.save(image_path)

    print("\nCreating train, validation, and test directories...")
    for subset_name in ["train", "test", "valid"]:
        for folder_name in ["tiles", "masks"]:
            os.makedirs(
                f"data/gold/{subset_name}/{folder_name}", exist_ok=True)

    print("\nSplitting data into train, validation, and test sets...")

    train_paths, test_paths = train_test_split(
        list(zip(tile_paths, mask_paths)), test_size=0.4, random_state=RANDOM_SEED)
    valid_paths, test_paths = train_test_split(
        test_paths, test_size=0.5, random_state=RANDOM_SEED)

    print(f"Train: {len(train_paths)} | Validation: {
          len(valid_paths)} | Test: {len(test_paths)}")

    print("\nCopying images to train, validation, and test directories...")
    for subset_name, paths in zip(["train", "valid", "test"], [train_paths, valid_paths, test_paths]):
        for tile_path, mask_path in tqdm(paths):
            idx = parse_number_string(tile_path)
            for folder_name in ["tiles", "masks"]:
                from_path = f"data/silver/{folder_name}/{idx}.png"
                to_path = f"data/gold/{subset_name}/{folder_name}/{idx}.png"

                os.system(f"cp {from_path} {to_path}")
