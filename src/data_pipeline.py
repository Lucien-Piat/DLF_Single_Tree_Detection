import tensorflow as tf
import os


def parse_image(file_path):
    # Convert the file path to an image
    image = tf.io.read_file(file_path)
    # Use decode_jpeg for JPEG images
    image = tf.image.decode_png(image, channels=3)
    return image


def process_path(file_path):
    # Create file paths for the mask by replacing 'tiles' with 'masks' in the file path
    mask_path = tf.strings.regex_replace(file_path, 'tiles', 'masks')

    # Load the image and the mask
    image = parse_image(file_path)
    mask = parse_image(mask_path)

    return image, mask


def augment(image, mask):
    # Random rotation
    # Randomly choose a multiple of 90 degrees for rotation to avoid interpolation artifacts
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Random cropping
    concat_image = tf.concat([image, mask], axis=-1)
    concat_image = tf.image.random_crop(concat_image, size=[512, 512, 6])
    image, mask = concat_image[..., :3], concat_image[..., 3:]

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask


def resize_and_normalize(image, mask):
    # Resize images to 256x256
    image = tf.image.resize(
        image, [128, 128]
    )
    mask = tf.image.resize(
        mask, [128, 128],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    # Normalize images
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)
    return image, mask[:, :, 0]


def prepare_dataset(ds, batch_size=32):
    # Shuffle, repeat, and batch the dataset
    ds = ds.shuffle(1000)
    ds = ds.repeat()
    ds = ds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(resize_and_normalize,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def make_dataset(base_dir, batch_size=32):
    # List dataset files
    list_ds = tf.data.Dataset.list_files(
        os.path.join(base_dir, 'tiles', '*.png'), shuffle=False)
    # Map the files to images and masks
    list_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return prepare_dataset(list_ds, batch_size=batch_size)


if __name__ == "__main__":
    # Define the base directory
    base_dir = 'data/gold/train'

    # Create the dataset
    train_dataset = make_dataset(base_dir, batch_size=32)

    # Visualize the dataset
    import matplotlib.pyplot as plt

    for images, masks in train_dataset.take(1):
        plt.figure(figsize=(10, 5))
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.imshow(images[i])
            plt.text(0.5, -0.1, 'Max: {:.4f}'.format(tf.reduce_max(images[i])),
                     size=12, ha="center",
                     transform=plt.gca().transAxes)
            plt.text(0.5, -0.2, 'Min: {:.4f}'.format(tf.reduce_min(images[i])),
                     size=12, ha="center",
                     transform=plt.gca().transAxes)
            plt.axis('off')
            plt.subplot(2, 4, i+5)
            plt.imshow(masks[i], cmap='gray')
            plt.text(0.5, -0.1, 'Max: {:.4f}'.format(tf.reduce_max(masks[i])),
                     size=12, ha="center",
                     transform=plt.gca().transAxes)
            plt.text(0.5, -0.2, 'Min: {:.4f}'.format(tf.reduce_min(masks[i])),
                     size=12, ha="center",
                     transform=plt.gca().transAxes)
            plt.axis('off')
        plt.show()
