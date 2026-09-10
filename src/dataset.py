"""
Builds a tf.data.Dataset that reads images from disk lazily, by filepath,
instead of loading the whole dataset into memory. Necessary at this
dataset's size (130k+ 96x96x3 images -- multiple GB as raw arrays).

Note on TIFF: the real dataset ships as .tif, which tf.io.decode_image
does NOT support natively (only PNG/JPEG/GIF/BMP). We decode via PIL
inside a tf.numpy_function instead of quietly assuming decode_image will
work -- a common silent-failure point when people copy a generic
tf.data image pipeline onto this specific dataset.
"""

import numpy as np
import tensorflow as tf


def _load_image_numpy(filepath, image_size):
    from PIL import Image

    with Image.open(filepath.numpy().decode("utf-8")) as img:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size))
        return np.array(img, dtype=np.uint8)


def _load_image_tf(filepath, label, image_size):
    image = tf.py_function(
        func=lambda fp: _load_image_numpy(fp, image_size), inp=[filepath], Tout=tf.uint8
    )
    image.set_shape((image_size, image_size, 3))
    return tf.cast(image, tf.float32), label


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def build_tf_dataset(
    filepaths,
    labels,
    image_size: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((list(filepaths), list(labels)))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 10000), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda fp, lbl: _load_image_tf(fp, lbl, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
