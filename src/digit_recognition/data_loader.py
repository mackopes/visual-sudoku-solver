import cv2
import tensorflow as tf
import os
import numpy as np


def load_images_from_folder(folder, size=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if not filename.endswith(".jpg"):
            continue

        cur_images = load_image(os.path.join(folder, filename), size)
        cur_labels = [int(filename.split("_")[0])] * len(cur_images)
        images.extend(cur_images)
        labels.extend(cur_labels)
    return np.array(images), np.array(labels)


def load_image(img_path, size=None):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    if size:
        image = cv2.resize(image, size)

    return [image]


def load_ds(train_ds_dir, test_ds_dir, size=None):
    train_x, train_y = load_images_from_folder(train_ds_dir, size)
    train_x = train_x / 255.0
    train_x = train_x[..., tf.newaxis].astype("float32")
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000).batch(32)

    test_x, test_y = load_images_from_folder(train_ds_dir, size)
    test_x = test_x / 255.0
    test_x = test_x[..., tf.newaxis].astype("float32")
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)

    return train_ds, test_ds
