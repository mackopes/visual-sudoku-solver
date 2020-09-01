import cv2
import tensorflow as tf
from src.utils import show_image
import os
import numpy as np
from itertools import product, count
from collections import defaultdict


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

    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    # _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    # image = cv2.bitwise_not(image)
    # yesno = [True, False]

    # h, w = image.shape

    # images = []
    # for t, b, l, r in product(*[yesno for _ in range(4)]):
    # image_aug = image.copy()
    # if t:
    #     cv2.line(image_aug, (h, 0), (h, w), 255, 8)
    # if b:
    #     cv2.line(image_aug, (0, 0), (0, w), 255, 8)
    # if l:
    #     cv2.line(image_aug, (0, 0), (h, 0), 255, 8)
    # if r:
    #     cv2.line(image_aug, (h, 0), (h, w), 255, 8)
    # # cv2.line(image_aug, (h, 0), (h, w), 0, 4)
    # cv2.line(image_aug, (0, 0), (0, w), 0, 4)
    # cv2.line(image_aug, (0, 0), (h, 0), 0, 4)
    # cv2.line(image_aug, (0, w), (h, w), 0, 4)
    # images.append(image_aug)

    # show_image(image_aug)

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


# IMG_SIZE = (40, 40)
# TRAIN_DS_DIR = "resources/sudoku-image-solver/ocr_data/training/"
# TEST_DS_DIR = "resources/sudoku-image-solver/ocr_data/testing/"
# DST = 'resources/custom_testing_ds/'


# train_x, train_y = load_images_from_folder(TRAIN_DS_DIR, IMG_SIZE)
# areas = defaultdict(float)
# counts = defaultdict(int)

# for x, y in zip(train_x, train_y):
#     middle_crop = x[10:30, 10:30]
#     areas[y] += np.sum(middle_crop) / 255.0
#     counts[y] += 1

# for k, v in areas.items():
#     print(k, v / counts[k])

    # test_x, test_y = load_images_from_folder(TEST_DS_DIR, IMG_SIZE)

    # for i, img, label in zip(count(), test_x, test_y):
    #     cv2.imwrite(f'{DST}{label}_{i}_dl.jpg', img)
