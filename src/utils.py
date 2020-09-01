import cv2
import numpy as np
from enum import Enum


class Color(Enum):
    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)


def show_image(image, delay=10000):
    cv2.namedWindow("macka", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("macka", 800, 800)
    cv2.imshow("macka", image)

    cv2.waitKey(delay)


def crop_out_polygon(img, polygon, maxsize=361):
    tl, tr, bl, br = polygon

    side_size = max(np.linalg.norm(tl - tr),
                    np.linalg.norm(tr - br),
                    np.linalg.norm(br - bl),
                    np.linalg.norm(bl - tl),
                    maxsize)

    target_polygon = np.array([[0, 0], [side_size - 1, 0], [side_size - 1, side_size - 1], [0, side_size - 1]], dtype='float32')

    m = cv2.getPerspectiveTransform(np.array(polygon, dtype='float32'), target_polygon)

    return cv2.warpPerspective(img, m, (int(side_size), int(side_size)))


def overlay_images(background, foreground, alpha):
    foreground = foreground.astype(float) / 255
    background = background.astype(float) / 255
    alpha = alpha.astype(float) / 255

    assert foreground.shape == alpha.shape

    dim_diff = [b - a for b, a in zip(background.shape, alpha.shape)]

    foreground = np.pad(foreground, [(0, max(diff, 0)) for diff in dim_diff])
    alpha = np.pad(alpha, [(0, max(diff, 0)) for diff in dim_diff])

    bck_shp = background.shape
    foreground = foreground[:bck_shp[0], :bck_shp[1], :bck_shp[0]]
    alpha = alpha[:bck_shp[0], :bck_shp[1], :bck_shp[0]]

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)

    result = cv2.add(foreground, background)

    return result


def sudoku_grid():
    img = np.zeros((800, 800, 3), np.uint8)
    alpha = np.zeros((800, 800, 3), np.uint8)

    cv2.rectangle(img, (1, 1), (800, 800), Color.BLUE, 5)
    cv2.rectangle(alpha, (1, 1), (800, 800), Color.WHITE, 5)

    cv2.line(img, (0, 266), (800, 266), Color.RED, 3)
    cv2.line(alpha, (0, 266), (800, 266), Color.WHITE, 3)

    cv2.line(img, (0, 533), (800, 533), Color.RED, 3)
    cv2.line(alpha, (0, 533), (800, 533), Color.WHITE, 3)

    cv2.line(img, (266, 0), (266, 800), Color.RED, 3)
    cv2.line(alpha, (266, 0), (266, 800), Color.WHITE, 3)

    cv2.line(img, (533, 0), (533, 800), Color.RED, 3)
    cv2.line(alpha, (533, 0), (533, 800), Color.WHITE, 3)

    return img, alpha


def draw_numbers_in_sudoku_grid(img, sudoku, font_color):
    cell_size = img.shape[0] / 9, img.shape[1] / 9

    img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # TODO: make this variable of cell_size
    font_scale = 2
    lineType = 6

    for i, row in enumerate(sudoku):
        for j, n in enumerate(row):
            if n != 0:
                cv2.putText(img, str(n),
                            (int((j + 0.3) * cell_size[1]), int((i + 0.8) * cell_size[0])),
                            font,
                            font_scale,
                            font_color,
                            lineType)

    return img
