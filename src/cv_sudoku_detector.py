import cv2
import numpy as np
import operator
from utils import crop_out_polygon

RESIZE_FACTOR = 4


def find_sudoku(sudoku_image):
    initial_polygon = _find_sudoku(sudoku_image, RESIZE_FACTOR)
    tl, tr, br, bl = initial_polygon

    offset = 15
    top = min(tl[1], tr[1]) - offset
    bottom = max(bl[1], br[1]) + offset
    left = min(tl[0], bl[0]) - offset
    right = max(tr[0], br[0]) + offset

    # bounding_box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])

    if abs((bottom - top) - (right - left)) > 0.25 * (bottom - top):
        return None

    crop = sudoku_image[top:bottom, left:right]
    polygon_refined = _find_sudoku(crop)

    return polygon_refined + np.array([left, top])


def _find_sudoku(sudoku_image, scale=1):
    sudoku_small = cv2.resize(sudoku_image, (0, 0), fx=1 / scale, fy=1 / scale)

    sudoku_gray = cv2.cvtColor(sudoku_small, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(sudoku_gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

    kernel = np.ones((5, 5), np.uint8)
    not_threshold = cv2.bitwise_not(threshold)
    # erode = cv2.erode(not_threshold, kernel)
    dilate = cv2.dilate(not_threshold, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    # largest_contour_image = cv2.drawContours(sudoku_image, largest_contour, -1, (0, 0, 255), 2)

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    points = np.array([largest_contour[top_left][0], largest_contour[top_right][0], largest_contour[bottom_right][0], largest_contour[bottom_left][0]])

    points = points * scale

    return points
