import cv2
import numpy as np
import operator


def find_sudoku(sudoku_image):
    sudoku_gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)

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
    points = [largest_contour[top_left][0], largest_contour[top_right][0], largest_contour[bottom_right][0], largest_contour[bottom_left][0]]

    return points
