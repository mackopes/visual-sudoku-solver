import cv2
import numpy as np


def show_image(image, delay=10000):
    cv2.namedWindow("macka", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("macka", 800, 800)
    cv2.imshow("macka", image)

    cv2.waitKey(delay)


def crop_out_polygon(img, polygon):
    tl, tr, bl, br = polygon

    side_size = max(np.linalg.norm(tl - tr),
                    np.linalg.norm(tr - br),
                    np.linalg.norm(br - bl),
                    np.linalg.norm(bl - tl))

    target_polygon = np.array([[0, 0], [side_size - 1, 0], [side_size - 1, side_size - 1], [0, side_size - 1]], dtype='float32')

    m = cv2.getPerspectiveTransform(np.array(polygon, dtype='float32'), target_polygon)

    return cv2.warpPerspective(img, m, (int(side_size), int(side_size)))
