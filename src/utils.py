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
