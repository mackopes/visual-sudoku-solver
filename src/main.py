import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf

from utils import crop_out_polygon, show_image, overlay_images
from cv_sudoku_detector import find_sudoku


SUDOKU_IMAGES = "../resources/sudoku-image-solver/sudoku_images"
TEST_SUDOKU_IMAGE = SUDOKU_IMAGES + "/sudoku12.jpg"

model_file = Path("model_data")


def sudoku_grid():
    img = np.zeros((800, 800, 3), np.uint8)
    alpha = np.zeros((800, 800, 3), np.uint8)

    cv2.rectangle(img, (1, 1), (800, 800), (255, 0, 0), 5)
    cv2.rectangle(alpha, (1, 1), (800, 800), (255, 255, 255), 5)

    # cv2.line(img, (0, 266), (800, 266), (0, 0, 255), 3)
    # cv2.line(alpha, (0, 266), (800, 266), (0, 0, 255), 3)

    # cv2.line(img, (0, 533), (800, 533), (0, 0, 255), 3)
    # cv2.line(alpha, (0, 533), (800, 533), (0, 0, 255), 3)

    # cv2.line(img, (266, 0), (266, 800), (0, 0, 255), 3)
    # cv2.line(alpha, (266, 0), (266, 800), (0, 0, 255), 3)

    # cv2.line(img, (533, 0), (533, 800), (0, 0, 255), 3)
    # cv2.line(alpha, (533, 0), (533, 800), (0, 0, 255), 3)

    return img, alpha


def draw_numbers_in_grid(img, numbers, font_color):
    cell_size = img.shape[0] / 9, img.shape[1] / 9

    img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    lineType = 2

    for i, row in enumerate(numbers):
        for j, n in enumerate(row):

            cv2.putText(img, str(n),
                        (int((j + 0.3) * cell_size[1]), int((i + 0.8) * cell_size[0])),
                        font,
                        font_scale,
                        font_color,
                        lineType)

    return img


random_sudoku = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 ]


model = tf.keras.models.load_model(model_file)


def collect_cells(img):
    if img is None:
        return []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    img = cv2.bitwise_not(img)
    img = img.astype("float32") / 255

    cell_size = img.shape[0] / 9, img.shape[1] / 9

    cells_list = []
    for i in range(9):
        for j in range(9):
            cell = img[int(i * cell_size[1]):int((i + 1) * cell_size[1]), int(j * cell_size[0]):int((j + 1) * cell_size[0])]
            cell = cv2.resize(cell, (40, 40))
            cells_list.append(cell)

    return cells_list


def classify_sudoku(img):
    if img is None:
        return []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    img = cv2.bitwise_not(img)
    img = img.astype("float32") / 255

    cell_size = img.shape[0] / 9, img.shape[1] / 9

    grid = [[None for _ in range(9)] for _ in range(9)]

    cells_list = []
    for i in range(9):
        for j in range(9):
            cell = img[int(i * cell_size[1]):int((i + 1) * cell_size[1]), int(j * cell_size[0]):int((j + 1) * cell_size[0])]
            cell = cv2.resize(cell, (40, 40))
            # cv2.line(cell, (40, 0), (40, 40), 0, 4)
            # cv2.line(cell, (0, 0), (0, 40), 0, 4)
            # cv2.line(cell, (0, 0), (40, 0), 0, 4)
            # cv2.line(cell, (0, 40), (40, 40), 0, 4)
            cells_list.append(cell)
            cell = cell[tf.newaxis, ..., tf.newaxis]
            grid[i][j] = np.argmax(model(cell))

    cv2.imshow('frame2', np.concatenate(cells_list, axis=1))

    return grid


def main():
    saved_cells = 0
    # sudoku_image = cv2.imread(TEST_SUDOKU_IMAGE)
    # show_image(sudoku_image)

    cap = cv2.VideoCapture(0)

    # polygon = find_sudoku(sudoku_image)

    # cropped_sudoku = crop_out_polygon(np.copy(sudoku_image), polygon)

    # show_image(cropped_sudoku)

    overlay_img, overlay_alpha = sudoku_grid()
    # draw_numbers_in_grid(overlay_img, random_sudoku, (0, 255, 0))
    # draw_numbers_in_grid(overlay_alpha, random_sudoku, (255, 255, 255))

    src_polygon = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype='float32')
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        polygon = find_sudoku(frame)
        if polygon is None:
            crop = None
        else:
            crop = crop_out_polygon(frame, polygon)

        sudoku_classified = classify_sudoku(crop)

        i = draw_numbers_in_grid(overlay_img, sudoku_classified, (0, 255, 0))
        a = draw_numbers_in_grid(overlay_img, sudoku_classified, (255, 255, 255))

        m = cv2.getPerspectiveTransform(src_polygon, np.array(polygon, dtype='float32'))
        i = cv2.warpPerspective(i, m, (1280, 700))
        a = cv2.warpPerspective(a, m, (1280, 700))

        img = overlay_images(frame, i, a)

        # Display the resulting frame
        cv2.imshow('frame', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('a'):
            cells = collect_cells(crop)
            for cell in cells:
                print(f"saving macka_{saved_cells}.jpg")
                cv2.imwrite(f'resources/training_ds/macka_{saved_cells}.jpg', cell * 255)
                saved_cells += 1


if __name__ == '__main__':
    main()
