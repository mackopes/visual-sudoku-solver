import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from copy import deepcopy
import threading
import time

from utils import crop_out_polygon, overlay_images, sudoku_grid, draw_numbers_in_sudoku_grid, Color
from cv_sudoku_detector import find_sudoku
from webcam_stream import WebcamStream
from sudoku_solver.sudoku import Sudoku

# TODO: do not use global variables
model_file = Path("model_data")
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

    img = cv2.resize(img, (360, 360))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    img = cv2.bitwise_not(img)
    img = img.astype("float32") / 255

    cell_size = img.shape[0] / 9, img.shape[1] / 9

    assert cell_size == (40, 40)
    grid = [[None for _ in range(9)] for _ in range(9)]

    for i in range(9):
        for j in range(9):
            cell = img[int(i * cell_size[1]):int((i + 1) * cell_size[1]), int(j * cell_size[0]):int((j + 1) * cell_size[0])]
            assert cell.shape == (40, 40)
            if np.sum(cell[10:30, 10:30]) > 310:
                grid[i][j] = np.array([1], [0], [0], [0], [0], [0], [0], [0], [0], [0])
            else:
                cell = cell[tf.newaxis, ..., tf.newaxis]
                grid[i][j] = model(cell)

    return np.array(grid)


def get_sudoku_solving_func_with_cache():
    def solve_sudoku(sudoku):
        similar_count = 0
        if solve_sudoku.prev_sudoku_to_solve is not None:
            for row_c, row_p in zip(sudoku, solve_sudoku.prev_sudoku_to_solve):
                for elem_c, elem_p in zip(row_c, row_p):
                    if elem_p == elem_c and elem_p != 0:
                        similar_count += 1
            if similar_count > 20:
                return solve_sudoku.prev_result_to_sudoku

        sudoku_copy = deepcopy(sudoku)
        sudoku = Sudoku(sudoku)

        t = threading.Thread(target=sudoku.solve)
        t.start()

        t.join(timeout=1)

        if t.is_alive():
            return sudoku_copy
        else:
            solve_sudoku.prev_sudoku_to_solve = sudoku_copy
            solve_sudoku.prev_result_to_sudoku = sudoku.grid
            return sudoku.grid

    solve_sudoku.prev_sudoku_to_solve = None
    solve_sudoku.prev_result_to_sudoku = None

    return solve_sudoku


def show_frame_and_wait_key(frame):
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        return True

    return False


# TODO: do not use global variables for this
src_polygon = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype='float32')
overlay_img, overlay_alpha = sudoku_grid()

# coputed overlay for the sudoku (image and alpha)
ready_i, ready_a = None, None

# cropped out sudoku.
# useful for constructing training set
ready_crop = None


def compute_and_draw_sudoku(frame, solve_sudoku):
    global ready_i, ready_a, ready_crop
    try:
        polygon = find_sudoku(frame)
    except:
        ready_i, ready_a = None, None
        return

    if polygon is None:
        ready_i, ready_a = None, None
        return

    crop = crop_out_polygon(frame, polygon)
    ready_crop = crop

    sudoku_proc = classify_sudoku(crop)
    sudoku_classified = np.argmax(sudoku_proc, axis=3).reshape((9, 9))

    solved_sudoku = solve_sudoku(sudoku_classified.tolist())

    i = draw_numbers_in_sudoku_grid(overlay_img, solved_sudoku, Color.GREEN)
    a = draw_numbers_in_sudoku_grid(overlay_img, solved_sudoku, Color.WHITE)

    m = cv2.getPerspectiveTransform(src_polygon, np.array(polygon, dtype='float32'))
    i = cv2.warpPerspective(i, m, (1280, 700))
    a = cv2.warpPerspective(a, m, (1280, 700))

    ready_i, ready_a = i, a


def main():
    saved_cells = 0
    webcam = WebcamStream()
    solve_sudoku = get_sudoku_solving_func_with_cache()

    webcam.start()
    sudoku_computing_thread = None
    i, a = None, None
    while True:
        frame = webcam.read()

        if sudoku_computing_thread is None or not sudoku_computing_thread.is_alive():
            # get the previous result
            i, a = ready_i, ready_a
            sudoku_computing_thread = threading.Thread(target=compute_and_draw_sudoku, args=[frame, solve_sudoku])
            sudoku_computing_thread.start()

        if i is not None and a is not None:
            frame = overlay_images(frame, i, a)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('a') and ready_crop is not None:
            cells = collect_cells(ready_crop)
            for cell in cells:
                label = np.argmax(model(cell[tf.newaxis, ..., tf.newaxis]))
                print(f"saving {label}_{3}_{saved_cells}.jpg")
                cv2.imwrite(f'resources/training_ds/{label}_{saved_cells}.jpg', cell * 255)
                saved_cells += 1

        time.sleep(1 / 30)

    if sudoku_computing_thread is not None:
        sudoku_computing_thread.join()

    webcam.stop()

    print("FPS:", webcam.fps())


if __name__ == '__main__':
    main()
