import cv2
import numpy as np

from utils import crop_out_polygon, show_image
from cv_sudoku_detector import find_sudoku


SUDOKU_IMAGES = "../resources/sudoku-image-solver/sudoku_images"
TEST_SUDOKU_IMAGE = SUDOKU_IMAGES + "/sudoku12.jpg"


def main():
    # sudoku_image = cv2.imread(TEST_SUDOKU_IMAGE)
    # show_image(sudoku_image)

    cap = cv2.VideoCapture(0)

    # polygon = find_sudoku(sudoku_image)

    # cropped_sudoku = crop_out_polygon(np.copy(sudoku_image), polygon)

    # show_image(cropped_sudoku)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        polygon = find_sudoku(frame)

        if polygon is None:
            crop = frame
        else:
            crop = crop_out_polygon(frame, polygon)

        # Display the resulting frame
        cv2.imshow('frame', crop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
