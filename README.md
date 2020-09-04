# visual-sudoku-solver

## TODO:
- Remove constants from the code and make it more generic
- Clean up main.py (extract functions to other files, add comments, etc.)
- Remove global variables from main
- Write up this readme (lol)
  - add README to `src/`
  - add some nice pics and gifs :P

## How to install and run
### 1. Make sure you have Python 3.6 or later installed
```bash
python3 --version
```
If not, please install newer version of Python.
### 2. Clone the repo
```bash
$ git clone https://github.com/mackopes/visual-sudoku-solver.git
```
### 3. Go to the cloned directory
``` bash
$ cd visual-sudoku-solver
```
### 4. Install all the requirements
It's prefered to install these in a virtual environment, so you do not clutter your global python environment.
You can install all the requirements by
```bash
$ python3 -m pip install -r requirements.txt
```
### 5. Finally run the code
```
python3 src/main.py
```
This will summon the webcam stream. If you now show a sudoku image to the camera you should see it solved. This can take a couple of seconds depending on the computer you are running this on.

## Code base
### `src/`
`src/` contains all the functional code to the sudoku detection, digit classification and AR functionality.
### `model_data/`
This directory contains data for the trained TensorFlow model which classifies the digits of the sudoku.
### `resources/`
Currently only training and testing datasets for the digit classification model reside here.

## How does it work?
The whole algorithm consists of a couple of steps executed repeatedly in a loop:
1. Capture the image from the webcam
2. Detect the sudoku grid in the image
3. Crop out sudoku grid from the image
4. Dissect the cropped out sudoku into separate cells
5. Run each cell through a classifier to find out the cell’s digit (if any)
6. Solve the sudoku
7. Write all the digits back to the original image
8. Show the image to the user
9. Repeat from step 1!

Simple, eh? Not really as the real code is multithreaded for better and seamless experience, but the overall gist of it stays the same.

Now we can discuss each step in more detail. Note that the following guide does not strictly follow the naming used in the codebase and should be used rather as a guide than the codebase explanation.

### 1. Image capture
I utilise the OpenCV library. To take a photo using the webcam you can simply do
```python
import cv2

cap = cv2.VideoCapture(0)
_, frame = cap.read()
```

### 2. Sudoku grid detection
The overall gist of the sudoku detection is to look for the largest *blob* of connected lines in the image and determine its boundaries.

Sudoku images are usually printed in black on white paper. Therefore in majority of cases any colour information are unnecessary and we can simplify the problem by turning the image into grayscale.
`frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`

As we are interested only contours of the sudoku, we can apply some thresholding. Thresholding turns each pixel into either pure black or pure white depending on whether the value of the pixel is above or below a user specified threshold, thus the name thresholding.
For example the following code assigns value 255 to any pixel greater or equal than the threshold and 0 to anything below the threshold.
`_, frame_threshold = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY)`

Since in the next step we will be looking for **connected** lines (contours) we need to make sure all the lines stay connected even after the thresholding. This can be a problem as lines in sudokus are usually fairly thin and can get disconnected easily when the threshold value is not ideal. We solve this by blurring the sudoku first, thus making all the lines thicker. Another issue that arises is the question on how to actually pick *the best* threshold value. The good news are we do not have to if we use *adaptive thresholding* where the value is automatically calculated based on a small regions of the image.

```
frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
frame_threshold = cv2.adaptiveThreshold(frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
```

To find more about thresholding functions, refer to [OpenCV: Image Thresholding](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)

Now onto localising *the biggest blob* in the image. For this OpenCV comes to the rescue once again! We simply determine all the contours (“curve joining all the continuous points (along the boundary), having same colour or intensity”) in the image.

`contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`

then find the largest one
`largest_contour = max(contours, key=cv2.contourArea)`
and then get all the points. Here we assume that sudoku has a shape of rhomboid. The following code will go without explanation.
```python
bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
```

To learn more about contours in OpenCV, definitely check out [OpenCV: Contours : Getting Started](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html).

Note that in the code I have done certain optimisations, such as resizing the original image to a smaller size (as working with smaller image is faster), obtaining a *rough estimate* of where the sudoku probably is and then refining this estimate by running the same algorithms on full-size image, but cropped according to the estimate. Therefore I never run the algorithm on the complete full-size image.
### 3. Sudoku crop
Here I used a simple perspective transformation.
```python
m = cv2.getPerspectiveTransform(source_polygon, target_polygon)
frame_crop = cv2.warpPerspective(frame, m, (size, size))
```

More information and better examples can be found [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#perspective-transformation)

### 4. Extract cells from the crop
There is no compvision magic to get each cell of the image. As I already transformed possibly skewed and warped sudoku image into a perfect square, I can just split the whole frame into 81 equal sized squares and call it a day.

```python
cell_size = img.shape[0] / 9, img.shape[1] / 9
for i in range(9):
    for j in range(9):
        cell = img[int(i * cell_size[1]):int((i + 1) * cell_size[1]), int(j * cell_size[0]):int((j + 1) * cell_size[0])]
```

This step has certain issues and very often grid itself will appear in the cells, but this will be handled in the next step.

### 5. Classify digit in a cell
This can be handled by else than the buzzwordy (buzzworthy?) machine learning.
For those unfamiliar with ML (and TensorFlow) I recommend reading up on [TensorFlow 2 quickstart for experts  |  TensorFlow Core](https://www.tensorflow.org/tutorials/quickstart/advanced).

I won’t go into too much detail on how this is done as I could write a couple of blogposts just on this topic and there are countless tutorials online anyways.

But for those interested, the model consists of two convolutional layers followed by two fully-connected layers intercalated with dropout layers to cope with overfitting.
### 6. How to solve the sudoku itself?
Here refer to my actual sudoku solver repo [GitHub - mackopes/sudoku-solver: A simple sudoku solver](https://github.com/mackopes/sudoku-solver). I promise the code description is coming up!
### 7. Write the numbers back to the image for the cool AR effect
Here three OpenCV functions are used:
[`cv2.rectangle`](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#rectangle), [`cv2.line`](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#line) and [`cv2.putText`](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#puttext). This part is left as an exercise to the reader.
### 8. Show the image to the user
```python
cv2.imshow('frame', frame)
```
lol, what did you expect?

## Limitations and future work
* On slower computers the delay between showing the sudoku to the camera and getting back the AR result can be noticable. Therefore some optimisations could be done there
* Better digit classification. One way would be by extending the dateset itself to containg more cases, thus making the model more generic. And the other approach would be cleaning up the data fed to the model themselves. Currently the biggest issue are the grid lines that are still visible in extracted sudoku cells.
* Using faster sudoku solver. This was a weekend project and I wanted to use my own sudoku solver for this, but it might not have been the best idea performance wise.