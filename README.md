# visual-sudoku-solver

## TODO:
- Remove constants from the code and make it more generic
- Clean up main.py (extract functions to other files, add comments, etc.)
- Remove global variables from main
- Write up this readme (lol)
  - add README to `src/`
  - description of the algorithm
  - what are the limitations
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
TODO

## Limitations and future work
TODO