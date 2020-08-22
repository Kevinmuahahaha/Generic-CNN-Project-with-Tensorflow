# Generic-CNN-Project-with-Tensorflow
Classify images using CNN module from tensorflow.

# Project setup
## Install dependencies
1. install python & tensorflow
  - beware of GPU driver's version & tensorflow's version.
  - **do make sure softwares are compatible**. See https://www.tensorflow.org/install/source#gpu
2. and everything else
  - Pillow
  - matplotlib
  - numpy
## Running
$ python3 /path/to/classes.py

with tkinter, the GUI should look something like this:

![Image of the interface](https://github.com/Kevinmuahahaha/Generic-CNN-Project-with-Tensorflow/blob/master/assets/demo_interface.png)



# Usage
## Prepare your dataset
1. Each folder should contain multiple images.
2. Each folder represents 1 class.
3. Each folder's name will be used as class name.

## Training
Simply select "Train" and click "Run".

Wait till the training is over.

![Image of training session.](https://github.com/Kevinmuahahaha/Generic-CNN-Project-with-Tensorflow/blob/master/assets/demo_training.png)

## Predicting
Select target directory containing images.

Click "Test" and then "Run".

Predictions are shown in the message box(top left).

![Image of predicting](https://github.com/Kevinmuahahaha/Generic-CNN-Project-with-Tensorflow/blob/master/assets/demo_predicting.png)

# Limitations
- Images are not scaled dynamically. Sampling height/width should equal to that of training.
- Images are not cleaned, anything from the background are used in training.
- "Training Checkpoints" not yet implemented, beware of machine shutting down during training.
