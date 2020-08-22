# Generic-CNN-Project-with-Tensorflow
Classify images using CNN module from tensorflow.

# Usage
## Prepare your dataset
1. Each folder should contain multiple images.
2. Each folder represents 1 class.
3. Each folder's name will be used as class name.

## Training
Simply select "Train" and click "Run".

Wait till the training is over.

## Predicting
Select target directory containing images.

Click "Test" and then "Run".

Predictions are shown in the message box(top left).

# Limitations
- Images are not scaled dynamically. Sampling height/width should equal to that of training.
- Images are not cleaned, anything from the background are used in training.
- "Training Checkpoints" not yet implemented, beware of machine shutting down during training.
