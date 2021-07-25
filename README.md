# Behavioral Cloning - Writeup

## Goals

The goals/steps of this project are the following:

- Use the simulator to collect data of good driving behavior.
- Build, a convolution neural network in [Keras](https://keras.io/) that predicts steering angles from images.
- Train and validate the model with a training and validation set.
- Test that the model successfully drives around track one without leaving the road.
- Summarize the results with a written report.

## Rubric points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

- **model.py** : Containing the script to create and train the model
- **drive.py** : For driving the car in autonomous mode in the simulator
- **model.h5** : Containing a trained convolution neural network.
- **README.md** : Summarizing the results
- **run2.mp4** : Results

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file; the car can be driven autonomously around the track by executing

```
Python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Model Architecture

The model used in this project is adopted from "nVidia Autonomous Car Group"'s model. I made some modifications, such as changing the filter size, input normalization, and the NN size. 

My model consists of a convolutional NN with both 5x5 filters size and 3x3 filter sizes with depths between 24 and 64 followed by NN layers (clone.py lines 61-72). The MaxPooling dropout layer is used to reduce overfitting and ReLu layers are used to introduce nonlinearity. 

The lambda layer is used to normalized the data. It is followed by an extra layer to crop the input image.

Here is the summary of the model.

```
______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 76, 316, 24)       1824      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 38, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 154, 32)       19232     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 77, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 75, 48)        13872     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
dense_2 (Dense)              (None, 32)                3232      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 33        
=================================================================
Total params: 736,533
Trainable params: 736,533
Non-trainable params: 0
```

#### Model parameter tuning

Dataset: 34425.
Optimizer: Adam.
Error metric: Mean squared error.
Validation Split: 20%.
Epochs: 2.

#### Data Collection and Augmentation

The data are manually collected using the simulator. The data consists of:
1. Two laps of center lane driving on both clockwise and counter-clockwise tracks.
2. More data on driving smoothly around curves.
3. Additional data to train the model to drive the car from the sides to the center.

Two methods are used to augment the data.
1. Process the image from the right camera and correct the steered value by 0.3.
2. Perform image flip. Based on my experiment, doing this is better than processing the left camera images. Somehow the car is oscillating when I used both left and right camera images. By flipping the image, it can also help to train the model as it is in the left side.

![alt text](/data/center.jpg "Center Camera Image")

![alt text](/data/right.jpg "Right Camera Image")

I do lots of repetition in data collection. After training, I validated if the car was able to stay in the track or not. If not, I would check where it was not able to run within the track. With this information, I obtained more data where the car was not able to stay in the track. By adding more data, I hope the model can be better. And I repeated this cycle.

## Results

```
Epoch 1/3
26924/26924 [==============================] - 79s 3ms/step - loss: 0.0168 - val_loss: 0.0228
Epoch 2/2
26924/26924 [==============================] - 75s 3ms/step - loss: 0.0129 - val_loss: 0.0222
```

Although the validation loss is higher than the training loss indicating over fitting, at the end the car is able to stay in the track. The video can be found [here](run2.mp4)

