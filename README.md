# CarND_Behaviour_Cloning

# Model Architecture

The model used in this project is adopted from "nVidia Autonomous Car Group"'s model. I made some modifications, such as changing the filter size, input normalization, and the NN size. Here is the summary of the model.

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

# Training Strategy

## Configuration

Dataset: 34425.
Optimizer: Adam.
Error metric: Mean squared error.
Validation Split: 20%.
Epochs: 3.

## Data Collection and Augmentation

The data are manually collected using the simulator. The data consists of:
1. Two laps of center lane driving on both clockwise and counter-clockwise tracks.
2. More data on driving smoothly around curves.
3. Additional data to train the model to drive the car from the sides to the center.

Two methods are used to augment the data.
1. Process the image from the right camera and correct the steered value by 0.3.
2. Perform image flip. Based on my experiment, doing this is better than processing the left camera images. Somehow the car is oscillating when I used both left and right camera images.

![alt text](/data/center.jpg "Center Camera Image")

![alt text](/data/left.jpg "Right Camera Image")

## Results

```
Epoch 1/3
26342/26342 [==============================] - 76s 3ms/step - loss: 0.0176 - val_loss: 0.0217
Epoch 2/3
26342/26342 [==============================] - 73s 3ms/step - loss: 0.0133 - val_loss: 0.0234
Epoch 3/3
26342/26342 [==============================] - 73s 3ms/step - loss: 0.0115 - val_loss: 0.0224
```



