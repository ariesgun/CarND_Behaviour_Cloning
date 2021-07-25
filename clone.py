import csv
import cv2
import numpy as np

data = ['../data/',  # Center Lane driving
        '../data2/', # round
        '../data4/', # more rounding
        '../data7/',
       ]

augmented_images = []
augmented_measurements = []

for da in data:
    lines = []
    with open(da + 'driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        lines.append(line)

    images = []
    measurements = []
    for line in lines:
      for i in [0,2]: # Center, Right lanes
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = da + 'IMG/' + filename
        image = cv2.imread(current_path)
        if (np.any(image)):
          images.append(image)
          measurement = float(line[3])
          if i == 1:
            measurement += 0.2
          elif i == 2:
            measurement -= 0.3
          measurements.append(measurement)
        else:
          print(current_path + " is none")

    ## Data Augmentation : Flipping Images
    for image, measurement in zip(images, measurements):
      augmented_images.append(image)
      augmented_measurements.append(measurement)
      augmented_images.append(cv2.flip(image, 1))
      augmented_measurements.append(measurement*-1.0)
    
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Training Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalize input data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((55,25),(0,0))))
model.add(Convolution2D(24, (5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(32, (5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48, (3,3), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')

stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)

