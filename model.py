import csv
import cv2
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt

lines = []

with open('./data/driving_log.csv') as csvfile:
# with open('D:/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# print(type(lines))
# print(lines[0])
# print(lines[1])

images = []
measurements = []
correction = 0.2

for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
#         current_path = 'D:/data/data/IMG/' + filename
#         image = cv2.imread(current_path)  # cv2.imread will get images in BGR format, while drive.py uses RGB
        image = ndimage.imread(current_path)
        images.append(image)

        measurement = float(line[3])
        if i == 0:
            measurements.append(measurement)
        elif i == 1:
            measurements.append(measurement + correction)
        elif i == 2:
            measurements.append(measurement - correction)
        else:
            print('error')

# data augmentation by flipping images and steering angles
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Conv2D, MaxPooling2D, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

### LeNet
# model.add(Conv2D(6, 5, 5))
# # model.add(MaxPooling2D())
# # model.add(Dropout(0.5))
# model.add(Activation('relu'))

# model.add(Conv2D(6, 5, 5))
# # model.add(MaxPooling2D())
# # model.add(Dropout(0.5))
# model.add(Activation('relu'))

# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# # model.add(Activation('relu'))
# model.add(Dense(1))

### Nvidia 
# model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
# model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
# model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
# model.add(Conv2D(64,3,3, activation='relu'))
# model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

###
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 3, verbose = 1)

model.save('model.h5')

### print the keys contained in the history object
# print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()