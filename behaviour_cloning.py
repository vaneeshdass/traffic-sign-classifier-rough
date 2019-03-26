# importing the requireed libraries
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

# this stores all the data as list
samples = []  # simple array to append all the entries present in the .csv file

# opening the csv file for data
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # here i am skimping the header as first line is header in csv file
    next(reader, None)
    for line in reader:
        samples.append(line)

print("Done")

# here splitting the data using sklearn in train and validation. 15% dataset is being used for validation and rest for training
train_samples, validation_samples = train_test_split(samples, test_size=0.15)


def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)  # shuffling the data
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                # here 3 images are in consideration center, left & right
                # here total we got 6 images after this loop
                for i in range(0, 3):

                    name = './data/IMG/' + batch_sample[i].split('/')[-1]

                    # converting from BGR to RGB
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

                    # here taking the steering angle
                    center_angle = float(batch_sample[3])  # getting the steering angle measurement
                    images.append(center_image)

                    # introducing correction for left and right images
                    # if image is in left we increase the steering angle by 0.2
                    # if image is in right we decrease the steering angle by 0.2

                    if (i == 0):
                        steering_angles.append(center_angle)
                    elif (i == 1):
                        steering_angles.append(center_angle + 0.2)
                    elif (i == 2):
                        steering_angles.append(center_angle - 0.2)

                    # here we flipping the image(data augmentation)
                    images.append(cv2.flip(center_image, 1))
                    if (i == 0):
                        steering_angles.append(center_angle * -1)
                    elif (i == 1):
                        steering_angles.append((center_angle + 0.2) * -1)
                    elif (i == 2):
                        steering_angles.append((center_angle - 0.2) * -1)

            X_train = np.array(images)
            y_train = np.array(steering_angles)

            yield sklearn.utils.shuffle(X_train, y_train)


# here compiling and training the model
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()

# pre-processing the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# trimming image , removing unwanted area
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

# layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

# layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

# layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

# layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

# flatten image
model.add(Flatten())

# layer 6- fully connected
model.add(Dense(100))
model.add(Activation('elu'))

# Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
# drop out layer for curbing overfitting
model.add(Dropout(0.25))

# layer 7- fully connected
model.add(Dense(50))
model.add(Activation('elu'))

# layer 8- fully connected
model.add(Dense(10))
model.add(Activation('elu'))

# layer 9- fully connected
# This will contain only one value as its regression
model.add(Dense(1))

# adam optimizer is used here
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

# here saving the model
model.save('model.h5')

print('Model is being saved!!')

# here printing the summary
model.summary()
