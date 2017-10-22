import csv
import cv2
import numpy as np
import os.path
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg

data_path = 'data/0_training_data/'

# read csv file and store lines in "samples"
# "samples" include image paths (center,left,right) and steering angle
samples = []
with open(data_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# splitting data set in training and validation 80% and 20% respectively
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Center, left and right images are recorded. Left and right images get and
# correction factor of the steering angle
# Trade-off here: Higher correction factor -> higher oscillations (bad)
# Lower correction factor -> harder learning to stay on track (also bad)
correction_factor = 0.15

# Generator pattern (yield) in order to not save all data in memory at one time
# Only batch portion which is currently needed by the corresponding 
# model.fit_generator is held in memory

# data augmentation
augmented_fac_flipping = 2
augmented_fac_center_left_right = 3

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # center 0, left 1, righ 2 images handling
                for i in range(augmented_fac_center_left_right):
                    name = batch_sample[i].strip()
                    if(os.path.isfile(name) ):
                        image = mpimg.imread(name)
                        angle = float(batch_sample[3]+'.'+batch_sample[4])
                        if (i == 1):
                            angle += correction_factor
                        if (i == 2):
                            angle -= correction_factor

                        images.append(image)
                        angles.append(angle)

                        # data augmentation by flipping
                        image_flipped = np.fliplr(image)
                        angle_flipped = -angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Conv2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(6,3,3,activation='relu', subsample=(2,2)))
model.add(Conv2D(9,3,3,activation='relu', subsample=(2,2)))
model.add(Conv2D(12,3,3,activation='relu', subsample=(2,2)))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(20,activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

augmented_fac_total = augmented_fac_flipping * augmented_fac_center_left_right

history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples*augmented_fac_total), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('models/data.h5')
exit()