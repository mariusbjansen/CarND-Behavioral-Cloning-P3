import csv
import cv2
import numpy as np
import os.path
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg

data_path = 'data/created/'

samples = []
with open(data_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].strip()
                if(os.path.isfile(name) ):
                #if(os.path.isfile(data_path+name) ):
                    center_image = mpimg.imread(name)
                    #center_image = mpimg.imread(data_path+name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

                    # data augmentation by flipping
                    center_image_flipped = np.fliplr(center_image)
                    center_angle_flipped = -center_angle
                    images.append(center_image_flipped)
                    angles.append(center_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Conv2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,activation='relu', subsample=(2,2)))
model.add(Conv2D(36,5,5,activation='relu', subsample=(2,2)))
model.add(Conv2D(48,5,5,activation='relu', subsample=(2,2)))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(50,activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples*2), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

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