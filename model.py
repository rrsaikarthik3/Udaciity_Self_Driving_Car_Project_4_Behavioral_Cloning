
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Dropout
import cv2
from keras.models import Model
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import math
import sklearn
import tensorflow as tf
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            batch_samples = samples[offset:offset+batch_size]
            for batch_sample in batch_samples:
            
                # create adjusted steering measurements for the side camera images
                steering_center = float(batch_sample[3])
                correction = 0.5 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = cv2.imread(batch_sample[0])
                img_left = cv2.imread(batch_sample[1])
                img_right = cv2.imread(batch_sample[2])
                img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                img_center = np.array(img_center)
                img_left = np.array(img_left)
                img_right = np.array(img_right)
                #img_center  = np.expand_dims(img_center, axis=0)
                #img_left  = np.expand_dims(img_left, axis=0)
                #img_right  = np.expand_dims(img_right, axis=0)
                img_center_f = np.fliplr(img_center)
                steering_center_f = -steering_center
                # add images and angles to data set
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                images.append(img_center_f)

                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
                angles.append(steering_center_f)


                # trim image to only see section with road
            X_t = np.array(images)
            #X_t  = np.expand_dims(X_t, axis=0)
            y_t = np.array(angles)
            yield sklearn.utils.shuffle(X_t,y_t)


lines = []
path = []
samples = []
car_images = []
steering_angles = []
root = "./Train_Data/"
dirList = os.listdir(root) # current directory
for dire in dirList: 
    if(dire == 'training_Data_2'):
        continue
    
    path = (root + str(dire) + '/')
    with open(root + str(dire) + '/' + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        flag = 0
        for line in reader:
            if(flag == 0):
                flag = 1
                continue
            line[0] = root + str(dire) + '/' + line[0]
            line[1] = root + str(dire) + '/' + line[1]
            line[2] = root + str(dire) + '/' + line[2]
            line[0] = line[0].replace(" ", "") 
            line[1] = line[1].replace(" ", "") 
            line[2] = line[2].replace(" ", "") 
            samples.append(line)



from sklearn.model_selection import train_test_split
samples = sklearn.utils.shuffle(samples)
print(samples[10])
train_samples,validation_samples = train_test_split(samples, test_size=0.2, random_state=1)


batch_size=32


# compile and train the model using the generator function
train_generator = generator(train_samples , batch_size=batch_size)
val_generator = generator(validation_samples, batch_size=batch_size)
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
ch, row, col = 80,320,3  # Trimmed image format
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(Conv2D(24,(5,5),strides=(2, 2),padding="valid",activation = "relu"))
model.add(Conv2D(36,(5,5),strides=(2, 2),padding="valid",activation = "relu"))
model.add(Conv2D(48,(5,5),strides=(2, 2),padding="valid",activation = "relu"))
model.add(Conv2D(64,(3,3),strides=(2, 2),padding="valid",activation = "relu"))
model.add(Dropout(.2, input_shape=(2,)))
model.add(Conv2D(64,(3,3),strides=(2, 2),padding="valid",activation = "relu"))


model.add(Flatten())

#2nd Layer - Add a fully connected layer
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

#model.fit_generator(train_generator #,steps_per_epoch=math.ceil(len(train_samples)/batch_size),validation_data=val_generator,validation_steps=math.ceil(len(validation_samples)/batch_size),epochs=2, #verbose=1,use_multiprocessing = True)


history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    val_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)
model.save('model_sub_5_dp.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('Plot_res.png')



            



