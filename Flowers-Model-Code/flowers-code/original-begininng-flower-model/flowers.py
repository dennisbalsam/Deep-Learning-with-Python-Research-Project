# output the directory of the kaggle flowers dataset to ensure the needed sets are there
import os

# import the libraries that are needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
import seaborn as sns

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)
plt.show();
# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# preprocess.
from keras.preprocessing.image import ImageDataGenerator

# dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras import models
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image

#ignore warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


#define path for flowers
data = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers'
# List out the directories inside the main input folder
folders = os.listdir(data)
print(folders)


# define arrays for images and their corresponding labels
image_names = []
train_labels = []
train_images = []
size = 150, 150


for folder in folders:
    # tqdm allows for progress bar output
    for file in tqdm(os.listdir(os.path.join(data, folder))):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img, size)
            train_images.append(im)
        else:
            continue

#output 10 random images
fig, ax = plt.subplots(5, 2)
fig.set_size_inches(15, 15)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(train_labels))
        ax[i, j].imshow(train_images[l])
        ax[i, j].set_title('Flower: ' + train_labels[l])
plt.tight_layout()
plt.show()


# Transform the image array to a numpy type
train = np.array(train_images)
# Reduce the RGB values between 0 and 1
train = train.astype('float32') / 255.0
# create corrseponding Y array with potential flower types
le=LabelEncoder()
Y=le.fit_transform(train_labels)
Y=to_categorical(Y, 5)

#output the shape of the arrays
print(train.shape)
print(Y.shape)

# split the data set into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(train, Y, test_size=0.25, random_state=42)

#output shapes of the training and validation sets
print('Training Set 1 size: ', x_train.shape)
print('Training Set 2 size: ', y_train.shape)
print('Test Set 1 size: ', x_test.shape)
print('Test Set 2 size: ', y_test.shape)

#start building the convnet model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation="softmax"))

# define number of epoch and batch size
batch_size=128
epochs=50

# add a LR Annealear
from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

# include data augmnetation to avoid overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

#compile the model and output the summary
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()



# calculate the models dropout
History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(x_test, y_test),
                              verbose=1, steps_per_epoch=x_train.shape[0] // batch_size)


acc = History.history['acc']
val_acc = History.history['val_acc']
loss = History.history['loss']
val_loss = History.history['val_loss']

#evaluate the model
test_eval = model.evaluate(x_test, y_test, verbose=1)

#print the loss and accuracy print("Loss=",test_eval[0])
print("Loss=", test_eval[0])
print("Accuracy=", test_eval[1])

#save the model
model.save('flowers.h5')

#output the models performance - loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#output the models -performance - accuracy
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


