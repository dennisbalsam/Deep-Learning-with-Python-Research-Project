#include libs that are needed
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

from PIL import Image, ImageFile
# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight




# dl libraraies
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Dense, Activation
# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os

#ignore warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#function for pie chart outputs of percentage and total number
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

#function for outputting total number of images when imported
def plot_total_values(number_classes, folders):
    # Data to plot
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
    # Plot
    # pie chart
    plt.title('Number of patterns in each class')
    plt.pie(number_classes, labels=folders, colors=colors, autopct=make_autopct(number_classes), shadow=True,
            startangle=90)
    plt.axis('equal')
    plt.show()
    # bar graph
    plt.figure(figsize=[12, 6])
    plt.bar(y_pos, samples, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.title('Number of Samples per Class before Augmentation')
    plt.show()


#define path for flowers
#laptop path
# data = 'C:/Users/denni/Documents/CSI-Courses/CSC450/Flowers-Model-Code/flowers'
#pc path
data = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers'
# List out the directories inside the main input folder
folders = os.listdir(data)
print(folders)


#create array with 5 classes of flowers
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


# define arrays for images and their corresponding labels
image_names = []
train_labels = []
X_data = []
Y_data = []
size = 128, 128
i = 0

# import the samples
for folder in folders:
    #tqdm allows for progress bar output
    for file in tqdm(os.listdir(os.path.join(data, folder))):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data, folder, file))
            train_labels.append(folder)
            img = Image.open(os.path.join(data, folder, file))
            img = img.resize((128, 128), Image.ANTIALIAS)  # resizes image without ratio
            img = np.array(img)
            if img.shape == (128, 128, 3):
                X_data.append(img)
                Y_data.append(i)
        else:
            continue
    i+=1

#print out shape of the data arrays
X = np.array(X_data)
Y = np.array(Y_data)

# Print shapes to see if they are correct
print(X.shape)
print(Y.shape)



#output total number of samples per class
Y = np.array(Y_data)
number_classes = [0,0,0,0,0]
for i in range(len(Y)):
    if (Y[i] == 0):
        number_classes[0] += 1
    elif (Y[i] == 1):
        number_classes[1] += 1
    elif (Y[i] == 2):
        number_classes[2] += 1
    elif (Y[i] == 3):
        number_classes[3] += 1
    elif (Y[i] == 4):
        number_classes[4] += 1

objects = folders
y_pos = np.arange(len(objects))
samples = number_classes



#print total number of classes before augmentation
print("Number of Samples per Class before Augmentation: ", number_classes)
plot_total_values(number_classes, folders)


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
        vertical_flip=False,
        fill_mode='nearest')  # randomly flip images

# # Augmenting and saving the images into Class Directory
# # Reset data folder if running again
# for folder in folders:
#     #tqdm allows for progress bar output
#     for file in tqdm(os.listdir(os.path.join(data, folder))):
#         if file.endswith("jpg"):
#             img = load_img(os.path.join(data,folder,file))
#             x = img_to_array(img)
#             x = x.reshape((1,) + x.shape)
#             i = 0
#
#             for batch in datagen.flow(x,save_to_dir=data + '/' + folder, save_prefix=str(folder), save_format='jpg'):
#                 i += 1
#                 if (folder == 'daisy'):
#                     if i > 2:
#                         break
#                 elif (folder == 'dandelion'):
#                     if i > 1:
#                         break
#                 elif (folder == 'rose'):
#                     if i > 2:
#                         break
#                 elif (folder == 'sunflower'):
#                     if i > 2:
#                         break
#                 elif (folder == 'tulip'):
#                     if i > 1:
#                         break
#     print("Finished Augmenting Class ", folder)
#
# # define arrays for images and their corresponding labels
# image_names = []
# train_labels = []
# X_data = []
# Y_data = []
# size = 128, 128
# i = 0
#
# # import the samples
# for folder in folders:
#     #tqdm allows for progress bar output
#     for file in tqdm(os.listdir(os.path.join(data, folder))):
#         if file.endswith("jpg"):
#             image_names.append(os.path.join(data, folder, file))
#             train_labels.append(folder)
#             img = Image.open(os.path.join(data, folder, file))
#             img = img.resize((128, 128), Image.ANTIALIAS)  # resizes image without ratio
#             img = np.array(img)
#             if img.shape == (128, 128, 3):
#                 X_data.append(img)
#                 Y_data.append(i)
#         else:
#             continue
#     i+=1
#
#
#
#
# #output total number of samples per class with augmentation
# Y = np.array(Y_data)
# number_classes = [0,0,0,0,0]
# for i in range(len(Y)):
#     if (Y[i] == 0):
#         number_classes[0] += 1
#     elif (Y[i] == 1):
#         number_classes[1] += 1
#     elif (Y[i] == 2):
#         number_classes[2] += 1
#     elif (Y[i] == 3):
#         number_classes[3] += 1
#     elif (Y[i] == 4):
#         number_classes[4] += 1
#
# objects = folders
# y_pos = np.arange(len(objects))
# samples = number_classes
#
#
# #function for pie chart outputs of percentage and total number
# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
#     return my_autopct
#
# #print total number of classes before augmentation
# print("Number of Samples per Class after Augmentation: ", number_classes)
# plot_total_values(number_classes, folders)

# Reduce the RGB values between 0 and 1
X = X.astype('float32') / 255.0
y_cat = to_categorical(Y_data, len(class_labels))


#start building the convnet model with 4 layers
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation="softmax"))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# split the data set into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.20, random_state=42)
print("The model has " + str(len(x_train)) + " inputs")


#output amount of samples in train and test
print("The model has " + str(len(x_train)) + " inputs")
objects = ('Train','Test')
y_pos = np.arange(len(objects))
samples = []
samples.append(len(x_train))
samples.append(len(x_test))

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
#pie chart
plt.pie(samples, labels=objects, colors=colors,autopct=make_autopct(samples), shadow=True, startangle=90)
plt.axis('equal')
plt.title('Number of total samples (Train/Test)')
plt.show()

#bar graph
plt.bar(y_pos, samples, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('Number of Samples')
plt.show()
# output total amount of samples per set of each class
# y_train[1][8]
split_classes_train = [0, 0, 0, 0, 0]
split_classes_test = [0, 0, 0, 0, 0]

for i in range(len(y_train)):
    for j in range(len(folders)):
        if (y_train[i][j] == 1):
            split_classes_train[j] += 1

for i in range(len(y_test)):
    for j in range(len(folders)):
        if (y_test[i][j] == 1):
            split_classes_test[j] += 1

objects = folders
y_pos = np.arange(len(objects))

samples_train = split_classes_train
samples_test = split_classes_test

#Pie charts
# Train Samples Plot
plt.pie(samples_train, labels=folders, colors=colors,autopct=make_autopct(samples_train), shadow=True, startangle=90)
plt.axis('equal')
plt.title('Number of Samples per Class (Train)')
plt.show()


# Test Samples
plt.pie(samples_test, labels=folders, colors=colors,autopct=make_autopct(samples_test), shadow=True, startangle=90)
plt.axis('equal')
plt.title('Number of Samples per Class (Test)')
plt.show()

#bar graphs
#Train Samples
plt.figure(figsize=[12,6])
plt.bar(y_pos, samples_train, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('Number of Samples per Class (Train)')
plt.show()

#Test Samples
plt.figure(figsize=[12,6])
plt.bar(y_pos, samples_test, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('Number of Samples per Class (Test)')
plt.show()

#begin training the model
train_generator = datagen.flow(x_train, y_train, batch_size=128)
validation_generator = datagen.flow(x_test, y_test, batch_size=128)


#computing the weights to balance the dataset
classes={0,1,2,3,4}
weights = compute_class_weight('balanced', np.unique(Y_data), Y_data)
print("Class Weights: ", weights)