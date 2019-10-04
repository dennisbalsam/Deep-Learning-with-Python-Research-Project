###########################################################################
# this program is for analyzing the performance of the CNN model          #
# it evaluates the number of errors in the classification of the images     #
# from the validation and test data set.                                   #
###########################################################################

import tensorflow
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

# loading the model
model = load_model('C:/Users/denni/PycharmProjects/IMAGEDETECT/cats_and_dogs_small_1.h5')

validation_dir = 'C:/Users/denni/Downloads/cats_and_dogs_small/validation'
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# retriving the probabilities for the validation data set
validation_generator.reset
yvalid = model.predict_generator(validation_generator, steps=50)
yvalid.shape

# defining a dataframe to display the results in a table
df = pd.DataFrame({
    'filename': validation_generator.filenames,
    'predict': yvalid[:, 0],
    'y': validation_generator.classes
})

# displaying the results of the predictions as classes (0 or 1)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df['y_pred'] = df['predict'] > 0.5
df.y_pred = df.y_pred.astype(int)
df.head(1000)

# displaying the the 500 results fromm raw 500 to 1000
df.iloc[500:999, 0:4]

# counting the number of erros in the classification on the validation data set
misclassified = df[df['y'] != df['y_pred']]
print('Total misclassified image from 1000 Validation images : %d' % misclassified['y'].count())

# confusion matrix for the validation data set
conf_matrix = confusion_matrix(df.y, df.y_pred)
sns.heatmap(conf_matrix, cmap="Blues", annot=True, fmt='g');
plt.xlabel('predicted value')
plt.ylabel('true value');

# rescaling the data from the test data set
test_dir = 'C:/Users/denni/Downloads/cats_and_dogs_small/test'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# retriving the probabilities for the test data images
test_generator.reset
ytest = model.predict_generator(test_generator, steps=50)

df = pd.DataFrame({
    'filename': test_generator.filenames,
    'predict': ytest[:, 0],
    'y': test_generator.classes
})

pd.set_option('display.float_format', lambda x: '%.5f' % x)
df['y_pred'] = df['predict'] > 0.5
df.y_pred = df.y_pred.astype(int)

# printing the results for classes and predict values for tes data-set
df.head(1000)
df.iloc[500:999, 0:4]

# counting the number of erros in the classification for images of the test dataset
misclassified = df[df['y'] != df['y_pred']]
print('Total misclassified image from 1000 Test images : %d' % misclassified['y'].count())

# confucion matrix
conf_matrix = confusion_matrix(df.y, df.y_pred)
sns.heatmap(conf_matrix, cmap="Blues", annot=True, fmt='g');
plt.xlabel('predicted value')
plt.ylabel('true value');

# Some of Cat image misclassified as Dog.

CatasDog = df['filename'][(df.y == 0) & (df.y_pred == 1)]
fig = plt.figure(figsize=(15, 10))
columns = 5
rows = 3
for i in range(columns * rows):
    img = image.load_img('C:/Users/denni/Downloads/cats_and_dogs_small/test/' + CatasDog.iloc[i],
                         target_size=(150, 150))
    fig.add_subplot(rows, columns, i + 1)
    plt.text(35, 140, CatasDog.iloc[i], color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
    plt.imshow(img)

plt.show()
CatasDog.shape

# Some of Dog image misclassified as Cat.

DogasCat = df['filename'][(df.y == 1) & (df.y_pred == 0)]
fig = plt.figure(figsize=(15, 10))
columns = 5
rows = 3
for i in range(columns * rows):
    img = image.load_img('C:/Users/denni/Downloads/cats_and_dogs_small/test/' + DogasCat.iloc[i],
                         target_size=(150, 150))
    fig.add_subplot(rows, columns, i + 1)
    plt.text(35, 140, DogasCat.iloc[i], color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
    plt.imshow(img)

plt.show()
DogasCat.shape

# performance of the model at random images from the test data set
fig = plt.figure(figsize=(15, 10))
columns = 5
rows = 3
for i in range(columns * rows):
    fig.add_subplot(rows, columns, i + 1)
    m = test_generator.filenames[np.random.choice(range(1000))]
    img1 = image.load_img('C:/Users/denni/Downloads/cats_and_dogs_small/test/'
                          + m, target_size=(150, 150))
    img = image.img_to_array(img1)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, batch_size=None, steps=1)
    if (prediction[:, :] > 0.5):
        value = 'Dog :%1.2f' % (prediction[0, 0])
        plt.text(35, 140, value, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
        print(m)
    else:
        value = 'Cat :%1.2f' % (1.0 - prediction[0, 0])
        plt.text(35, 140, value, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
        print(m)
    plt.imshow(img1)

# evaluating the model accuracy
train_dir = 'C:/Users/denni/Downloads/cats_and_dogs_small/train'
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

x1 = model.evaluate_generator(train_generator, steps=100)
x2 = model.evaluate_generator(test_generator, steps=50)

print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f' % (x1[1] * 100, x1[0]))
print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f' % (x2[1] * 100, x2[0]))

