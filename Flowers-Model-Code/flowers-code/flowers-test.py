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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# preprocess.

# dl libraraies
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os

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


# Transform the image array to a numpy type
train = np.array(train_images)
# Reduce the RGB values between 0 and 1
train = train.astype('float32') / 255.0


# create corrseponding Y array with potential flower types
encoder=LabelEncoder()
Y=encoder.fit_transform(train_labels)
Y=to_categorical(Y, 5)


#output the shape of the arrays
print(train.shape)
print(Y.shape)

# split the data set into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(train, Y, test_size=0.25, random_state=42)


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#load the model into the program
model = load_model('C:/Users/dkrup/PycharmProjects/FlowersModel/flowers.h5')


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).astype(int)

y_true = np.argmax(y_test, axis=1).astype(int)

#output the stats of correctly/non-correctly identified flowers
corr = []
incorr = []
corr_count = 0
incorr_count = 0

for i in range(len(y_test)):
    if (y_pred[i] == y_true[i]):
        corr.append(i)
        corr_count += 1
    else:
        incorr.append(i)
        incorr_count += 1

print("Found %d correct flowers" % (corr_count))
print("Found %d incorrect flowers" % (incorr_count))

import itertools
#create a confusion matrix
i = 0
j = 0
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j])
plt.show()





# output correctly identified flowers
fig, ax = plt.subplots(4, 3)
fig.set_size_inches(13, 13)

count = 0
for i in range(4):
    for j in range(3):
        ax[i, j].imshow(x_test[corr[count]])

        ax[i, j].set_title("Actual Flower : " + str(
            encoder.inverse_transform([y_true[corr[count]]])) + "\n" + "Predicted Flower : " + str(
            encoder.inverse_transform([y_pred[corr[count]]])))

        count += 1

plt.tight_layout()

#output incorrectly identified flowers
fig, ax = plt.subplots(4, 3)
fig.set_size_inches(13, 13)
count = 0
for i in range(4):
    for j in range(3):
        ax[i, j].imshow(x_test[incorr[count]])
        ax[i, j].set_title("Actual Flower : " + str(
            encoder.inverse_transform([y_true[incorr[count]]])) + "\n" + "Predicted Flower : " + str(
            encoder.inverse_transform([y_pred[incorr[count]]])))

        count += 1

plt.tight_layout()
plt.show()

