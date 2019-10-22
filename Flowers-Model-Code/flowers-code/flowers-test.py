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

#function for pie chart outputs
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


#define path for flowers
data = 'C:/Users/denni/Documents/CSI-Courses/CSC450/Flowers-Model-Code/flowers'
# List out the directories inside the main input folder
folders = os.listdir(data)
print(folders)

#load the model into the program
model = load_model('C:/Users/denni/Documents/CSI-Courses/CSC450/Flowers-Model-Code/flowers-code/flowers.h5')
model.summary()

# define arrays for images and their corresponding labels
image_names = []
train_labels = []
train_images = []
Y_data = []
size = 150, 150
i = 0

for folder in folders:
    #tqdm allows for progress bar output
    for file in tqdm(os.listdir(os.path.join(data, folder))):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img, size)
            train_images.append(im)
            Y_data.append(i)
        else:
            continue
    i+=1

#output total number of classes of images
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

# Data to plot
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
# Plot
plt.title('Total Number of Photos in the dataset')
plt.pie(number_classes, labels=folders, colors=colors,autopct=make_autopct(number_classes), shadow=True, startangle=90)
plt.axis('equal')
plt.show()


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
x_train, x_test, y_train, y_test = train_test_split(train, Y, test_size=0.20, random_state=42)


#output amount of samples in train and test
print("The model has " + str(len(x_train)) + " inputs")
objects = ('Train','Test')
y_pos = np.arange(len(objects))
samples = []
samples.append(len(x_train))
samples.append(len(x_test))

plt.pie(samples, labels=objects, colors=colors,autopct=make_autopct(samples), shadow=True, startangle=90)
plt.axis('equal')
plt.title('Number of total samples (Train/Test)')
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


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')



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



#print out classification report
print(classification_report(y_true, y_pred, target_names=folders))

# output correctly identified flowers
fig, ax = plt.subplots(3 , 3)
fig.set_size_inches(8, 8)
count = 0
for i in range(3):
    for j in range(3):
        ax[i, j].imshow(x_test[corr[count]])

        ax[i, j].set_title("Actual Flower : " + str(
            encoder.inverse_transform([y_true[corr[count]]])) + "\n" + "Predicted Flower : " + str(
            encoder.inverse_transform([y_pred[corr[count]]])))

        count += 1

plt.tight_layout()
plt.show()


#output incorrectly identified flowers
fig, ax = plt.subplots(3, 3)
fig.set_size_inches(8, 8)
count = 0
for i in range(3):
    for j in range(3):
        ax[i, j].imshow(x_test[incorr[count]])
        ax[i, j].set_title("Actual Flower : " + str(
            encoder.inverse_transform([y_true[incorr[count]]])) + "\n" + "Predicted Flower : " + str(
            encoder.inverse_transform([y_pred[incorr[count]]])))

        count += 1

plt.tight_layout()
plt.show()


#create precision recall, F1 score, ROC curve, AOC graph output

from sklearn import preprocessing
from scipy import interp
from itertools import cycle
from sklearn import metrics

n_classes = len(folders)
lb = preprocessing.LabelBinarizer()
lb.fit(y_test)
y_true = lb.transform(y_true)
y_pred = lb.transform(y_pred)

lw = 2


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC and AUC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return metrics.roc_auc_score(y_test, y_pred, average=average)


print("ROC_AUC_Score:", multiclass_roc_auc_score(y_true, y_pred))


# ROC and AUC score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                        y_pred[:, i])
    average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
                                                                y_pred.ravel())
average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

#plot the figure
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')#,
                 #**step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
plt.show()


#precision-recall curve
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(21, 21))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class:{0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(lines, labels, loc=(0, -.2), prop=dict(size=10))
plt.show()