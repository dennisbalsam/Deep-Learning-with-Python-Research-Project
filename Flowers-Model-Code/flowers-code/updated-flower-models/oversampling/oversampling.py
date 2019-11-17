#include libs that are needed
# output the directory of the kaggle flowers dataset to ensure the needed sets are there

import shutil
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
from keras.optimizers import Adam
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

#output functions
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

#function for outputting total number of images when imported
def plot_total_values(number_classes, sub_labels, current_label):
    # Data to plot
    colors = ['yellow', 'turquoise', 'darkorange', 'cornflowerblue', 'magenta']
    # Plot
    # pie chart
    plt.title('Number of patterns in ' + current_label + ' directory')
    plt.pie(number_classes, labels=sub_labels, colors=colors, autopct=make_autopct(number_classes), shadow=True,
            startangle=90)
    plt.axis('equal')
    plt.show()
    # bar graph
    plt.figure(figsize=[12, 6])
    plt.bar(y_pos, number_classes, align='center', alpha=0.5)
    plt.xticks(y_pos, sub_labels)
    plt.title('Number of patterns in ' + current_label + ' directory')
    plt.show()

#define path for flowers and create sub directories
# laptop path
# data = 'C:/Users/denni/Documents/CSI-Courses/CSC450/Flowers-Model-Code/flowers'
#pc path
data = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/oversampling/flowers'
# List out the directories inside the main input folder
folders = os.listdir(data)
print(folders)



# new directory for split up images
base_dir = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/oversampling/flowers-2'
# os.mkdir(base_dir)
#
# #make 3 sub directories for training - validation - testing
# #directory for training data
# train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
# class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
#
# # # directory for validation data
# validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
#
# #directory for test data
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
#
#
#
# class labels
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
#
# # setting up sub directories in each directory
# for label in class_labels:
#     print(label)
#     os.mkdir(os.path.join(train_dir, label))
#     os.mkdir(os.path.join(validation_dir, label))
#     os.mkdir(os.path.join(test_dir, label))
#
#
# # split up the files per original folder
# #
# # declare sizes of each folder
#
# daisy = 769
# dandelion = 1055
# rose = 784
# sunflower = 734
# tulip = 984
#
# class_sizes = [769, 1055, 784, 734, 984]
# y = 0
# current_class_size = 0
# # import the samples
# for folder in folders:
#     i = 0
#     current_class_size = class_sizes[y]
#     #tqdm allows for progress bar output
#     for file in tqdm(os.listdir(os.path.join(data, folder))):
#         if file.endswith("jpg"):
#             if (i < int((current_class_size / 2))):
#                 src = os.path.join(data, folder, file)
#                 dst = os.path.join(train_dir, folder, file)
#                 shutil.copyfile(src, dst)
#             elif(i < (current_class_size - (int(current_class_size/4)))):
#                 src = os.path.join(data, folder, file)
#                 dst = os.path.join(validation_dir,folder, file)
#                 shutil.copyfile(src, dst)
#             elif (i < current_class_size):
#                 src = os.path.join(data, folder, file)
#                 dst = os.path.join(test_dir, folder, file)
#                 shutil.copyfile(src, dst)
#             i += 1
#         else:
#             continue
#     y+=1

y_pos = np.arange(len(folders))




# define arrays for images and their corresponding labels
sub_labels = os.listdir(base_dir)
image_names = []
training_labels = [[],[],[]]
# testing, training, validation, imports
X_data = [[], [], []]
Y_data = [[], [], []]
size = 128, 128
i = 0
for folder in sub_labels:
    z = 0
    for sub_folder in os.listdir(os.path.join(base_dir, folder)):
        for file in tqdm(os.listdir(os.path.join(base_dir, folder, sub_folder))):
            if file.endswith("jpg"):
                image_names.append(os.path.join(base_dir, folder, sub_folder, file))
                training_labels[i].append(sub_folder)
                img = Image.open(os.path.join(base_dir, folder, sub_folder, file))
                img = img.resize((128, 128), Image.ANTIALIAS)  # resizes image without ratio
                img = np.array(img)
                if img.shape == (128, 128, 3):
                    X_data[i].append(img)
                    Y_data[i].append(z)
            else:
                continue
        z += 1
    i+=1



#print out shape of the data arrays
X = [[],[],[]]
Y =[0,0,0]
for i in range(3):
    X[i] = np.array(X_data[i])
    Y[i] = np.array(Y_data[i])


Y = np.array(Y)
print(Y.shape)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
        fill_mode='nearest')  # randomly flip images


test_labels = to_categorical(Y_data[0], len(class_labels))
train_labels = to_categorical(Y_data[1], len(class_labels))
validation_labels = to_categorical(Y_data[2], len(class_labels))

# define vars for each group
test = X[0]
train = X[1]
validation = X[2]

print(train.shape)

#reshape samples to work with smote
m_samples = train.shape[0]
train_matrix = train.reshape(m_samples, -1)

n_samples =validation.shape[0]
validation_matrix = validation.reshape(n_samples,-1)

#use smote for oversampling
from imblearn.over_sampling import SMOTE
smt = SMOTE()
train, train_labels = smt.fit_sample(train_matrix, train_labels)
validation, validation_labels = smt.fit_sample(validation_matrix, validation_labels)

#print shape after SMOTE
print(train.shape)
print(validation.shape)


#retunr back to 4D shape
train = train.reshape(train.shape[0], 128, 128, 3)
validation = validation.reshape(validation.shape[0],128,128,3)


#display new sample count
daisy= 0
dandelion =0
rose = 0
sunflower = 0
tulip = 0


#show new values
for label in train_labels:
    if (label[0] == 1):
        daisy+=1
    elif (label[1] == 1):
        dandelion+=1
    elif (label[2] == 1):
        rose +=1
    elif (label[3] == 1):
        sunflower+=1
    elif (label[4] == 1):
        tulip +=1

classes = [daisy,dandelion,rose,sunflower,tulip]
current_label = "train"
plot_total_values(classes, class_labels, current_label)


daisy= 0
dandelion =0
rose = 0
sunflower = 0
tulip = 0


#output values
for label in validation_labels:
    if (label[0] == 1):
        daisy+=1
    elif (label[1] == 1):
        dandelion+=1
    elif (label[2] == 1):
        rose +=1
    elif (label[3] == 1):
        sunflower+=1
    elif (label[4] == 1):
        tulip +=1

classes = [daisy,dandelion,rose,sunflower,tulip]
current_label = "validation"
plot_total_values(classes, class_labels, current_label)




#start building the convnet model with 4 layers
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(128, 128, 3)))
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
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])


# model = load_model('flowers-augmentation-2.h5')

# #computing the weights to balance the dataset
# classes={0,1,2,3,4}
# weights = compute_class_weight('balanced', np.unique(Y_data), Y_data)
# print("Class Weights: ", weights)
#

# define generators
test_generator = datagen.flow(test, test_labels, batch_size=128)
train_generator = datagen.flow(train, train_labels, batch_size=128)
validation_generator = datagen.flow(validation, validation_labels, batch_size=128)



# Training the model for 50 epochs
History = model.fit_generator(
    generator=train_generator,
    epochs=50,
    validation_data=validation_generator,
    verbose=1)

model.save('flowers-oversampling.h5')


#output table with loss at each epoch
acc = History.history['acc']
val_acc = History.history['val_acc']
loss = History.history['loss']
val_loss = History.history['val_loss']

print("acc \t val_acc \t loss \t val_loss")
print("------------------------------------------------------------------")

for i in range(len(acc)):
    print('{:.3}'.format(acc[i]), " \t", '{:.3}'.format(val_acc[i]), "\t\t",
          '{:.3}'.format(loss[i]), "\t", '{:.3}'.format(val_loss[i]))




#save the history object
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(History.history)

# save to json:
hist_json_file = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/oversampling/history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/oversampling/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)



# summarize history for accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, len(History.history['acc'])), History.history['acc'], 'b')
plt.plot(np.arange(1, len(History.history['val_acc']) + 1), History.history['val_acc'], 'g')
plt.title('Training Accuracy vs. Validation Accuracy')
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='best')

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(History.history['loss']) + 1), History.history['loss'], 'b')
plt.plot(np.arange(1, len(History.history['val_loss']) + 1), History.history['val_loss'], 'g')
plt.title('Training Loss vs. Validation Loss')
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'validation'], loc='best')

plt.show()

# #output some predictions on images shown to the model

y_img_batch, y_class_batch = test_generator[1]
y_pred = np.argmax(model.predict(y_img_batch),-1)
y_true = np.argmax(y_class_batch,-1)

plt.figure(figsize=(15,15))
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow((y_img_batch[i]))
    actual_label = class_labels[y_true[i]]
    predict_label = class_labels[y_pred[i]]
    plt.title("Actual: " + actual_label + "| Prediction: " + predict_label)
plt.show()
#create a confusion matrix
import itertools
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
print(classification_report(y_true, y_pred, target_names=class_labels))


#create precision recall, F1 score, ROC curve, AOC graph output

from sklearn import preprocessing
from scipy import interp
from itertools import cycle
from sklearn import metrics

n_classes = len(folders)
lb = preprocessing.LabelBinarizer()
lb.fit(test_labels)
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