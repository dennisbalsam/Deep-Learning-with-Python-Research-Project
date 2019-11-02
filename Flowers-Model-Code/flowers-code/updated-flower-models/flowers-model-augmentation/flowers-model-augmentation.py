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
data = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/flowers'
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

# Augmenting and saving the images into Class Directory
# Reset data folder if running again
for folder in folders:
    #tqdm allows for progress bar output
    for file in tqdm(os.listdir(os.path.join(data, folder))):
        if file.endswith("jpg"):
            img = load_img(os.path.join(data,folder,file))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0

            for batch in datagen.flow(x,save_to_dir=data + '/' + folder, save_prefix=str(folder), save_format='jpg'):
                i += 1
                if (folder == 'daisy'):
                    if i > 2:
                        break
                elif (folder == 'dandelion'):
                    if i > 1:
                        break
                elif (folder == 'rose'):
                    if i > 2:
                        break
                elif (folder == 'sunflower'):
                    if i > 2:
                        break
                elif (folder == 'tulip'):
                    if i > 1:
                        break
    print("Finished Augmenting Class ", folder)

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




#output total number of samples per class with augmentation
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


#function for pie chart outputs of percentage and total number
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

#print total number of classes before augmentation
print("Number of Samples per Class after Augmentation: ", number_classes)
plot_total_values(number_classes, folders)

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
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])


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


#train the model
History = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=2*(len(x_train) // 128),
    epochs=50,
    validation_steps=10,
    validation_data=validation_generator,
    verbose=1 )


#output table with loss at each epoch
acc = History.history['acc']
val_acc = History.history['val_acc']
loss = History.history['loss']
val_loss = History.history['val_loss']

print("epoch \t  acc \t \t val_acc \t loss \t val_loss")
print("------------------------------------------------------------------")

for i in range(len(acc)):
    print("{}".format(i+1), " \t", " \t", '{:.3}'.format(acc[i]), " \t", '{:.3}'.format(val_acc[i]), "\t\t",
          '{:.3}'.format(loss[i]), "\t", '{:.3}'.format(val_loss[i]))

model.save('flowers-augmentation.h5')

#save the history object
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(History.history)

# save to json:
hist_json_file = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/flowers-model-augmentation/history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv:
hist_csv_file = 'C:/Users/dkrup/OneDrive/Documents/CSI CSC Courses/Deep-Learning-with-Python-Research-Project/Flowers-Model-Code/flowers-code/updated-flower-models/flowers-model-augmentation/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)



# summarize history for accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, len(History.history['acc'])), History.history['acc'], 'r')
plt.plot(np.arange(1, len(History.history['val_acc']) + 1), History.history['val_acc'], 'g')
plt.title('Training Accuracy vs. Validation Accuracy')
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='best')

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(History.history['loss']) + 1), History.history['loss'], 'r')
plt.plot(np.arange(1, len(History.history['val_loss']) + 1), History.history['val_loss'], 'g')
plt.title('Training Loss vs. Validation Loss')
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'validation'], loc='best')

plt.show()

#output some predictions on images shown to the model
y_img_batch, y_class_batch = validation_generator[1]
y_pred = np.argmax(model.predict(y_img_batch),-1)
y_true = np.argmax(y_class_batch,-1)

plt.figure(figsize=(15,15))
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow((y_img_batch[i]))
    actual_label = class_labels[y_true[i]]
    predict_label = class_labels[y_pred[i]]
    plt.title("Actual: " + actual_label + "| Prediction: " + predict_label)

#output confusion matrix
con_matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])

plt.figure(figsize=(10,10))
plt.title('Prediction of Flower types')
sns.heatmap(con_matrix, annot=True, fmt="d", linewidths=.5, cmap="Blues")



#print out classification report
print(classification_report(y_true, y_pred, target_names=class_labels))


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