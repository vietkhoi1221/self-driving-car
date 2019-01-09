import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
import pickle
import time

fixed_size = tuple((200,200))


train_path = "dataset/train"


bins = 8


seed = 9

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)

global_features = []
labels = []

images_per_class = 73

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    for x in range(1,images_per_class+1):
        file = "./" + dir + "/" + "image_" + str(x) + ".jpg"
        file = file.replace('\\', '/')
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        fv_hu_moments = fd_hu_moments(image)
        fv_histogram  = fd_histogram(image)
        global_feature = np.hstack([fv_histogram,fv_hu_moments])
        labels.append(current_label)
        global_features.append(global_feature)
        print(".")

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

clf  =  MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4,solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
clf.fit(global_features, global_labels)
filename = 'kh.sav'
pickle.dump(clf, open(filename, 'wb'))
