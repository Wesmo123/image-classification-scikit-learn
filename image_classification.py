import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import h5py
import matplotlib.pyplot as plt


# preparing data

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# visualize an example image
#index = 10
#plt.imshow(train_x_orig[index])
#plt.show()
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


data = []
labels = []

for element_id, element in enumerate(train_x_orig):
    data.append(element)
    labels.append(classes[train_y[0,element_id]].decode("utf-8"))

data = np.asarray(data)
labels = np.asarray(labels)