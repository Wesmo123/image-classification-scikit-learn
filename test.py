import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time

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
for element_id, element in enumerate(train_x_orig):
    plt.imshow(element)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print ("y = " + str(train_y[0,element_id]) + ". It's a " + classes[train_y[0,element_id]].decode("utf-8") +  " picture.")