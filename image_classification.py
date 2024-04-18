import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# !!preparing data!!

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


for element_id, element in enumerate(train_x_orig): # iterating through the train set features and adding the features into data array and its corresponding labebels from classes and train_y to the labels array
    data.append(element.flatten()) # make sure to flatten images as they are in a 2d array but they need to be in a 1d array for training, .flatten from numpy flattens images from 2d to 1d arrays
    labels.append(classes[train_y[0,element_id]].decode("utf-8"))

for element_id, element in enumerate(test_x_orig):  #combining the test set with the train set for the sake of the example to split later, obviously dont do this normally there is no point to joining and resplitting the data
    data.append(element.flatten()) # make sure to flatten images as they are in a 2d array but they need to be in a 1d array for training, .flatten from numpy flattens images from 2d to 1d arrays
    labels.append(classes[test_y[0,element_id]].decode("utf-8"))



data = np.asarray(data)
labels = np.asarray(labels)

# !!train/test split!!

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,shuffle=True, stratify=labels) # splitting data into a training and test set using the sklearn train_test_split function at a ratio of 80% train data and 20% test data, images are shuffled and stratify by labels just makes sure the proportions of classes is the same in each set as they were in the origonal data set, the ration is determined by our labels

# !!train classifier!!

classifier = SVC() #creating new instance of SVC and calling the instance classifier, look up SVC on sklearn

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma':[0.01, 0.001, 0.0001]}] # parameters is a list containing a dictionary with 2 keys gamma and C each key has values
# gamma and C are both parameters of an SVC object, in effect this will allow us to train not 1 image classifier but 12(gamma * C)

grid_search = GridSearchCV(classifier, parameters) # grid search allows us to take an SVC instance (classifier) and a set of parameters (parameters) then run every possible combination of parameters in the object to find the best performing model, the amount of models evaluated is equal to (paramater * parameter * paramater.. until there are no svc parameters left)

grid_search.fit(x_train, y_train) # runs through all the models of our SVC object from the previous step with our training data, this is training

# !!test performance!!

best_estimator = grid_search.best_estimator_ # best_estimator_ allows us to get the best of all the classifiers that were trained, this is our best model

y_prediction = best_estimator.predict(x_test) #  runs our model on the test data and stores the results in y_prediction, y_prediction is an array of labels our "test results"

score = accuracy_score(y_prediction,y_test) # uses the sklearn.metrics accuracy_score module to calculate the accuracy of our best estimator mmodule, y_test is the true array of labels, a "memo"

print('{}% of samples were correctly classified'.format(str(score * 100))) # displays our calculated accuracy score

pickle.dump(best_estimator, open('./model.p','wb')) # uses the pickle library to dump the best model and saves it as model.p in our directory for use in other projects or taking it elsewhere