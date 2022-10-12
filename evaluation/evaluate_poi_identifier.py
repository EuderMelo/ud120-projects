#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("C:/Users/euderasm/GitHub/ud120-projects/tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("C:/Users/euderasm/GitHub/ud120-projects/final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.4, random_state=0)

clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test)*100)