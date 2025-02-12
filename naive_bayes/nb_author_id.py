#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append('/Users/eudermelo/Documents/GitHub/ud120-projects/tools/')
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
#features_train, features_test, labels_train, labels_test = preprocess()
features_train, features_test, labels_train, labels_test = preprocess()

t0 = time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0,3), "s")
t0 = time()
clf.predict(features_test)
print("Predicting time:", round(time()-t0,3), "s")
print("Score:", round(clf.score(features_test,labels_test)*100,2), "%")