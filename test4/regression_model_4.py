import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression,Ridge
import pickle
import sys


def build_regression(file):
    # get file name
    # Load file as array: 
    data = np.loadtxt(file, delimiter=';', skiprows=1)
    print data.shape
    # allocate data
    X =  data[:,:-1]
    X = preprocessing.scale(X)
    y =  data[:,-1]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # linear regression 
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)    
    print('linear regression',confidence)
    with open('linear_regression_4.pickle','wb') as f:
        pickle.dump(clf, f)

    # ridge regression 
    clf = Ridge()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)    
    print('ridge regression',confidence)    

    for k in ['linear','poly','rbf']:
        clf = svm.SVR(kernel=k,gamma=0.2)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print('svm-'+k,confidence)
    return data,X,y    

data,X,y = build_regression(sys.argv[1])


