import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression,Ridge
import pickle
import sys

# removing outliers
def reject_outliers(data,ax=0,index=3, m = 2.):
    d = np.abs(data[:,index] - np.median(data,axis=ax)[index])
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    dfinal= data[s<m]
    return dfinal

def build_regression(data,id):
    # get file name
    # Load file as array: 
    data = reject_outliers(data,ax=0,index=id, m = 2.4) 
    print data.shape    # allocate data
    X =  data[:,:-1]
    X = preprocessing.scale(X)
    y =  data[:,-1]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # linear regression 
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)    
    print('linear regression',confidence)
    with open('linear_regression_2.pickle','wb') as f:
        pickle.dump(clf, f)

    # ridge regression 
    clf = Ridge()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)    
    print('ridge regression',confidence)    

    for k in ['linear','poly','rbf']:
        clf = svm.SVR(kernel=k)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print('svm-'+k,confidence)
    return data,X,y    


data = np.loadtxt('testData2.csv', delimiter=';', skiprows=1)
# data_new = np.loadtxt('testData1_new.csv', delimiter=',', skiprows=1)

data,X,y = build_regression(data,id=6)
# data,X,y = build_regression(data_new,id=4)
data_filtered = pd.DataFrame(data)
data_filtered.to_csv("test2_filtered.csv")


