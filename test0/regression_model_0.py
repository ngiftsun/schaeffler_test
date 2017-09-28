import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import pickle
import sys
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# removing outliers
def reject_outliers(data,ax=0,index=3, m = 2.):
    d = np.abs(data[:,index] - np.median(data,axis=ax)[index])
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    dfinal= data[s<m]
    return dfinal

def build_regression(file,ty='rejection'):
    # get file name
    # Load file as array: 
    data = np.loadtxt(file, delimiter=';', skiprows=1)
    if ty == 'rejection':
        data = reject_outliers(data,ax=0,index=3, m = 2.3) 
    # allocate data
    X = preprocessing.scale(data[:,:-1])
    X =  data[:,:3]
    y =  data[:,-1]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    #print X_train[0] 
    X1_train =  np.matrix(X_train[:,0]).T
    X1_test = np.matrix(X_test[:,0]).T
    #print X1_train[0]
    X2_train =  X_train[:,:2]
    X2_test = X_test[:,:2]
    #print X2_train[0]
    X3_train =  X_train[:,:3]
    X3_test = X_test[:,:3] 
    #print X3_train[0]

    # linear regression 
    clf1 = LinearRegression()
    clf1.fit(X1_train, y_train)
    confidence = clf1.score(X1_test, y_test)    
    print('linear regression a',confidence)

    clf2 = LinearRegression()
    clf2.fit(X2_train, y_train)
    confidence = clf2.score(X2_test, y_test)    
    print('linear regression b or c ',confidence)

    clf3 = LinearRegression()
    clf3.fit(X3_train, y_train)
    confidence = clf3.score(X3_test, y_test)    
    print('linear regression b c',confidence)    

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf3.score(X_test, y_test)    
    print('linear regression a b c',confidence)    

    with open('linear_regression_0.pickle','wb') as f:
        pickle.dump(clf, f)

    return data,X,y
    # # ridge regression 
    # clf = Ridge()
    # clf.fit(X_train, y_train)
    # confidence = clf.score(X_test, y_test)    
    # print('ridge regression',confidence)    
    
    # for k in ['linear']:
    #     clf = svm.SVR(kernel=k,gamma=0.2)
    #     clf.fit(X_train, y_train)
    #     confidence = clf.score(X_test, y_test)
    #     print('svm-'+k,confidence)

 
print 'with no cleaning'
data,X,y = build_regression(sys.argv[1],ty= 'no_rejection')
print 'After rejecting outliers'
data,X,y = build_regression(sys.argv[1],ty= 'rejection')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # for i in range(X.shape[0]):
# #     ax.scatter(X[i,0],X[i,1],y[i],c='r', marker='o')
# # Y = clf.predict(X)
# # for i in range(X.shape[0]):
# #     ax.scatter(X[i,0],X[i,1],Y[i],c='g', marker='o')
# plt.show()




