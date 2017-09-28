import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
import sys
import pickle

def build_clustering(file):
    # get file name
    # Load file as array: 
    data = np.loadtxt(file, delimiter=';', skiprows=1)
    print data.shape
    # allocate data
    X = preprocessing.scale(data)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1,7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return data,X,best_gmm

data,X,gmm = build_clustering(sys.argv[1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r','g','b','c','k','y','m']
labels = gmm.predict(X)


for i in range(X.shape[0]):
    ax.scatter(X[i,0],X[i,1],X[i,2],c=colors[labels[i]], marker='o')

with open('clustering_test6.pickle','wb') as f:
    pickle.dump(gmm, f)