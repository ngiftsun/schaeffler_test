# schaeffler_test

### Test 6 
The model uses a Gaussian Mixture Model(GMM) to cluster the given data purely because of the nature of the distribution.  Scikit tools provide the information-theoretic criteria for each GMM model depending on the number of clusters and the covariance type. Iterating over a range of possible clusters and different covariance types, the model is selected with high information-theoretic criteria score. The figure below shows the prediction of the final trained model which clusters the given data in to 5 clusters. Run this script to check for other data.
```
import numpy,pickle
X = np.loadtxt(file, delimiter=';', skiprows=1)
pickle_in = open('clustering_test6.pickle','rb')
clf = pickle.load(pickle_in)
labels = clf.predict(X)
```
![Alt text](/test6/clustering.png?raw=true "Results")



