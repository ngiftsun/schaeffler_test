# schaeffler_test
## Test 0
Different regression models were tested to find out that linear regression is more suitable for the give data set but it is also understood that the feature 'a' has no use while 'b' and 'c' has a linear pattern in the scatter plots. 
![alt-text-1](/test0/analysis_1.png "title-2")

Though regression is suitable, it did not work well due to the obvious outliers as seen in the plots. But after rejecting the outliers using a median based filter improves the performance of the model. The below metrics show the accuracy score of linear regression model with different configurations
```
With no cleaning
('linear regression a', -0.00022688400900561234)
('linear regression b or c ', 0.42507803749145168)
('linear regression b c', 0.42510010815810628)
('linear regression a b c', 0.42510010815810628)
After rejecting outliers
('linear regression a', -0.0018738558069439915)
('linear regression b or c ', 0.94238257661225511)
('linear regression b c', 0.9412393885098318)
('linear regression a b c', 0.9412393885098318)
```
## Test 1
Similar to Test 0, after filtering outliers, linear regression captures the model to an accuracy of 92%. Features 'b' and 'd' don't seem to have any value to the model.
```
('linear regression', 0.91455294682158228)
```
## Test 2
Similar to the previous tests, after filtering outliers, linear regression captures the model to an accuracy of around 80%. 
```
('linear regression', 0.85297415222630291)
('ridge regression', 0.85293572025048303)
('svm-linear', 0.84991946407442953)
('svm-poly', 0.73381953944876288)
('svm-rbf', 0.79683567808858091)
```
## Test 3
Similar to the previous tests, after filtering outliers, linear regression captures the model to an accuracy of 85%. Features 'b' and 'd' don't seem to have any value to the model.
```
('linear regression', 0.85439841655456128)
```
## Test 4
Different regression models are tested to find out that SVMs, Linear and ridge regression work pretty well for this data set. The model accuracy is verified by cross validation to ensure its genericness. Since the context of the data is unknown, the problem is considered continuous rather than categorical ignoring the integerness of the 't'
#### Training
```
ipython -i regression_model_4.py testData4.csv
```
##### Comparison of different regression models' confidence
```
('linear regression', 0.99975667051268147)
('ridge regression', 0.99975448579514004)
('svm-linear', 0.99974844384657435)
('svm-poly', 0.90306540828055337)
('svm-rbf', 0.99943503958248603)
``` 
Linear regression is chosen finally
## Test 5 
Different regression models were tested to find out that SVM-RBF is the one that works pretty well for this data set. The model accuracy is verified by cross validation to ensure its genericness. Since the context of the data is unknown and due to the negative sign in the label 't', the problem is considered continuous rather than categorical.
#### Training
```
ipython -i regression_model_5.py testData5.csv
```
##### Comparison of different regression models' confidence
```
('linear regression', 0.31254125041724956)
('ridge regression', 0.31253836180967165)
('svm-linear', 0.24515524013479773)
('svm-poly', 0.67599392487404608)
('svm-rbf', 0.97356940341638021)
``` 
SVM-RBF is chosen finally
## Test 6 
The model uses a Gaussian Mixture Model(GMM) to cluster the given data purely because of the nature of the distribution.  Scikit tools provide the information-theoretic criteria for each GMM model depending on the number of clusters and the covariance type. Iterating over a range of possible clusters and different covariance types, the model is selected with high information-theoretic criteria score. The figure below shows the prediction of the final trained model which clusters the given data in to 5 clusters. Run this script to check for other data.
#### Training
```
ipython -i clustering.py testData6.csv
```
#### Loading existing Model
```
import numpy,pickle
X = np.loadtxt(file, delimiter=';', skiprows=1)
pickle_in = open('clustering_test6.pickle','rb')
clf = pickle.load(pickle_in)
labels = clf.predict(X)
```
![Alt text](/test6/clustering.png?raw=true "Results")



