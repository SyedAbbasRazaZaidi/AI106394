# AI106394 :  Becoming a Pro pn Kaggle (Digit Recognizer)  

### PROJECT MEMBERS:

Student ID      |     Name
--------------- | -------------
   63805        | **Syed Abbas Raza Zaidi** (group Leader)
   63583        | Amta Nadeem
   61363        | Syeda Mahrukh Zehra



## Project Description:
This Project is an extension of our previous assignment. In this pirticular project we are implying filters and convolutionm, We are using 5x5 7x7  and 9x9 weighted and unweighted filters so basically what is convolution here is a short description about concolution:
It is the single most important technique in Digital Signal Processing. Using the strategy of impulse decomposition, systems are described by a signal called the impulse response. Convolution is important because it relates the three signals of interest: the input signal, the output signal, and the impulse response. 

*** How it Works ? ***

The convolution is performed by sliding the kernel over the image, generally starting at the top left corner, so as to move the kernel through all the positions where the kernel fits entirely within the boundaries of the image. In this report we have worked for 5x5, 7x7 and 9x9 weighted and unweighted with four techniques.

Convolution network receives a normal color image as a rectangular box whose width and height are measured by the number of pixels along those dimensions, and whose depth is three layers deep, one for each letter in RGB. Those depth layers are referred to as channels. As images move through a convolution network, here we discussed its techniques, expressing them as matrices of multiple dimensions in this form: 5x5, 7x7, 9x9.

After applying convoluation and filters we have applied various techniques on our datasets techniques we have used are:
-SVM
-KNN
-Linear Regression
-Multimonial Naive Bayes

So, actullly what are these techniques?  why have we used them? and how did we used them? these are the question aroused in your mind here is the answer to it !

-SVM:
      Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outlier’s detection. We have used SVM because it is 
      Effective in high dimensional spaces. Still effective in cases where number of dimensions is greater than the number of samples. Uses a subset of training points in the         decision function (called support vectors), so it is also memory efficient.
      
-KNN:
      KNN is a non-parametric and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution. In other words, the model structure           determined from the dataset. All training data used in the testing phase. This makes training faster and testing phase slower and costlier. Costly testing phase means time       and memory. In the worst case, KNN needs more time to scan all data points and scanning all data points will require more memory for storing training data. In KNN, K is         the number of nearest neighbors. The number of neighbors is the core deciding factor. K is generally an odd number if the number of classes is 2. When K=1, then the             algorithm is known as the nearest neighbor algorithm.
      KNN has the following basic steps:
      1.	Calculate distance
      2.	Find closest neighbors
      3.	Vote for labels
      
-Linear Regression:
                    Linear Regression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset,                     and the targets predicted by the linear approximation.

-Multimonial Naive Bayes:
                          It implements the naive Bayes algorithm for multinomial distributed data, and is one of the two classic naive Bayes variants used in text                                         classification (where the data are typically represented as word vector counts, although TF-IDF vectors are also known to work well in practice). 
                    
## Project Insights:

So, far in this project we have learned and read many things every member was assigned with multiple task's both for reading and Code. Scikit Learn was a big help and the refrences listed in the end also helped alot.Here are some insights for what we have learned from this project:

- Applying Filters of different Variants 
- Using Convolve Function & also making a convolve function
- Using different models to train Data 
- Testing data for desired Accuracy


## Code: 

This Section Contains importants function we have used through out he code:
```python 

#libraries we have used through out the code

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
import math

#function to perform convolution
def convolve2D(image, filter):
  fX, fY = filter.shape # Get filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) #new dimension of the reduced image size
  newImage = np.zeros((nn,nn)) #empty new 2D imange
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage
  
  
#Create Filter for convolution [9x9] we have changed the filters accordingly this is a 9x9 unweighted filter
filter = np.array([[1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
	        [1,1,1,1,1,1,1,1,1],
	        [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1]])

#convert from dataframe to numpy array
X = X.to_numpy()
print(X.shape)

#new array with reduced number of features to store the small size images
sX = np.empty((0,400), int)

ss = 500 #subset size for dry runs change to 42000 to run on whole data

#Perform convolve on all images
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,400))
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

#spliting train and test data
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)

#Sample code for Applying SVM on the Dataset:

svm_clf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svm_clf.fit(sXTrain,yTrain)
y_test_pred_svm = svm_clf.predict(sXTest)
s=metrics.accuracy_score(yTest, y_test_pred_svm)
print("Accuracy for SVM\n",s)

#Sample code for Applying Linear Regression on the Dataset:

reg = LinearRegression()
reg.fit(sXTrain,yTrain)
regYpred = reg.predict(sXTest)
regAcc = metrics.r2_score(yTest,regYpred) 
print('Linear Regression Accuracy: ', regAcc)

#Sample code for Applying KNN on the Dataset:

classifier = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
classifier.fit(sXTrain,yTrain)
Y_pred = classifier.predict(sXTest)
print(classification_report(yTest,Y_pred))
print(accuracy_score(yTest,Y_pred))

#Sample code for Applying MultimonialNB on the Dataset:

clf = MultinomialNB()
clf.fit(sXTrain, yTrain)
print(clf.class_count_)
print(clf.score(sXTest, yTest))

```
## Outputs

Here are outputs we have achieved within Google Colab:

 Filter Size   |   Techniques    |   Weighted Filter  |  Un-Weighted Filter  
-------------- | --------------- | ------------------ | -------------------
               |  SVM            |       0.9          |          0.89
               |  LinearReg      |     -1.2306        |        -2.700 
   *[5x5]*     |  KNN            |       0.83         |          0.82
               |  MultiMonialNB  |       0.81         |          0.82
-------------- | --------------- | ------------------ | -------------------
               |  SVM            |       0.89         |          0.9
               |  LinearReg      |     -2.4992        |        -3.6308 
   *[7x7]*     |  KNN            |       0.76         |          0.77
               |  MultiMonialNB  |       0.73         |          0.76
-------------- | --------------- | ------------------ | -------------------
               |  SVM            |       0.86         |          0.88
               |  LinearReg      |     -779.92        |         -30.94
   *[9x9]*     |  KNN            |       0.75         |          0.77
               |  MultiMonialNB  |       0.7          |          0.74


## References:
- [1] https://scikit-learn.org/stable/modules/cross_validation.html
- [2] https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- [3] https://scikit-learn.org/stable/modules/naive_bayes.html
- [4] https://inblog.in/Categorical-Naive-Bayes-Classifier-implementation-in-Python-dAVqLWkf7E

