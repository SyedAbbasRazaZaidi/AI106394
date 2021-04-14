# AI106394 :  Digit Recognizer Kaggle Readiness  

### PROJECT MEMBERS:

StdID     |     Name
----------| -------------
63805     | **Syed Abbas Raza Zaidi**
63583     | Amta Nadeem
61363     | Syeda Mahrukh Zehra



## Project Description:
In this project we are using scikit learn libraries to cross validate data predict it's accuracy according for desired result.We are submitting th output file in Kaggle Digit Recognizer. In first step we are performing cross validation of data by using "train.CSV".By using Pandas we are reading our CSV file and checking the data frames.then we are splitting our training data and test data. In the next step we are using linear regression for prediction and catogorizing data by giving it a score.
we are using built-in models provided by scikit learn such as Gaussian Naive Bayes , Bernoulli Naive Bayes  and Multinomial Naive Bayes. In the final Step we are writing our data into another .csv so we can submit it on Kaggle.


## Project Insights:

## Code: 
```python
#importing needed Libraries

import pandas as pd
import numpy as nmpy
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import CategoricalNB

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC

import scipy.stats as ss
import seaborn as sb

import math

#reading csv Filewith panda builtin function pan.read_csv and framing the data 

df=pd.read_csv("/content/sample_data/train.csv")
df
df.columns
df["label"].value_counts()
df.isnull().sum() #checking if any data frame is null


X = df.drop(["label"], axis=1) #removing specified data by using drop function 
y = df["label"]
X = X / 255

#Spliting Data into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#training data by using builtin linearregression function provided by scikit learn
reg = LinearRegression()
reg.fit(X_train, y_train)

#testing our data by giving it a score
y_test_pred_reg = reg.predict(X_test)
metrics.r2_score(y_test, y_test_pred_reg)

#using gaussian Naive Bayes using different models to get the best accuracy so it can be used further 
gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
accuracy_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
print("Accuracy by Gaussian: ",accuracy_gnb)

#Now using bernoulli Naive Bayes for same purpose we are using GaussianNB
bnb = BernoulliNB()
y_pred_bnb = bnb.fit(X_train, y_train).predict(X_test)
accuracy_bnb = metrics.accuracy_score(y_test, y_pred_bnb)
print(" Accuracy by Bernoulli: ",accuracy_bnb)

#Now using MultiNomial Naive Bayes for same purpose we are using GaussianNB
mnb = MultinomialNB()
y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)
accuracy_mnb = metrics.accuracy_score(y_test, y_pred_mnb)
print(" Accuracy by MultinomialNB: ",accuracy_mnb)

#using C-Support Vector classification by specifying kernel type on which algorithm is running, 
#using verbose to take advantage of a per-process runtime setting, controlling random number genration for shuffling data
svm_clf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svm_clf.fit(X_train, y_train)

#checking if the sample data x is equal to predicted data Y 
y_test_pred_svm = svm_clf.predict(X_test)
metrics.accuracy_score(y_test, y_test_pred_svm)

#reading File test.csv 
test=pd.read_csv("/content/sample_data/test.csv")
test=test/255
svmFinalpred=svm_clf.predict(test)

#writing the final predictions to abbas_SData.csv file using panda
finalPred=pd.DataFrame(svmFinalpred,columns=["Label"])
finalPred['ImageId']=finalPred.index+1
finalPred = finalPred.reindex(['ImageId','Label'], axis=1)
finalPred.to_csv('abbas_SData.csv',index=False)
```
## Output

### 1.
![colabout](https://user-images.githubusercontent.com/61627416/114673334-81133a00-9d1f-11eb-85ac-dc3292ef48e6.PNG)

### 2.
![file](https://user-images.githubusercontent.com/61627416/114673545-bddf3100-9d1f-11eb-9633-e5d47382e129.PNG)


### Kaggle Score

![abbasOutput](https://user-images.githubusercontent.com/61627416/114662651-81f19f00-9d12-11eb-9806-37b7bc11cac8.PNG)



## References:
- [1] https://scikit-learn.org/stable/modules/cross_validation.html
- [2] https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- [3] https://scikit-learn.org/stable/modules/naive_bayes.html
- [4] https://inblog.in/Categorical-Naive-Bayes-Classifier-implementation-in-Python-dAVqLWkf7E

