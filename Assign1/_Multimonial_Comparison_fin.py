#I have used another Naive Bayes Approach Multimonial and after that i have compared the result to our desired Accuracy at the last i have made a CSV file for 
submission on kaggle.I further completed code of my group members Amta and Mahrukh After combining efforts and code of all our members 
i have compiled a final code file which is also on github kindly view it. Thanks

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

#Now using MultiNomial Naive Bayes for same purpose we are using GaussianNB
Multinomial = MultinomialNB()
y_pred_mnb = Multinomial.fit(X_train, y_train).predict(X_test)
accuracy_mnb = metrics.accuracy_score(y_test, y_pred_mnb)
print(" Accuracy by MultinomialNB: ",accuracy_mnb)

#using C-Support Vector classification by specifying kernel type on which algorithm is running, 
#using verbose to take advantage of a per-process runtime setting, controlling random number genration for shuffling data
svm_clf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svm_clf.fit(X_train, y_train)

#checking if the sample data x is equal to predicted data Y 
y_test_pred_svm = svm_clf.predict(X_test)
#predicting score
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