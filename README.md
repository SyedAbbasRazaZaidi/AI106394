# AI106394 :  Digit Recognizer Kaggle Readiness  

### PROJECT MEMBERS:

StdID     |     Name
----------| -------------
63805     | **Syed Abbas Raza Zaidi**
63583     | Amta Nadeem
61363     | Syeda Mahrukh Zehra



## Project Description:





## Project Insights:

## Code: 
```
import pandas as pan
import numpy as nmpy
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import scipy.stats as s
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import CategoricalNB

df=pan.read_csv("train.csv")
df

df.columns
df["label"].value_counts()

df.isnull().sum()

df.info()

X = df.drop(["label"], axis=1)
y = df["label"]
X = X / 255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_test_pred_reg = reg.predict(X_test)
metrics.r2_score(y_test, y_test_pred_reg)

gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
bnb = BernoulliNB()

y_pred_bnb = bnb.fit(X_train, y_train).predict(X_test)
mnb = MultinomialNB()
y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)

accuracy_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
accuracy_bnb = metrics.accuracy_score(y_test, y_pred_bnb)
accuracy_mnb = metrics.accuracy_score(y_test, y_pred_mnb)
print("Accuracy of GaussianNB: ",accuracy_gnb," Accuracy of BernoulliNB: ",accuracy_bnb," Accuracy of MultinomialNB: ",accuracy_mnb)

svm_clf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svm_clf.fit(X_train, y_train)

y_test_pred_svm = svm_clf.predict(X_test)

metrics.accuracy_score(y_test, y_test_pred_svm)

test=pan.read_csv("test.csv")
test=test/255
svmFinalpred=svm_clf.predict(test)

finalPred=pan.DataFrame(svmFinalpred,columns=["Label"])

finalPred['ImageId']=finalPred.index+1
finalPred = finalPred.reindex(['ImageId','Label'], axis=1)

finalPred.to_csv('abbas_SData.csv',index=False)
```


## References:
- [refrence link1]

