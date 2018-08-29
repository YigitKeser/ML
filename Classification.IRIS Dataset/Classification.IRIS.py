# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:09:42 2018

@author: Yigit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#1 Data preproccessing
#1.1. Data Import
veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,:4].values #dependent variables
y = veriler.iloc[:,4].values  #independent variables

#1.2.Splitting data--train and test
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#1.3.Scaling data (Standardization)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#2 Classification algorithms
#2.1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #Train
y_pred = logr.predict(X_test)
#Classification Report
print('-----LR-----')
print(classification_report(y_test, y_pred))
#Confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#Score
LR_accscore=accuracy_score(y_test, y_pred)
print('LR Score:',LR_accscore)

#2.2. KNN ( K Nearest Neighbor)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski') # Default 5,minkowski
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print ('-----KNN-----')
print(classification_report(y_test, y_pred))
print(cm)
KNN_score=accuracy_score(y_test, y_pred)
print('KNN Score:',KNN_score)

#2.3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('-----SVC-----')
print(classification_report(y_test, y_pred))
print(cm)
SVC_score=accuracy_score(y_test, y_pred)
print('SVC Score:',SVC_score)
# 2.4.Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('-----GNB-----')
print(classification_report(y_test, y_pred))
print(cm)
GNB_score=accuracy_score(y_test, y_pred)
print('GNB Score:',GNB_score)
# 2.5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('-----DTC-----')
print(classification_report(y_test, y_pred))
print(cm)
DT_score=accuracy_score(y_test, y_pred)
print('DT Score:',DT_score)
# 2.6. Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('-----RFC-----')
print(classification_report(y_test, y_pred))
print(cm)
RFC_score=accuracy_score(y_test, y_pred)
print('RFC Score:',RFC_score)