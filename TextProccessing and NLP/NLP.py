# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:02:31 2018

@author: Yigit
"""

import numpy as np
import pandas as pd

yorumlar= pd.read_csv('Restaurant_Reviews.csv')

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords
#Preproccesing
islenmis = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    islenmis.append(yorum)
#Feature Extraction
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(islenmis).toarray() #undependent
y = yorumlar.iloc[:,1].values #dependent

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33 ,random_state = 0)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #Train
y_pred = logr.predict(X_test)
print('-----LR-----')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)
LR_accscore=accuracy_score(y_test, y_pred)
print('LR Score:',LR_accscore)

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