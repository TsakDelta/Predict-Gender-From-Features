# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:21:30 2023

@author: Dimitris Tsakatsonis

project: Predict_gender_from_shoe_size(train)
"""

from sklearn import svm
from random import randint
import joblib

#Height(cm), Weight(kg), Shoesize(US)
#Gender 0: Male, 1: Female

n = 100_000
m_height = [165,183]
m_weight = [59,91]
m_shoe = [9,12]
f_height = [152,167]
f_weight = [50,75]
f_shoe = [6,8]

X = []
y = []

for i in range(n):
    j = randint(0,1)
    if j:
        h = randint(min(f_height), max(f_height))
        w = randint(min(f_weight), max(f_weight))
        s = randint(min(f_shoe), max(f_shoe))
        X.append([h,w,s])
        y.append(j)
    else:
        h = randint(min(m_height), max(m_height))
        w = randint(min(m_weight), max(m_weight))
        s = randint(min(m_shoe), max(m_shoe))
        X.append([h,w,s])
        y.append(j)
        
clf = svm.SVC()
clf.fit(X,y)

model_filename = "predict_gender_from_shoe_size.pkl"
joblib.dump(clf, model_filename)