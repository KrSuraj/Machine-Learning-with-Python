# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:31:02 2016

@author: Kumar Suraj
"""
#Loading libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn import svm
import os

#loading dataset
cdr = os.getcwd()
os.chdir("C:\\Users\\Kr_Suraj_Baranwal\\Desktop\\Academics\\DACS\\DA\\Project\\Titanic")
#loading the data
df = pd.read_csv('train.csv')
df.drop(['Ticket', 'Cabin'], axis = 1)
df.dropna()

formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'

plt.figure(figsize=(8,6))

y, x = dmatrices(formula_ml, data=df, return_type='matrix')

feature_1 = 2
feature_2 = 3

X = np.asarray(x)
X = X[:,[feature_1, feature_2]] 

y = np.asarray(y)

y = y.flatten()

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

#training dataset

n_cv = int(.9 * n_sample)
X_train = X[:n_cv]
y_train = y[:n_cv]
X_test = X[n_cv:]
y_test = y[n_cv:]

ktypes = ['linear', 'rbf', 'poly']
color_map = plt.cm.RdBu_r

for fig_num, kernel in enumerate(ktypes):
    clf = svm.SVC(kernel=kernel, gamma = 3)
    clf.fit(X_train, y_train)
    plt.figure(fig_num)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap= color_map)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)  
    plt.pcolormesh(XX, YY, Z>0, cmap= color_map)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-.5, 0, .5])
    plt.title(kernel)
    plt.show()
    
    
    
    

# Train Data modification
