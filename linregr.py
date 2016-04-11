# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 07:32:50 2016

@author: Kumar Suraj
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

rawtitanic = pd.read_csv("C:\\Users\\Kr_Suraj_Baranwal\\Desktop\\Academics\\DACS\\DA\\Titanic\\train.csv")
Fare = rawtitanic["Fare"][~np.isnan(rawtitanic["Age"])&~np.isnan(rawtitanic["Fare"])]
Age = rawtitanic["Age"][~np.isnan(rawtitanic["Age"])&~np.isnan(rawtitanic["Fare"])]
titanic = pd.DataFrame({"age" : Age, "fare" : Fare})
Age = np.reshape(Age, (len(Age),1))
Fare  = np.reshape(Fare,(len(Age),1))
regmodel = linear_model.LinearRegression()
regmodel.fit(Age,Fare)
plt.scatter(Age, Fare, color = 'black')
plt.plot(Age, regmodel.predict(Age), color = 'red', linewidth = 1)
test = pd.read_csv("C:\\Users\\Kr_Suraj_Baranwal\\Desktop\\Academics\\DACS\\DA\\Titanic\\test.csv") 
tAge = test["Age"][~np.isnan(test["Age"])&~np.isnan(test["Fare"])]
tFare = test["Fare"][~np.isnan(test["Age"])&~np.isnan(test["Fare"])]
plt.xticks()
plt.yticks()
plt.show()
tAge = np.reshape(tAge, (len(tAge),1))
tFare = np.reshape(tFare, (len(tFare),1))
plt.scatter(tAge, tFare, color = 'blue')
plt.plot(tAge, regmodel.predict(tAge), linewidth = 1, color = 'yellow')
plt.xticks()
plt.yticks()
plt.show()
regmodel.score(tAge, tFare, sample_weight = None)
