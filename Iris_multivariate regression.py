# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:36:00 2016

@author: Kumar Suraj
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as ptl
from matplotlib import cm

iris = pd.read_csv("C:\\Users\\Kr_Suraj_Baranwal\\Desktop\\Academics\\DACS\\DA\\iris\\iris.csv")
print (iris)

X = pd.DataFrame({'slength' : iris["SepalLengthCm"], 'swidth' :iris["SepalWidthCm"]})
print (X)

y = iris["PetalLengthCm"]
print (y)

multi_regmodel = linear_model.LinearRegression()
multi_regmodel.fit(X[0:-20],y[0:-20])
op = multi_regmodel.predict(X[:])
fig = ptl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[-20:]["slength"], X[-20:]["swidth"], y[-20:], color = 'blue')
ax.scatter(X[-20:]["slength"], X[-20:]["swidth"],op[-20:], color = "red")
xx = np.arange(min(X["slength"]), max(X["slength"]), 0.25)
yy = np.arange(min(X["swidth"]), max(X["swidth"]), 0.25)
xx, yy = np.meshgrid(xx, yy)
zz = multi_regmodel.coef_[0]*xx+ multi_regmodel.coef_[1]*yy+multi_regmodel.intercept_
ax.plot_surface(xx, yy, zz, cmap=cm.hot)
ax.set_xlabel('Sepal Length', size = 13)
ax.set_ylabel('Sepal Width'  , size = 13)
ax.set_zlabel('Petal Length', size = 13)
ptl.show()
ptl.subplots_adjust(left=0.10)

