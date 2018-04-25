# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data and train/test data
processeddata_path = os.path.join(os.path.pardir,'data','processed')
X = pd.read_csv(os.path.join(processeddata_path,'X.csv')).values
y = pd.read_csv(os.path.join(processeddata_path,'y.csv')).values


# Fitting Decision Tree Regression to the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)

# Checking the result through scatter plot
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Decision Tree Regression Results')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

# To make the curve continous we will increase the resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression Results')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()