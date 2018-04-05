#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:17:47 2018

@author: aevarjohannesson
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

#Generate data. Check for Nan values.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.isnull().sum().sum() # there seem to be no NAN values.


loss=np.asarray(train['loss'])
train.drop('loss',axis=1,inplace=True)
con=train['cont2']
plt.scatter(con,np.log(loss))


sns.stripplot(x="cat1", y="loss", data=train);
sns.stripplot(x="cat2", y="loss", data=train);
sns.stripplot(x="cat3", y="loss", data=train);
sns.stripplot(x="cat4", y="loss", data=train);

##Correlation
dataCont = train.iloc[:,117:131]
dataCont.iloc[:,14]= np.log(dataCont.iloc[:,14]) #Continous variables.
corr = dataCont.corr()


sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# We can see that cont11 and cont 12 are highly correlated.


#Henda út einni af þessum.
con11=train['cont11'] 
con12=train['cont12']
con1=train['cont1']
con9=train['cont9']
plt.scatter(con11,con12)
plt.scatter(con1,con9)

train.drop(['cont11','cont9'],axis=1,inplace=True)


#Linear Regression:

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

n,p =train.shape # We have 188318 rows and 132 columns

y=loss
X=train.iloc[:,1:]
X = pd.get_dummies(X)  #ONE hot encoding of categorical variables.

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.20,
                                                    random_state=42)

regrModel = linear_model.LinearRegression(fit_intercept=False)

# Þjálfa modelið á train gögnunum mínum
regrModel.fit(X_train, y_train)

#  Skila Coefficients
#print('Coefficients: \n', regrModel.coef_)

regr_y_pred = regrModel.predict(X_test)

#Reikna MSE
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_y_pred)) # 967762933.49
