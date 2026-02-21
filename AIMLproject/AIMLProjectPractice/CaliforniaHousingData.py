from statistics import linear_regression

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
#Ftech California housing data
housing=fetch_california_housing()
print(housing);

#convert it in to dataframe
housing_df=pd.DataFrame(housing["data"],columns=housing["feature_names"])
print(housing_df)

#Add Target
housing_df["target"]=housing["target"]
print(housing_df)

#Import algorithm
#Create random seed
np.random.seed(20)
X=housing_df.drop("target",axis=1)
y=housing_df["target"]

#splitting data in to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#instantiate and fit the model
from sklearn.linear_model import Ridge
model=Ridge()
fit=model.fit(X_train, y_train)
score=model.score(X_test, y_test)

print(fit)
print(score)

#To improve model using ensemble models
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train, y_train)
s=model.score(X_test, y_test)
print(s)












