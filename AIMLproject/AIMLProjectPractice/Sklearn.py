import pandas as pd
import numpy as np
from matplotlib.pyplot import figure

from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing()
print(housing)

housing_df=pd.DataFrame(housing["data"],columns=housing["feature_names"])
print(housing_df)

housing_df["target"]=housing["target"]
print(housing_df.head())

#ridge won't fit or didn't get accurate result
#try to fit different model (Regressor model)

from sklearn.ensemble import RandomForestRegressor
#set up random seed
np.random.seed(42)

#Create X and y
X=housing_df.drop("target",axis=1)
y=housing_df["target"]

#split into Train and test data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=RandomForestRegressor(n_estimators=10)
model.fit(X_train,y_train)

score=model.score(X_test,y_test)
print(score)


#####################################################################
#using Ridge algorithm
#import algorithm
#set up random seed
np.random.seed(42)

#create X and y
X=housing_df.drop("target",axis=1)
y=housing_df["target"]

#split into train and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#fit the model on the training set

model=Ridge()
model.fit(X_train,y_train)
#check score on the model
score=model.score(X_test,y_test)

print(score)

import matplotlib.pyplot as plt
y_pred=model.predict(X_test)
plt.figure(figsize=(5,5))
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.metrics import accuracy_score
ascore = accuracy_score(y_test,y_pred)
print(ascore)








