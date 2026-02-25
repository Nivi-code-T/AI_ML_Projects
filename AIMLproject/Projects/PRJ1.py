#Make Predictions
import numpy as np
import pandas as pd

import os

from IPython.terminal.shortcuts.filters import pass_through

print(os.getcwd())

car_sales = pd.read_csv(r"C:\Users\nsamd\Desktop\AIMLProjects\car-sales-extended.csv")
print(car_sales)
print(car_sales.value_counts())
from sklearn.model_selection import train_test_split, cross_val_score

#Create X and y
X=car_sales.drop("Price",axis=1)
y=car_sales["Price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


#Change categorical value in to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cate_features=["Make","Colour","Doors"]
one_hot=OneHotEncoder()
Transform=ColumnTransformer([("one_hot",one_hot,cate_features)],remainder="passthrough")
newX=Transform.fit_transform(X)
print(newX)

#Convert it in to DF
car_sales_df=pd.DataFrame(newX)
print(car_sales_df)

#Refit the model
from sklearn.ensemble import RandomForestRegressor
X_train,X_test,y_train,y_test=train_test_split(newX,y,test_size=0.2)
model=RandomForestRegressor()
model.fit(X_train,y_train)
score=model.score(X_test,y_test)
print(score)

#predict
y_preds=model.predict(X_test)
print(y_preds)

from sklearn.metrics import r2_score

print(r2_score(y_test, y_preds))

print(y_preds[:10])

print(len(y_test))
print(len(y_preds))

#Visualisation
# import matplotlib.pyplot as plt
# #plt.scatter(X_test,y_test)
# plt.scatter(y_test,y_preds)
# plt.show()

#Compare prediction through the truth
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_preds))

#Evalute the model
cross_single_value=model.score(X_test,y_test)
cross_val_score=np.mean(cross_val_score(model,newX,y,cv=2))

#compare
print(cross_single_value,cross_val_score)




