from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from fontTools.subset import subset
from pandas import Categorical
from sklearn.externals.array_api_extra import one_hot

from Sklearn import X_train, y_train

#Read Data from CSV
car_sales_missing=pd.read_csv("car-sales-extended-missing-data.csv")
print(car_sales_missing.head(40))

#Find Missing Values
Count_missing_value=car_sales_missing.isna().sum()
print(Count_missing_value)

#Drop rows with no label
car_sales_missing.dropna(subset=["Price"],inplace=True)
print(car_sales_missing.dropna(subset=["Price"],inplace=True))

#Split X and y
X=car_sales_missing.drop("Price",axis=1)
y=car_sales_missing["Price"]

#Now Fill the Missing data using SKlearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

cat_imputer=SimpleImputer(strategy="constant",fill_value="missing")
door_imputer=SimpleImputer(strategy="constant",fill_value=4)
num_imputer=SimpleImputer(strategy="mean")

#Define Columns
cat_feature=["Make","Colour"]
door_feature=["Doors"]
num_feature=["Odometer (KM)"]

#Fill the values
imputer=ColumnTransformer([("cat_imputer",cat_imputer,cat_feature),
                           ("door_imputer",door_imputer,door_feature),
                           ("num_imputer",num_imputer,num_feature)])

#fir and Transform
filled_X=imputer.fit_transform(X)
print(filled_X)

car_sales_filled=pd.DataFrame(filled_X,
                              columns=["Make","Colour","Doors","Odometer (KM)"])
print(car_sales_filled)


#Convert categorical data in to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

Categorical_feature=["Make","Colour","Doors"]
OneHot=OneHotEncoder()
transformer=ColumnTransformer([("OneHot",OneHot,Categorical_feature)],remainder="passthrough")
new_x=transformer.fit_transform(car_sales_filled)
print(new_x)

#Now Split X and y
X1=car_sales_filled.drop("Price",axis=1)
np.random.seed(20)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X1_train,y1_train,X1_test,y1_test=train_test_split(new_x,
                                                   y,test_size=0.2)
model=RandomForestRegressor()
model.fit(X1_train,y1_train)
model.score(X1_test,y1_test)













