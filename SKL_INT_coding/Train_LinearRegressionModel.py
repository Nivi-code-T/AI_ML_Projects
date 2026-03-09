#Task: Predict a continuous value using Linear Regression.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X=[[1],[2],[3],[4]]
y=[2,4,8,10]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=LinearRegression()
fit=model.fit(X_train,y_train)
#score=model.score(X_test,y_test)
y_preds=model.predict(X_test)
print(y_preds)