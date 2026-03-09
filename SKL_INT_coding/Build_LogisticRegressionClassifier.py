#Use Logistic Regression for binary classification.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X=[[1],[2],[3],[4]]
y=[2,4,8,10]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=LogisticRegression()
model.fit(X_train,y_train)
y_preds=model.predict(X_test)
print(y_preds)
