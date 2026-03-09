from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X=[[1],[2],[3],[4]]
y=[1,2,3,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
y_preds=model.predict(X_test)
print(y_preds)

