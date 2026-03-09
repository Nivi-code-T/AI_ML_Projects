from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X=[[1],[2],[3],[4]]
y=[1,2,3,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_preds=model.predict(X_test)
print(y_preds)

#note=DecisionTreeClassifier(criterion='gini',max_depth=3,min_samples_split=2)

