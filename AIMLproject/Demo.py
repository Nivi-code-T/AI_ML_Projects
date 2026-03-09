import pandas as pd
from numpy.ma.core import filled
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


df=load_iris()
print(df)
dfnew=pd.DataFrame(df.data,columns=df.feature_names)
print(dfnew.head(10))
DecisionTreeClassifier(criterion="gain",
                       max_depth=3,
                       min_samples_split=2)

dfnew["target"] = df.target
print(dfnew.head(10))

print(df.target_names)
print(df.feature_names)

X=dfnew.drop("target",axis=1)
y=dfnew["target"]
print(X,y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=DecisionTreeClassifier()
fit=model.fit(X_train,y_train)
score=model.score(X_test,y_test)
y_preds=model.predict(X_test)
accuracy=accuracy_score(y_test,y_preds)

print(fit,score)

#Visualization
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(10,6))
tree.plot_tree(model,filled=True)
plt.show()

