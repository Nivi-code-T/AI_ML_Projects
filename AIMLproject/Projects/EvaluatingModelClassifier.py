#4 steps to Evaluate classifier
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

data=pd.read_csv(r"C:\Users\nsamd\Desktop\AIMLProjects\heart-disease.csv")
print(data)

#creat X and y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X=data.drop("target",axis=1)
y=data["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=RandomForestClassifier()
model.fit(X_train,y_train)
score=model.score(X_test,y_test)
print(score)

#Predict the model
y_pred=model.predict(X_test)
print(y_pred)

#Single training and test split score
clf_single_score=model.score(X_test,y_test)

#take the mean of cross validation score
clf_cross_val_score=np.mean(cross_val_score(model,X,y,cv=2))

#compare two
print(clf_single_score, clf_cross_val_score)

#1.Accuracy

X=data.drop("target",axis=1)
y=data["target"]

clf=RandomForestClassifier(n_estimators=100)
cross_val_score=cross_val_score(clf,X,y,cv=3)

print(np.mean(cross_val_score))
print(f"{np.mean(cross_val_score)*100:.2f}%")

#2.ROC curve area under receiver operating characteristics)
from sklearn.metrics import roc_curve
clf.fit(X_train,y_train)
y_probs=clf.predict_proba(X_test)
y_probs[:10]
print(y_probs[:10],len(y_probs))
y_probs_positive=y_probs[:,1]
print(y_probs_positive)

#Calculate fpr,tpr,and thresholds
fpr,tpr,thresholds=roc_curve(y_test,y_probs_positive)
print(fpr)

import matplotlib.pyplot as plt

def plot_roc_curve(fpr,tpr):
  plt.plot(fpr,tpr,color="orange",label="ROC")
  plt.plot([0,1],[0,1],color="darkblue",linestyle="--",label="Guessing")

  plt.xlabel("fpr")
  plt.ylabel("tpr")
  plt.title("ROC")
  plt.legend()
  plt.show()
plot_roc_curve(fpr,tpr)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_probs_positive)

#plot perfect ROC curve and AUC score
fpr,tpr,thresholds=roc_curve(y_test,y_test)
plot_roc_curve(fpr,tpr)
#perfect AUC score
roc_auc_score(y_test,y_test)

#Confusion metrics
from sklearn.metrics import confusion_matrix
y_preds=clf.predict(X_test)
confusion_matrix(y_test,y_preds)

#pandas cross tab
cm=pd.crosstab(y_test,y_preds,rownames=["Actual Label"],
            colnames=["Predicted Labels"])
print(cm)

import seaborn as sns
import sys
#set the font scale
sns.set(font_scale=1.5)

#Create a confusion matrix
conf_mat=confusion_matrix(y_test,y_pred)

#plot it using seaborn
sns.heatmap(conf_mat)

#Other way
#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_preds=clf.predict(X_test)
confusion_matrix(y_test,y_preds)

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(estimator=clf,X=X,y=y)
plt.show()














