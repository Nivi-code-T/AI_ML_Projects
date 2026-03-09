from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from SKL_INT_coding.Build_LogisticRegressionClassifier import X,y

model = LogisticRegression()

scores = cross_val_score(model, X, y, cv=5)

print(scores)