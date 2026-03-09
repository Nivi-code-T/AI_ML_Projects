from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from SKL_INT_coding.Train_SupportVectorMachine import X_train,y_train

params = {'C':[1,10], 'kernel':['linear','rbf']}

grid = GridSearchCV(SVC(), params)

grid.fit(X_train, y_train)

print(grid.best_params_)