from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from SKL_INT_coding.Train_SupportVectorMachine import X_train,y_train

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])

pipeline.fit(X_train, y_train)