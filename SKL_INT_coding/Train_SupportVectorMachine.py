from sklearn.svm import SVC
from SKL_INT_coding.Train_LinearRegressionModel import X_train, y_train

model=SVC(kernel='linear')
model.fit(X_train,y_train)

