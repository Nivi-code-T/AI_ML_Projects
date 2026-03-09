from sklearn.model_selection import KFold

from SKL_INT_coding.Train_LinearRegressionModel import X

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    print(train_index, test_index)