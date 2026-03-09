from sklearn.metrics import confusion_matrix

from SKL_INT_coding.Train_LinearRegressionModel import y_test,y_preds

cm=confusion_matrix(y_test,y_preds)

