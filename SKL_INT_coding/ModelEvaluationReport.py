from sklearn.metrics import classification_report

from SKL_INT_coding.Train_LinearRegressionModel import y_test,y_preds

print(classification_report(y_test, y_preds))