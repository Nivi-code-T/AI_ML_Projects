from sklearn.model_selection import train_test_split

from SKL_INT_coding.Train_LinearRegressionModel import X,y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)