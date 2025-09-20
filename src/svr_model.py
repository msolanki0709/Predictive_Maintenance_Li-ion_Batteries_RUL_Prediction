from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

def train_svr(X_train, y_train, C, gamma, epsilon):
    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)
    return model

def svr_objective(X_train, y_train, C, gamma, epsilon):
    model = train_svr(X_train, y_train, C, gamma, epsilon)
    y_pred = model.predict(X_train)
    return mean_squared_error(y_train, y_pred)
