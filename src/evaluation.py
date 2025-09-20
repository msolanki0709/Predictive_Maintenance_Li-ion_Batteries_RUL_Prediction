from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    perc_err = np.mean(np.abs(y_pred - y_test) / y_test) * 100

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, %Error: {perc_err:.2f}%")
    return mae, rmse, perc_err
