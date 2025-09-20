from src.data_loader import load_battery_data
from src.svr_model import svr_objective, train_svr
from src.abc_optimizer import ABCOptimizer
from src.evaluation import evaluate_model
import numpy as np

X_train, y_train, X_test, y_test = load_battery_data("B0005")

# Define search space for C, gamma, epsilon
bounds = [[1e-1, 1e3], [1e-4, 1], [0.01, 0.5]]

def fitness(params):
    return svr_objective(X_train, y_train, *params)

opt = ABCOptimizer(fitness, bounds, colony_size=20, max_iter=100)
best_params, best_fit = opt.optimize()
print("Best Params:", best_params)

model = train_svr(X_train, y_train, *best_params)
evaluate_model(model, X_test, y_test)
