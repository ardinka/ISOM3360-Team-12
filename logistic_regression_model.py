from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np


def train_logistic_regression_model(X_train, y_train):
    # Build the logistic regression model
    try_grid = {"C": np.logspace(-4, 4, 40), "penalty": ["l1", "l2"]}
    logicReg = LogisticRegression(solver='liblinear', max_iter=1000)

    # Define your Model using GridSearchCV
    lr_gridsearch = GridSearchCV(logicReg, try_grid, cv=10, scoring='accuracy')
    lr_gridsearch.fit(X_train, y_train)

    # Extract the best parameters from the GridSearchCV model
    best_params = lr_gridsearch.best_params_

    # Create the logistic regression model using the best parameters
    lr_model = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        C=best_params['C'],
        penalty=best_params['penalty']
    )

    # Train the logistic regression model
    lr_model.fit(X_train, y_train)

    return lr_model