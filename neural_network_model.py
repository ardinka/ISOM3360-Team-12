from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def train_neural_network_model(X_train, y_train, hidden_layer_sizes=(100,)):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (200,)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
        # Add more hyperparameters and their respective values as needed
    }

    mlp = MLPClassifier()
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    mlp_best = MLPClassifier(hidden_layer_sizes=grid_search.best_params_['hidden_layer_sizes'],
                             activation=grid_search.best_params_['activation'],
                             alpha=grid_search.best_params_['alpha'])
    mlp_best.fit(X_train, y_train)
    return mlp_best
