from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Train the Random Forest model
def train_random_forest_model(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rfm = RandomForestClassifier()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rfm, param_grid=param_grid, cv=5)
    # Train the model with grid search
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from the GridSearchCV model
    best_params = grid_search.best_params_

    # Create the decision tree model using the best parameters
    random_forest_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap']
    )

    # Train the decision tree model
    random_forest_model.fit(X_train, y_train)

    return random_forest_model
