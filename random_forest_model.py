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
    return grid_search
