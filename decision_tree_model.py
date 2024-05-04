from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_decision_tree_model(X_train, y_train):
    # Define the parameter grid for GridSearchCV
    try_grid = {
        'max_depth': range(1, 11),
        'max_leaf_nodes': range(5, 16),
        'min_samples_leaf': range(1, 6)
    }

    # Create the GridSearchCV model
    dtm = GridSearchCV(DecisionTreeClassifier(), param_grid=try_grid, cv=10)
    dtm.fit(X_train, y_train)

    # Extract the best parameters from the GridSearchCV model
    best_params = dtm.best_params_

    # Create the decision tree model using the best parameters
    decision_tree_model = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        max_leaf_nodes=best_params['max_leaf_nodes'],
        min_samples_leaf=best_params['min_samples_leaf']
    )

    # Train the decision tree model
    decision_tree_model.fit(X_train, y_train)

    return decision_tree_model
