from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_knn_model(X_train, y_train):
    param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    return knn_model


def predict_knn_model(model, X_test):
    return model.predict(X_test)

