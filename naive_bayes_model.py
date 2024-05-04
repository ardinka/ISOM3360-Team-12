from sklearn.naive_bayes import GaussianNB


def train_naive_bayes_model(X_train, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model
