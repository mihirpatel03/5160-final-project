from sklearn.utils import resample
from scipy.stats import mode
import numpy as np
from sklearn.metrics import accuracy_score



class BaggedKNN:
    def __init__(self, base_model, n_estimators, n_neighbors, X_train, y_train):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []
        self.X_train = X_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors

        # Initialize all models
        for _ in range(self.n_estimators):
            model = self.base_model(n_neighbors, None, None, None, None)
            self.models.append(model)

    def fit(self):
        for model in self.models:
            # Bootstrap sample
            X_bootstrap, y_bootstrap = resample(self.X_train, self.y_train)
            model.X_train = X_bootstrap
            model.y_train = y_bootstrap
            model.fit()

    def predict(self, X_test):
        predictions = []

        # Collect predictions from all models
        for model in self.models:
            predictions.append(model.predict(X_test))

        # Transpose to shape (n_samples, n_estimators) and find the mode (most common class)
        predictions = np.array(predictions).T
        final_predictions, _ = mode(predictions, axis=1)

        return final_predictions.ravel()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Ensemble Accuracy: {accuracy:.2f}")
        return accuracy
