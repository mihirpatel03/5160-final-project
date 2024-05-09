import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

class knn:
    def __init__(self, n_neighbors, X_train, X_test, y_train, y_test):
        self.n_neighbors = n_neighbors

        #sklearn KNN 
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.radius_classifier = RadiusNeighborsClassifier(radius=0.75)
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        self.cmap_bold = ['darkorange', 'c', 'darkblue']

        #datasets
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        #fitting the knn model?
        self.classifier.fit(self.X_train, self.y_train,)
        unique_classes = np.unique(self.y_train)
        self.num_classes = len(unique_classes)
        self.cmap_light = ListedColormap(plt.cm.viridis(np.linspace(0, 1, self.num_classes)))
        self.cmap_bold = plt.cm.viridis(np.linspace(0, 1, self.num_classes))

        self.radius_classifier.fit(self.X_train, self.y_train)  # Fit the radius classifier

    def predict(self, test_list):
        #make a set of predictions on the test set
        return self.classifier.predict(test_list)

    def evaluate(self):
        #return the accuracy of the predictions on the test set
        y_pred = self.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy