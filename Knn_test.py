import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class knn:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.radius_classifier = RadiusNeighborsClassifier(radius=0.75)
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        self.cmap_bold = ['darkorange', 'c', 'darkblue']

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier.fit(X_train, y_train)
        unique_classes = np.unique(y_train)
        self.num_classes = len(unique_classes)
        self.cmap_light = ListedColormap(plt.cm.viridis(np.linspace(0, 1, self.num_classes)))
        self.cmap_bold = plt.cm.viridis(np.linspace(0, 1, self.num_classes))

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def plot_decision_boundaries(self, X_test, y_test):
        # Set up mesh grid
        h = 0.02  # step size in the mesh
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict the function value over the grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Start plotting
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=self.cmap_light)
        
        # Plot the training points
        sns.scatterplot(x=self.X_train[:, 0], y=self.X_train[:, 1], hue=self.y_train,
                        palette=list(self.cmap_bold), alpha=1.0, edgecolor="black",
                        legend=False, label='Training Data (Circle)', marker='o')
        
        # Plot the test points
        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test,
                        palette=list(self.cmap_bold), alpha=0.6, edgecolor="black",
                        legend=False, label='Testing Data (X)', marker='X')
        
        # Customize legend
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = [f"Training Data (Circle)" if 'Circle' in label else f"Testing Data (X)" for label in labels]
        plt.legend(handles, labels, title="Data Type")
        
        # Add plot limits and titles
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"{self.num_classes}-Class classification (k = {self.n_neighbors})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy


    def confusion_score(self):
        """
        Calculate the confusion score for each point as the number of unique classes divided by
        the total number of neighbors within a 1-unit radius.
        """
        # Retrieve the indices of neighbors for each training point within the specified radius (1 unit here).
        neighbors = self.radius_classifier.radius_neighbors(self.X_train, return_distance=False)

        # Initialize an empty list to store the confusion scores for each training point.
        confusion_scores = []

        # Iterate through each set of neighbor indices (neigh_indices) for each training point.
        for idx, neigh_indices in enumerate(neighbors):
            # Check if there are any neighbors within the radius.
            if neigh_indices.size > 0:
                # Retrieve the class labels of the neighbors.
                classes = self.y_train[neigh_indices]
                # Count the number of unique class labels among the neighbors.
                unique_classes_count = np.unique(classes).size
                # Get the total number of neighbors.
                total_neighbors = classes.size
                # Calculate the confusion score: number of unique classes divided by total neighbors.
                score = unique_classes_count / total_neighbors
            else:
                # If no neighbors are found within the radius, assign a confusion score of 0.
                score = 0
            # Append the calculated score to the list of confusion scores.
            confusion_scores.append(score)

        # Convert the list of confusion scores to a numpy array and return it.
        return np.array(confusion_scores)

    def store_predictions_as_vec(self):
        """
        Return an array that includes each training point's coordinates rounded to two decimals,
        true label, predicted label as integers, and confusion score rounded to two decimals.
        """
        # Calculate the confusion score using the custom method
        confusion_scores = self.confusion_score()

        # Stack the training data points, true labels, predicted labels, and confusion scores
        data = np.column_stack((self.X_train, self.y_train, self.train_predictions, confusion_scores))

        # Round coordinates and confusion scores to two decimal places
        data[:, [0, 1, -1]] = np.round(data[:, [0, 1, -1]], 2)

        # Convert true and predicted labels to integers
        data[:, [2, 3]] = data[:, [2, 3]].astype(int)

        input_data = data[:, :3]
        predicted_label = data[:,3]
        confusion_score = data[:,4]


        return input_data,predicted_label,confusion_score


# Create your dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, n_samples=200, random_state=42, n_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train classifier
classifier = knn(n_neighbors=6)
classifier.fit(X_train, y_train)

# Plot decision boundaries using training data
classifier.plot_decision_boundaries(X_test, y_test)

# Evaluate classifier on test data
classifier.evaluate(X_test, y_test)