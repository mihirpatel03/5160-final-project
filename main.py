from knn import knn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn import knn


def main():

    # Generate a random dataset
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1, n_samples=200, random_state=42, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    k_pred_arr = []
    plotter_arr = []
    for i in range(10):
    # Initialize and train classifier
        plotter = knn(i+1,X_train, X_test, y_train, y_test)
        plotter.fit()
        plotter.epsilon()
        plotter.beta()
        #plotter.plot_decision_boundaries()
        k_pred_arr.append(plotter.evaluate())
        plotter_arr.append(plotter)

    print(k_pred_arr)

    total, correct = 0,0
    for i in range(len(X_test)):
        pred_sum = 0
        for j in range(len(plotter_arr)):
            model_pred = plotter_arr[j].predict(X_test)[i]
            if model_pred<=0:
                model_pred = -1
            
            pred_sum+=model_pred*(plotter_arr[j].bet)

        pred=0
        if pred_sum>0:
            pred = 1
        if pred==y_test[i]:
            correct+=1
        total+=1
    
    print("boosting accuracy is: %s" % (correct/total))

    plt.plot(range(1,11),k_pred_arr,color="blue",label = "kNN test accuracy")
    plt.axhline(y=correct/total, color='black', linestyle='--', label = "boosting test accuracy")
    plt.xlabel("Number of Nearest Neighbors (k)")
    plt.ylabel("Prediction Accuracy")
    plt.title("Number of Nearest Neighbors (k) vs Prediction Accuracy")
    plt.show()


if __name__ == "__main__":
    main()

