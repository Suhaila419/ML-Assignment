import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target classes
feature_names = iris.feature_names
class_names = iris.target_names

# Define a function to plot the decision boundaries
def plot_decision_boundaries(X, y, classifier, title, feature1, feature2):
    # Define the limits of the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict class probabilities for each point in the grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(title)
    plt.show()

# Iterate through each pair of features
for (i, j) in combinations(range(4), 2):
    # Extract two features
    X_pair = X[:, [i, j]]
    
    # Train Logistic Regression model on the two features
    model = LogisticRegression()
    model.fit(X_pair, y)
    
    # Plot the decision boundary for this pair of features
    plot_title = f"Decision Boundaries ({feature_names[i]} vs {feature_names[j]})"
    plot_decision_boundaries(X_pair, y, model, plot_title, feature_names[i], feature_names[j])
