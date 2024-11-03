import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load and shuffle the Iris dataset
iris = pd.read_csv("iris_dataset/iris.csv")
iris = shuffle(iris, random_state=0)  # Shuffle data for balanced distribution across splits
data = iris.drop("species", axis=1)   # Input features
label = iris["species"]               # Target output

# Custom train-validate-test split function
def train_validate_test_split(data, label, testRatio=0.3, valRatio=0.3):
    test_size = int(len(data) * testRatio)
    val_size = int(len(data) * valRatio)
    train_size = len(data) - test_size - val_size

    data_train, label_train = data[:train_size], label[:train_size]
    data_val, label_val = data[train_size:train_size + val_size], label[train_size:train_size + val_size]
    data_test, label_test = data[train_size + val_size:], label[train_size + val_size:]

    return data_train, data_val, data_test, label_train, label_val, label_test

data_train, data_val, data_test, label_train, label_val, label_test = train_validate_test_split(data, label)

print("Training Data:\n", data_train)# 60 row x 4 column (input)
print("Validation Data:\n", data_val)# 45 row x 4 column (input)
print("Test Data:\n", data_test)# 45 row x 4 column (input)
print("Training label:\n", label_train)# 60 row x 4 column (output --> actual input)
print("Validation label:\n", label_val)# 45 row x 4 column (output --> actual input)
print("Test Label:\n", label_test)# 45 row x 4 column (output --> actual input)

# Verify class distribution in splits
print("Class distribution in training labels:\n", label_train.value_counts())
print("Class distribution in validation labels:\n", label_val.value_counts())
print("Class distribution in test labels:\n", label_test.value_counts())

# Train a Gaussian Naive Bayes model
GuassianNBModel = GaussianNB()
GuassianNBModel.fit(data_train, label_train)

# Evaluate the model
print("GaussianNB Model train score:", GuassianNBModel.score(data_train, label_train))
print("GaussianNB Model validation score:", GuassianNBModel.score(data_val, label_val))
print("GaussianNB Model test score:", GuassianNBModel.score(data_test, label_test))

# Encode the class labels as numeric values
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(label)

# Define a function to plot the decision boundaries
def plot_decision_boundaries(data, label, classifier, feature1_index, feature2_index, feature_names):
    # Extract the specific features for plotting
    data_pair = data.iloc[:, [feature1_index, feature2_index]].values  # Use .values for compatibility with np meshgrid
    
    # Define the limits of the plot
    x_min, x_max = data_pair[:, 0].min() - 1, data_pair[:, 0].max() + 1
    y_min, y_max = data_pair[:, 1].min() - 1, data_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Train classifier and predict class probabilities for each point in the grid
    classifier.fit(data_pair, label_encoded)  # Train only on the two features for plotting
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])  # Predict for each point in grid
    Z = Z.reshape(xx.shape)  # Reshape for plotting
    
     # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(data_pair[:, 0], data_pair[:, 1], c=label_encoded, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature_names[feature1_index])
    plt.ylabel(feature_names[feature2_index])
    plt.title(f"Decision Boundaries ({feature_names[feature1_index]} vs {feature_names[feature2_index]})")
    plt.show()
# Load feature names from the dataset
feature_names = data.columns

# Iterate through each pair of features to plot decision boundaries
for (i, j) in combinations(range(4), 2):
    model = LogisticRegression()
    plot_decision_boundaries(data, label, model, i, j, feature_names)

# Define a function to calculate accuracy
def calculate_accuracy(predicted_y, y):
    count=0    
    for i,j in zip(predicted_y, y):
        if i==j:
            count+=1
        else:
              continue
    accuracy=(count/len(y))*100
    return accuracy
# Calculate and display test accuracy
pred_test = GuassianNBModel.predict(data_test)
test_accuracy = calculate_accuracy(pred_test, label_test)
print(f"Test Accuracy: {test_accuracy}%")