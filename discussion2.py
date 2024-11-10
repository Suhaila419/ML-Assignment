import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate regression dataset
data, label = make_regression(n_samples=1000, n_features=3, noise=1, random_state=0)

# Split data into training and testing sets
data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=0)

# Convert data to numpy arrays and add a column of ones for the bias term
matrix_X_Train = np.array(data_train)
matrix_X_Test = np.array(data_test)

# Add a column of ones for the bias term
ones_column_Train = np.ones((matrix_X_Train.shape[0], 1))
ones_column_Test = np.ones((matrix_X_Test.shape[0], 1))

# Concatenate ones column to the feature matrices
X = np.column_stack((ones_column_Train, matrix_X_Train))
# Initialize weights randomly
weights = np.random.rand(X.shape[1])
# Set learning rate and number of iterations
learning_rate = 0.0001
iterations = 100
# Loss function based on Equation 1
def compute_loss(X, y, weights):
    residuals =np.mean( (y - X.dot(weights))**2)
    loss = 0.5 * residuals  # Equation 1
    return loss

# Gradient descent function using Equations 2 and 3
def gradient_descent(X, y, weights, learning_rate, iterations):
    for i in range(iterations):
        # Calculate gradient using Equation 2
        gradient = X.T.dot(X.dot(weights) - y)  # Equation 2
        
        # Update weights using Equation 3
        weights -= learning_rate * gradient  # Equation 3
        
        # Print loss every 10 iterations
        if i % 10 == 0:
            loss = compute_loss(X, y, weights)
            print(f"Iteration {i}: Loss = {loss}")
    
    return weights
# Train the model
final_weights = gradient_descent(X, label_train, weights, learning_rate, iterations)
predict_y = X_Test.dot(final_weights)

print("Optimal weights after training:", final_weights)
def MSE(predicted_y, y):
    mse = (1 / len(y)) * np.sum((y - predicted_y) ** 2)
    return mse
print("Mean Squared Error equal:", MSE(predict_y, label_test))
def MAE(predicted_y, y):
    mae = (1 / len(y)) * np.sum(np.abs(y - predicted_y))  # Absolute Error
    return mae
print("Mean Absolute Error equal:", MAE(predict_y, label_test))
# Optional: Ratio of MSE/MAE
print("Mean Squared Error over Mean Absolute Error:", MSE(predict_y, label_test) / MAE(predict_y, label_test))