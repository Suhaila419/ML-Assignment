import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# make_regression ----> create a dataset for testing or illustrating linear regression models
#  noise---> to make the data more realistic.
data, label = make_regression(n_samples=1000, n_features=3, noise=1, random_state=0)

data_train , data_test, label_train, label_test = train_test_split(data,label,test_size=0.3,random_state=0)# Shuffle data for balanced distribution across splits


matrix_X_Train=np.array(data_train)
matrix_X_Test=np.array(data_test)
# Create a column of ones with the same number of rows as matrix_X
ones_column_Train = np.ones((matrix_X_Train.shape[0], 1))
ones_column_Test = np.ones((matrix_X_Test.shape[0], 1))

# insert columns ones in matrix_x
X = np.column_stack((ones_column_Train, matrix_X_Train))
X_Test = np.column_stack((ones_column_Test, matrix_X_Test))

# X transpose
X_transpose=X.T

# X_transpose * X   ---> square matrix
square_matrix=X_transpose.dot(X)

# inverse (X_transpose * X) 
inverse_square_matrix=np.linalg.inv(square_matrix)

# (X_transpose * X) ^ inverse * X_transpose 
inv_Tran=inverse_square_matrix.dot(X_transpose)

# W=(((X^T)*X)^-1)*(X^T)* Y 
W=inv_Tran.dot(label_train)

# prediction to compute error
predict_y=X_Test.dot(W)

def compute_loss(predicted_y, y): 
    loss=0.5*(np.mean((y - predicted_y) ** 2))
    return loss
print("Loss = ",compute_loss(predict_y,label_test))