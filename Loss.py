import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and shuffle the Iris dataset
iris = pd.read_csv("iris_dataset/iris.csv")
data = iris.drop("species", axis=1)   # Input features
label = iris["species"]               # label --> output for actual input

data_train , data_test, label_train, label_test = train_test_split(data,label,test_size=0.3,random_state=0)# Shuffle data for balanced distribution across splits

# Train a Gaussian Naive Bayes model
GuassianNBModel = GaussianNB()
GuassianNBModel.fit(data_train, label_train)

# loss =0.5 * (y-y)^2
x=data_train.values
matrix_X_Train=np.array(x)
matrix_X_Test=np.array(data_test)
# Create a column of ones with the same number of rows as matrix_X
ones_column_Train = np.ones((matrix_X_Train.shape[0], 1))
ones_column_Test = np.ones((matrix_X_Test.shape[0], 1))

# insert columns ones in matrix_x
X = np.column_stack((ones_column_Train, matrix_X_Train))
X_Test = np.column_stack((ones_column_Test, matrix_X_Test))

# X transpose
X_transpose=X.T

# X_transpose * X  5 row x 105 column * 105 row x 5 column ---> square matrix (5 row x 5 column)
square_matrix=X_transpose.dot(X)

# inverse (X_transpose * X) 5 row x 5 column
inverse_square_matrix=np.linalg.inv(square_matrix)

# (X_transpose * X) ^ inverse * X_transpose ---> 5 row x 5 column * 5 row x 105 column ---> 5 row x 105 column
inv_Tran=inverse_square_matrix.dot(X_transpose)

# Encode the class labels as numeric values
label_encoder = LabelEncoder()
label_encoded_test = label_encoder.fit_transform(label_test)
label_encoded_train = label_encoder.fit_transform(label_train)

# W=(((X^T)*X)^-1)*(X^T)* Y ----> 5 row x 105 column * 105 row x 1 column --> 5 row x 1 column
W=inv_Tran.dot(label_encoded_train)

# prediction to compute error
predict_y=X_Test.dot(W)

def compute_loss(predicted_y, y): 
    loss=0.5*((np.subtract(y,predicted_y))**2)
    return sum(loss)
print(compute_loss(predict_y,label_encoded_test))