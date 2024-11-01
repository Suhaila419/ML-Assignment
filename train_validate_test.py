import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as pltImg
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = pd.read_csv("iris_dataset/iris.csv")
data = iris.drop("species", axis=1)  # input features ---> drop output feature
label = iris["species"]  # target output

#data_train,data_test,label_train,label_test=train_test_split(data,label,test_size=0.3,random_state=0,shuffle=True)#shuffle --->sort data
def train_validate_test_split(data, label, testRatio=0.3, valRatio=0.3):
    # Calculate split sizes
    test_size = int(len(data) * testRatio)
    val_size = int(len(data) * valRatio)
    train_size = len(data) - test_size - val_size

    # Split the data and labels
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

GuassianNBModel = GaussianNB()
GuassianNBModel.fit(data_train,label_train)
print("GuassianNBModel train score : ",GuassianNBModel.score(data_train,label_train))
print("GuassianNBModel validation score:", GuassianNBModel.score(data_val, label_val))
print("GuassianNBModel test score : ",GuassianNBModel.score(data_test,label_test))# --->overfitting

predictions = GuassianNBModel.predict(data_test)
print("\nClassification Report:\n", classification_report(label_test, predictions))