#%% import liblaries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#%% display the first rows of the data
data = pd.read_csv('Data/credit_card_fraud.csv')
print(data.head())

# %% Data pre processing
#handle missing values by filling with missing values for each column
data.fillna(data.median(),inplace=True)

#convert categorical columns to numerical values using one hot encoding
data = pd.get_dummies(data, drop_first=True)

#Normalize numerical columns for scaling
data['normalized_amount'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()

#separate features and target variables
x =  data.drop(columns=['Class'])
y = data['Class']

#split data into training and testing tests (80% train 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data pre-processing complete")

#%% Train a model

