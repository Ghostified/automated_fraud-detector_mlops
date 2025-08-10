import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ghost/Projects/Python/automated_fraud_detector/credit_card_fraud.csv')

#display the first rows
%%
print(df.head())