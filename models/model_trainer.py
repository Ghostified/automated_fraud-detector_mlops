#This  file takes the engineered data and split it to training and testing sets 
#It also trains two machine learning models XGBOOST and random forest and compares both
# After training it evaluates each model and saves the trained model to be used in the dashboard
 
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
  """Train Multiple models (XGBoost , Randm Forest) and saves them"""

  def __init__(self):
    self.models = {}
    self.X_test = None
    self.y_test = None

  def train(self, df):
    """
    Train multiple models on engineered data.
    Expects: df with features and is_fraud column
    """
    print("Starting model training...")

    #1. Separate features and targets
    if 'is_fraud' not in df.columns:
      raise ValueError("Target column 'is_fraud' not found")
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    print(f"...Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"...Fraud Rate: {y.mean()*100:.2f}%")

    #2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
    )
    self.X_test = X_test
    self.y_test = y_test

    #3. Train Random Forest
    print("")

