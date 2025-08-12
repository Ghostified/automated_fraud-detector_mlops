#%% import liblaries
import joblib
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#%% display the first rows of the data
data = pd.read_csv('Data/credit_card_fraud.csv')
print("First five rows")
print(data.head())

print("\nData shape:", data.shape)
print("\nColumns", data.columns.to_list())

#%% Convert to date time
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

#%% Extract useful features from dates
data['transaction_hour'] = data['trans_date_trans_time'].dt.hour
data['transaction_day'] = data['trans_date_trans_time'].dt.day_of_week #0=Monday
data['customer_age'] = (data['trans_date_trans_time'] - data['dob']).dt.days //365

#%%Drop original date columns or keep needed later
data.drop(['trans_date_trans_time', 'dob'], axis= 1, inplace=True)


# %% Data pre processing , handle missing values by filling with missing values for each column - only fill numeric columns 
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median(numeric_only=True))

#Fill categorical data with mode (or 'unknown')
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
  if data[col].nunique() < 100 : 
    data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown', inplace=True)
  else:
    data[col].fillna('unknown', inplace=True)



#convert categorical columns to numerical values using one hot encoding
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

#Normalize numerical columns for scaling
if 'amt' in data.columns:
  data['normalized_amt'] = (data['amt'] - data['amt'].mean()) / data['amt'].std()
  data.drop('amt', axis=1, inplace=True)

#separate features and target variables
X =  data.drop(columns=['is_fraud'])
y = data['is_fraud']

#split data into training and testing tests (80% train 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data pre-processing complete")

#%% Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate Model
print("Classification report:")
print(classification_report(y_test,y_pred))

print("Confussion Matrix:")
print(confusion_matrix(y_test, y_pred))

#%% plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d' , cmap='Blues' , xticklabels=['Non-fraud', 'Fraud'], yticklabels=['Non-fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#%%save model
joblib.dump(model, 'trained_model.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
print('Model and feature names saved')

