#dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


#set page config
st.set_page_config(page_title="Fraud detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Dahboard")

#Load Model and features
@st.cache_resource
def load_model():
  model = joblib.load('trained_model.pkl')
  feature_columns = joblib.load('feature_columns.pkl')
  return model, feature_columns

model, feature_columns = load_model()

#load data
@st.cache_data
def load_data():
  df = pd.read_csv('Data/credit_card_fraud.csv')
  df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
  df['dob'] = pd.to_datetime(df['dob'])
  return df

df = load_data()

#sidebar filter
st.sidebar.header("Filter Data")
st.sidebar.markdown("Adjust filter to explore dataset")

#Filter by amount
min_amt, max_amt = st.sidebar.slider("Transaction Amount", 0 , int(df['amt'].max()), (0, 500))

#filter by state
all_states = df['state'].unique()
selected_state = st.sidebar.multiselect("States", all_states, default=all_states[:3])

#Filter by fraud status
fraud_status = st.sidebar.radio("Fraud Status", ["All", "Non Fraud(0)", "Fraud (1)"])

#Apply filters
filtered_df = df[(df['amt'] >= min_amt) & (df['amt'] <= max_amt)]
filtered_df = filtered_df[filtered_df['state'].isin(selected_state)]

if fraud_status == "Non-Fraud (0)":
  filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
elif fraud_status == "Fraud (1)":
  filtered_df = filtered_df[filtered_df['is_fraud'] == 1]

  #Tabs
  tab1, tab2, tab3, tab4 = st.tabs("Overview" , "Visualizations" , "Predic Fraud" , "Model Info")
#--------------------
# TAB 1 : Overview
#--------------------
with tab1:
  st.header("Dataset Overview")
  col1, col2, col3 = st.columns(3)
  col1.metric("Total Transactions", f"{len(df):,}")
  col2.metric("Fraud Cases", f"{df['is_fraud'].sum()}", f"{df['is_fraud'].mean()*100:.1f}%")
  col3.metric("Features", len(feature_columns))

  st.subheader("Sample of Filtered Data")
  st.dataframe(filtered_df[['amt', 'category', 'state', 'city', 'is_fraud']].head(10))
  
#=====================
# TAB 2 : Visualiztion
#=====================
with tab2:
  st.header("Fraud Insights")

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Fraud By Category")
    category_fraud = df.groupby('Category')['is_fraud'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=category_fraud.values, y=category_fraud.index, ax=ax, palette="Reds_r")
    ax.set_xlabel("Fraud Rate")
    st.pyplot(fig)

    with col2:
      st.subheader("Transactions by Hour")
      df['hour'] = df['trans_date_trans_time'].dt.hour
      hourly = df['hour'].value_counts().sort_index()
      fig, ax = plt.subplots()
      ax.plot(hourly.index, hourly.values, color='skyblue', linewidth=2)
      ax.set_xlabel("Hour of Day")
      ax.set_ylabel("Transaction count")
      st.pyplot(fig)
    
    st.subheader("Fraud Rate By State")
    state_fraud = df.groupby('state')['is_fraud'].mean()sort_values(ascending=False).head(10)