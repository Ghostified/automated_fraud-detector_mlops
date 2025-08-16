# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# Load Model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load('trained_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, feature_columns

model, feature_columns = load_model()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Data/credit_card_fraud.csv')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔎 Filter Data")
st.sidebar.markdown("Adjust filters to explore the dataset.")

# Filter by amount
min_amt, max_amt = st.sidebar.slider("Transaction Amount", 0, int(df['amt'].max()), (0, 500))

# Filter by state
all_states = df['state'].unique()
selected_states = st.sidebar.multiselect("States", all_states, default=all_states[:3])

# Filter by fraud status
fraud_status = st.sidebar.radio("Fraud Status", ["All", "Non-Fraud (0)", "Fraud (1)"])

# Apply filters
filtered_df = df[(df['amt'] >= min_amt) & (df['amt'] <= max_amt)]
filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

if fraud_status == "Non-Fraud (0)":
    filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
elif fraud_status == "Fraud (1)":
    filtered_df = filtered_df[filtered_df['is_fraud'] == 1]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🛡️ Predict Fraud", "🧠 Model Info"])

# -------------------------------
# TAB 1: Overview
# -------------------------------
with tab1:
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases", f"{df['is_fraud'].sum()}", f"{df['is_fraud'].mean() * 100:.1f}%")
    col3.metric("Features", len(feature_columns))

    st.subheader("Sample of Filtered Data")
    st.dataframe(filtered_df[['amt', 'category', 'state', 'city', 'is_fraud']].head(10))

# -------------------------------
# TAB 2: Visualizations
# -------------------------------
with tab2:
    st.header("Fraud Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud by Category")
        category_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=category_fraud.values, y=category_fraud.index, ax=ax, palette="Reds_r")
        ax.set_xlabel("Fraud Rate")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Transactions by Hour")
        df['hour'] = df['trans_date_trans_time'].dt.hour
        hourly = df['hour'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.plot(hourly.index, hourly.values, color='skyblue', linewidth=2)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Transaction Count")
        st.pyplot(fig)
        plt.close(fig)

    # Heatmap: Fraud Rate by State
    st.subheader("Fraud Rate by State")
    state_fraud = df.groupby('state')['is_fraud'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=state_fraud.values, y=state_fraud.index, ax=ax, palette="coolwarm")
    ax.set_xlabel("Fraud Rate")
    st.pyplot(fig)
    plt.close(fig)

# -------------------------------
# TAB 3: Predict Fraud
# -------------------------------
with tab3:
    st.header("🛡️ Predict New Transaction")

    with st.form("prediction_form"):
        st.write("Enter transaction details:")
        amt = st.number_input("Amount ($)", min_value=0.0, value=99.99)
        state = st.selectbox("State", df['state'].unique())
        category = st.selectbox("Category", df['category'].unique())
        customer_age = st.slider("Customer Age", 18, 100, 35)
        transaction_hour = st.slider("Hour of Transaction", 0, 23, 12)
        city_pop = st.number_input("City Population", min_value=0, value=50000)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'amt': [amt],
            'customer_age': [customer_age],
            'transaction_hour': [transaction_hour],
            'city_pop': [city_pop],
            'lat': [35.0],
            'long': [-90.0],
            'merch_lat': [35.1],
            'merch_long': [-90.1],
        })

        # One-hot encode state and category
        for val in df['state'].unique():
            input_data[f"state_{val}"] = 1 if val == state else 0
        for val in df['category'].unique():
            input_data[f"category_{val}"] = 1 if val == category else 0

        # Ensure all model features are present
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_columns]  # Reorder to match training

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        confidence = max(prob) * 100

        # Display Result
        if pred == 1:
            st.error(f"🔴 FRAUD ALERT! Likely fraudulent transaction.")
            st.write(f"**Confidence:** {confidence:.1f}%")
        else:
            st.success(f"🟢 Safe transaction.")
            st.write(f"**Confidence:** {confidence:.1f}%")

# -------------------------------
# TAB 4: Model Info
# -------------------------------
with tab4:
    st.header("🧠 Model Information")
    st.markdown("""
    - **Model Type:** Random Forest Classifier
    - **Training Data:** Credit card transactions (simulated)
    - **Target Variable:** `is_fraud` (0 = safe, 1 = fraud)
    - **Features Used:** 60+ (after one-hot encoding)
    - **Accuracy:** ~99% on test set
    - **Purpose:** Educational demo for fraud detection

    ### 📂 Files Used:
    - `trained_model.pkl`: Trained Random Forest
    - `feature_columns.pkl`: Feature names for prediction
    """)

    st.code("""
    # To retrain:
    python model_script.py
    """, language='python')

    st.info("💡 Tip: Always match the input format to training data when predicting!")