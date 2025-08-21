"""
This file transforms raw transaction data into ai friendly features that help the model to spot fraud e.g transactions far from home, huge amount of transactions etv
"""
import pandas as pd
from haversine import haversine 


class FeatureEngineer:
  """Transforma raw data int ML-Ready features for fraud detection"""

  def engineer(self, df):
    """Apply feature engineering pipeline. """
    print("Starting the feature engineering...")

    #1. Extract time - based features
    df = self._add_time_features(df)

    #2. Calculate distance between customer and merchant
    df = self._add_distance_features(df)

    #3. Drop columns that are useless i.e high cardinality
    df = self._drop_unnecessary_columns(df)

    #4. Handle all missing values 
    df = self._fill_missing_values(df)

    #5. convert categories (like 'grocery_pos) into numbers
    df = self._encode_categorical_columns(df)

    #6. Normalize transactions amount
    df = self._normalize_amount(df)

    print("Feature engineering completeed")
    return df
  
  def _add_time_features(self, df):
    """Add hour of day, day of week, and customer age."""
    print("  adding time features...")
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
    df['transaction_day'] = df['trans_date_trans_time'].dt.dayofweek #0=Monday
    df['customer_age'] = df['trans_date_trans_time'].dt.days
    return df
  
  def _add_distance_feature(self, df):
    """Add distance in KM between the customer and merchant"""
    print("...Calculating distance from homre")

    def calculate_distance(row):
      customer = (row['lat'], row['long'])
      merchant = (row['merch_lat'], row['merch_long'])
      return haversine(customer, merchant)
    
    df['distance_km'] = df.apply(calculate_distance, axis=1)
    return df
  
  def _drop_unnecessary_columns(self, df):
    """Remove columns that cause  memory issues"""
    cols_to_drop = [
      'merchant',  #too many unique values  == memory explosions
      'job',   #high cardinality
      'trans_num',  #unique per transaction == useless
      'city',  #high cardinality
      'trans_date_trans_time',  #replaced with time features(day, time)
      'dob'  #replaced with customer age
    ]
    df.drop(colums=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df
  
  def _fill_missing_values(self, df):
    """Fill missing values in numeric and categorical columns"""
    print(" Filling missing values....")

    #Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    #Fill categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
      df[col] = df[col].fillna('unknown')

    return df
  
  def _encode_categorical_columns(self, df):
    """convert text categories (e.g 'grocery_pos) into numbers."""
    print("...encoding categorical columns.")
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df
  
  def _normalize_amount(self, df):
    """Scale 'amt' to have a mean=0, std=1"""
    if 'amt' in df.columns:
      print("...normalizing transaction amount")
      df['normalized_amt'] = (df['amt'] - df['amt'].mean()) / df['amt'].std()
      df.drop('amt', axis=1, inplace=True)
    return df