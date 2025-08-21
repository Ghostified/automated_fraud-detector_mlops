import pandas as pd
import os
from config import load_config


class DataLoader:
  """Load and validates the fraud dataset"""

  def __init__(self):
    self.config = load_config()

  def load(self):
    """Load the csv and parse datetime columns"""
    path = self.config['data']['path']

    #check if the file exists
    if not os.path.exists(path):
      raise FileNotFoundError(f"Data file not found: {path}\n"
                              "Make sure 'data/credit_card_fraud.csv' exists in the project. ")
    
    print(f"Loading data from {path}...")
    try:
      df = pd.read_csv(path)
    except Exception as e:
      raise Exception(f"Failed to read csv: {e}")
    
    #Check required columns from the data
    required_cols = ['trans_date_trans_time', 'dob', 'is_fraud']
    for col in required_cols:
      if col not in df.columns:
        raise ValueError(f"Missing required column: '{col}' in dataset")
      

    #parse datetime columns
    try:
      df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
      df['dob'] = pd.to_datetime(df['dob'])
    except Exception as e:
      raise ValueError(f"Failed to parse date columns: {e}")
    

    print(f"Loaded {len(df)} transactions with {len(df.columns)} columns.")
    return df