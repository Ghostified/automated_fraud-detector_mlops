import pandas as pd
from config import load_config


class DataLoader:
  def __init__(self):
    self.config = load_config()

  def load(self):
    path = self.config['data']['path']
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    print(f"Loaded {len(df)} transactions.")
    return df