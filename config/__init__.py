#Thisfiles loads the config.yaml file and makes it available as a python dictionary

import yaml

def load_config():
  """Load configuration from the  config.yaml"""
  with open('config/config.yaml', 'r') as f:
    return yaml.safe_load(f)