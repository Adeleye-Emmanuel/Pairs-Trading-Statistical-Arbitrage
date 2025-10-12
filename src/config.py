import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ETFS_DIR = os.path.join(DATA_DIR, 'etfs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')