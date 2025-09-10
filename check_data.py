import pandas as pd
import numpy as np
import os

if os.path.exists('new_data.csv'):
    df = pd.read_csv('new_data.csv')
    print('New data shape:', df.shape)
    print('New data columns (first 5):', df.columns[:5].tolist())
    print('New data columns (last 5):', df.columns[-5:].tolist())
    print('Data types:', df.dtypes.unique())
else:
    print('new_data.csv does not exist yet')
