import pandas as pd
import datetime

from util_helper import resample_to_csv

print(datetime.datetime.now())

path = 'data/cleaned_data.csv'
pathr = 'data/resampled.csv'
pathr2 = 'data/resampled_borderline.csv'
pathr3 = 'data/resampled_adasyn.csv'
pathr4 = 'data/resampled_tomek.csv'
randomState = 42

df = pd.read_csv(path, index_col=0)

print('Import done.')

# Extract labels from features
y = df['BK']
X = df.drop('BK', axis=1)

# Resample data
resample_to_csv(X, y, random_state=randomState, path=pathr4, method='tomek')
