import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

from util_helper import resample_to_csv

print(datetime.datetime.now())

path = 'data/cleaned_data.csv'
pathr = 'data/resampled_nn_train.csv'
pathr2 = 'data/resampled_borderline_train.csv'
pathr3 = 'data/resampled_adasyn_train.csv'
pathr4 = 'data/resampled_tomek_train.csv'
randomState = 42

df = pd.read_csv(path, index_col=0)

print('Import done.')

# Extract labels from features
y = df['BK']
X = df.drop('BK', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)
# X_test['BK'] = y_test
# X_test.to_csv('data/original_test.csv')

# Resample data
resample_to_csv(X_train, y_train, random_state=randomState, path=pathr, method='SMOTE-NN')
