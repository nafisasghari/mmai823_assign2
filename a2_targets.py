import pandas as pd
import datetime
from yellowbrick.target import ClassBalance

print(datetime.datetime.now())

path = 'data/cleaned_data.csv'
pathr = 'data/resampled.csv'
pathr2 = 'data/resampled_borderline.csv'
pathr3 = 'data/resampled_adasyn.csv'
pathr4 = 'data/resampled_tomek.csv'
randomState = 42
classLabels = ['Not Bankrupt', 'Bankrupt']

df = pd.read_csv(pathr4, index_col=0)

print('Import done.')

# Extract labels from features
y = df['BK']
X = df.drop('BK', axis=1)

# Instantiate Visualizer
viz = ClassBalance(labels=classLabels)

viz.fit(y)
viz.show()
print(viz.support_)
