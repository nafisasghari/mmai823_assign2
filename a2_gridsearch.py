import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


path = 'data/original_test.csv'
pathr = 'data/resampled_nn_train.csv'
pathr2 = 'data/resampled_borderline_train.csv'
pathr3 = 'data/resampled_adasyn_train.csv'
pathr4 = 'data/resampled_tomek_train.csv'
randomState = 42

# Import data
df = pd.read_csv(path, index_col=0)
dfr = pd.read_csv(pathr2, index_col=0)

# Separate labels
y_test = df['BK']
yr_train = dfr['BK']
X_test = df.drop('BK', axis=1)
Xr_train = dfr.drop('BK', axis=1)

# Grid Search
parameters = {'n_estimators': [15, 16, 17], 'max_depth': [8, 9, 10, 11]}
gscv = GridSearchCV(RandomForestClassifier(random_state=randomState), parameters, scoring='balanced_accuracy')
gscv.fit(Xr_train, yr_train)
print(gscv.best_params_)
print(gscv.best_score_)
