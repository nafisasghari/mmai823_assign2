import pandas as pd
import datetime
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from yellowbrick.model_selection import ValidationCurve
from yellowbrick.classifier import ROCAUC, ConfusionMatrix
from util_helper import resample_to_csv


print(datetime.datetime.now())

path = 'data/original_test.csv'
pathr = 'data/resampled_nn_train.csv'
pathr2 = 'data/resampled_borderline_train.csv'
pathr3 = 'data/resampled_adasyn_train.csv'
pathr4 = 'data/resampled_tomek_train.csv'
randomState = 42

df = pd.read_csv(path, index_col=0)
dfr = pd.read_csv(pathr4, index_col=0)

print('Import done.')

# Extract labels from features
y_test = df['BK']
yr_train = dfr['BK']
X_test = df.drop('BK', axis=1)
Xr_train = dfr.drop('BK', axis=1)

# Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)
# Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=randomState)

# Benchmark using regular RFC
rfr = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=randomState)
rfr.fit(Xr_train, yr_train)

# Using Resampled
estimators = np.arange(30, 60, 2)
depths = np.arange(2, 6, 1)
scores = []

# for est in estimators:
#     for dp in depths:
#         rfr = RandomForestClassifier(n_estimators=est, max_depth=dp, random_state=randomState)
#         rfr.fit(Xr_train, yr_train)
#         y_pred = rfr.predict(X_test)
#         rf_score = balanced_accuracy_score(y_test, y_pred)
#         print('Estimators: ' + str(est))
#         print('Depth: ' + str(dp))
#         print('AUC Score: ' + str(rf_score) + '\n')
#         scores.append(rf_score)
#
# df_scores = pd.DataFrame(columns=['n_estimators', 'max_depth', 'auc_score'])
# df_scores['n_estimators'] = estimators
# df_scores['max_depth'] = depths
# df_scores['auc_score'] = scores


# Predictions
y_pred = rfr.predict(X_test)
# yr_pred = rfr.predict(Xr_test)

# Balanced Accuracy Score
rf_score = balanced_accuracy_score(y_test, y_pred)
# rfr_score = balanced_accuracy_score(yr_test, yr_pred)

# AUC Score
rf_auc = roc_auc_score(y_test, y_pred)
# rfr_auc = roc_auc_score(yr_test, yr_pred)

# print('Benchmark Balanced Accuracy: ' + str(rf_score))
# print('Benchmark AUC Score:' + str(rf_auc))
# print(classification_report(y_test, y_pred, digits=3))

print('Resampled Balanced Accuracy: ' + str(rf_score))
print('Resampled AUC Score:' + str(rf_auc))
print(classification_report(y_test, y_pred, digits=3))

cm1 = ConfusionMatrix(rfr, classes=['Not Bankrupt', 'Bankrupt'])
# roc1 = ROCAUC(rf, classes=['Not Bankrupt', 'Bankrupt'])
# roc2 = ROCAUC(rfr, classes=['Not Bankrupt', 'Bankrupt'])
cm1.score(X_test, y_test)
# roc1.score(X_test, y_test)
# roc2.score(X_test, y_test)
cm1.show()
# roc1.show()
# roc2.show()

# Generate a validation curve to find the efficiency point
# rf = RandomForestClassifier(n_estimators=12, random_state=randomState)
# viz_rf = ValidationCurve(rf, param_name='max_depth',
#                          param_range=np.arange(8, 20, 1), cv=4, scoring='f1_weighted')
#
# viz_rf.fit(Xr_train, yr_train)
# viz_rf.show()
