import pandas as pd
import datetime
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from yellowbrick.model_selection import ValidationCurve
from yellowbrick.classifier import ConfusionMatrix

print(datetime.datetime.now())

path = 'data/cleaned_data.csv'
randomState = 42

df = pd.read_csv(path, index_col=0)

print('Import done.')

# Extract labels from features
y = df['BK']
X = df.drop('BK', axis=1)

print(X.info())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)

# Benchmark using regular RFC
rf = RandomForestClassifier(n_estimators=50, random_state=randomState)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rf_score = balanced_accuracy_score(y_test, y_pred)
print('Benchmark Balanced Accuracy: ' + str(rf_score))
print(classification_report(y_test, y_pred, digits=2))
cm1 = ConfusionMatrix(rf, classes=['Not Bankrupt', 'Bankrupt'])
# cm1.score(X_test, y_test)
# cm1.show()

# Use a balanced RFC
brf = BalancedRandomForestClassifier(n_estimators=50, random_state=randomState)
brf.fit(X_train, y_train)

y_pred = brf.predict(X_test)
brf_score = balanced_accuracy_score(y_test, y_pred)
print('Balanced Accuracy: ' + str(brf_score))
print(classification_report(y_test, y_pred, digits=2))
cm2 = ConfusionMatrix(brf, classes=['Not Bankrupt', 'Bankrupt'])
# cm2.score(X_test, y_test)
# cm2.show()

# Try resampling
smote_enn = SMOTEENN(random_state=randomState)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
X_trainr, X_testr, y_trainr, y_testr = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=randomState)

brf2 = RandomForestClassifier(n_estimators=50, random_state=randomState)
brf2.fit(X_trainr, y_trainr)

y_pred2 = brf2.predict(X_testr)
brf_score2 = balanced_accuracy_score(y_testr, y_pred2)
print('reBalanced Accuracy: ' + str(brf_score2))
print(classification_report(y_testr, y_pred2, digits=2))
cm3 = ConfusionMatrix(brf2, classes=['Not Bankrupt', 'Bankrupt'])
cm3.score(X_testr, y_testr)
cm3.show()

# Generate a validation curve to find the efficiency point
# viz_rf = ValidationCurve(rf, param_name='n_estimators',
#                          param_range=np.arange(25, 150, 25), cv=4, scoring='f1_weighted')
#
# viz_rf.fit(X, y)
#
# viz_brf = ValidationCurve(brf, param_name='n_estimators',
#                           param_range=np.arange(25, 150, 25), cv=4, scoring='f1_weighted')
#
# viz_brf.fit(X, y)
#
# viz_rf.show()
# viz_brf.show()
