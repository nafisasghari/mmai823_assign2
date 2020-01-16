#**************************#
#import packages
#**************************#
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.impute import SimpleImputer 

from sklearn.feature_selection import SelectKBest , chi2, f_classif

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix , classification_report, roc_auc_score, f1_score, accuracy_score
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

#**************************#
#import and Explore data
#**************************#
df = pd.read_excel("data/Bankruptcy_data_Final.xlsx")
df.head()

data = df.copy()

#Data Statistics
print(df.describe().transpose())

# Check missing value
print(df.isnull().sum())


# target distribution -- Class0: 99.4%, Class1 : 0.6%
print(df["BK"].value_counts())

#****************************#
#Data Cleaning and Engineering 
#****************************#
#Drop rows with more than 3 missing values
df = df[df.isnull().sum(axis=1) < 3]

#Fill remaining missing value with 0
df = df.fillna(0)

#Feature Scaling
df.drop(['Data Year - Fiscal' ], axis = 1 , inplace = True)

num_features = [col for col in df.columns if col != 'BK']

scaler = PowerTransformer(method='yeo-johnson')
df[num_features] = scaler.fit_transform(df[num_features])

#Histogram
df.hist(figsize = (13,13), bins = 20)
plt.show()

#Export the cleaned data
df.to_csv('data/cleaned_data.csv')