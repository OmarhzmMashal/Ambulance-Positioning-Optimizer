import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, auc
from sklearn import metrics
import glob, random
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR

import math
import traces
import tensorflow as tf

# make a prediction with a stacking ensemble
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
import catboost
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.linear_model import HuberRegressor

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')
import category_encoders as ce

from sklearn.model_selection import KFold , GroupShuffleSplit
from sklearn.multioutput import MultiOutputClassifier

folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(folder+"/german_credit_data.csv")   

#COLUMNS
cols = []
for i in range (1, df.shape[1]-1):
    cols.append(df.iloc[:, i].name)
print(cols)

#MOST FREQ WITHIN JOB GROUP
df.loc[df['Checking account'].isna(), 'Checking account'] = df.groupby('Job')['Checking account'].transform(lambda x: x.mode()[0] if any(x.mode()) else 'ALL_NAN')
df.loc[df['Saving accounts'].isna(), 'Saving accounts'] = df.groupby('Job')['Saving accounts'].transform(lambda x: x.mode()[0] if any(x.mode()) else 'ALL_NAN')
        
#MOST FREQ
#most_freq = df['Checking account'].mode()[0]
#df['Checking account'] = df['Checking account'].replace(np.nan, most_freq)
#most_freq = df['Saving accounts'].mode()[0]
#df['Saving accounts'] = df['Saving accounts'].replace(np.nan, most_freq)


for i in range (df.shape[1]):
    if df.iloc[:,i].isna().sum() != 0:
        print(df.iloc[:,i].name)
        print(df.iloc[:,i].isna().sum())
        
#ORDINAL ENCODING FOR TARGET
ordi = ce.OrdinalEncoder(cols=['Risk'], return_df=True)
df = ordi.fit_transform(df)

#TARGET ENCODING
te = ce.TargetEncoder(cols=['Purpose', 'Housing', 'Sex', 'Checking account', 'Saving accounts'],return_df=True)
df = te.fit_transform(df, df['Risk'])

#ONE HOT ENCODING
#te = ce.OneHotEncoder(cols=['Risk'],return_df=True)
#df = te.fit_transform(df)

target = df['Risk']
fea = df.drop(columns=['Risk'])

print(df)
n_fold=5
kf = KFold(n_splits=n_fold, shuffle=True)
#kf.get_n_splits(fea)
preds=0
models = []
fea = np.asarray(fea)
target = np.asarray(target)

#print(target)
for train_index, test_index in kf.split(fea):
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = fea[train_index], fea[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    #model = MultiOutputClassifier(CatBoostClassifier(verbose=0))
    model = CatBoostClassifier(verbose=0)
    model.fit(x_train, y_train)    
    models.append(model)
    
    print('catboost train accuracy: ' + str(model.score(x_train, y_train)))
    print('catboost test accuracy: ' + str(model.score(x_test, y_test)))
    
print('\n')

for i in range(n_fold):
    preds += models[i].predict_proba(x_test)

cla=[]
for i in range(len(preds)) :
    cla.append(np.argmax(preds[i])+1)    
    
fpr, tpr, thresholds = metrics.roc_curve(y_test, cla, pos_label=2)
auc = metrics.auc(fpr, tpr)
print(auc)