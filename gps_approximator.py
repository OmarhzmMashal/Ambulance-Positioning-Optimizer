import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob, random
import sklearn
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR

import math
import traces
import tensorflow as tf

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
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.linear_model import HuberRegressor
folder = os.path.dirname(os.path.abspath(__file__))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')
mm=MinMaxScaler()

def sub():
    ss=pd.read_csv(folder + "/SampleSubmission.csv")
    ss['Seq_ID'] = ss['Row_ID'].apply(lambda s: s.split(' X ')[0])
    ss['Time'] = ss['Row_ID'].apply(lambda s: s.split(' X ')[1][4:])
    ss['Col'] = ss['Row_ID'].apply(lambda s: s.split(' X ')[2])
    seq_ids = ss['Seq_ID'].unique() 
    ss['Time'] = ss['Time'].astype(int)
    ss = ss.sort_values(by='Time')

    model.fit(all_source_data,all_target_data)   
    for seq_id in seq_ids:
        test = pd.read_csv(folder+ f'/test_sequences/{seq_id}_source.csv')
        test = smooth(test)
        test = test[key_features]
        test['MA'] = test.Accuracy * test.Movement_Type
        test=test[key_features2]        
        
        pred = model.predict(test)
        ss.loc[(ss.Seq_ID == seq_id)&(ss.Col=='Latitude'), 'Prediction'] = pred[:,0]
        ss.loc[(ss.Seq_ID == seq_id)&(ss.Col=='Longitude'), 'Prediction'] = pred[:,1]
    
    ss.to_csv(folder + '/iieeu.csv', index=False)

def smooth(source):
    preds = pd.DataFrame({
        'Time':range(120),
        'Latitude': source.Latitude,
        'Longitude': source.Longitude,
        'Accuracy': source.Accuracy,
        'Movement_Type': source.Movement_Type
    })
    preds.loc[preds.Longitude.duplicated(), 'Accuracy']= (preds.loc[preds.Longitude.duplicated(), 'Accuracy'])*0.1
    preds.loc[preds.Latitude.duplicated(), 'Latitude']=np.nan
    preds.loc[preds.Longitude.duplicated(), 'Longitude']=np.nan 

    #INTERPOLATE
    preds.Latitude = preds.Latitude.astype(float).interpolate(method='polynomial',order=2)
    preds.Longitude = preds.Longitude.astype(float).interpolate(method='polynomial',order=2)
 
    preds.Latitude = preds.Latitude.astype(float).interpolate(method='linear')
    preds.Longitude = preds.Longitude.astype(float).interpolate(method='linear')
    
    return preds

##############################################################################
#model333 = XGBRegressor(random_state=1,learning_rate=0.5,
 #                    n_estimators=600)
#model3 = AdaBoostRegressor(n_estimators=500,learning_rate=0.5)
#model3 = CatBoostRegressor(silent = True, learning_rate=0.3, 
  #                         n_estimators=500)
model = LGBMRegressor(learning_rate=0.5,n_estimators=500)
#model = LGBMRegressor(learning_rate=0.5,n_estimators=1000)
#model = BaggingRegressor(base_estimator=CatBoostRegressor(silent=True, n_estimators=100,learning_rate=0.5),
#                           n_estimators=50, random_state=0) #1.37
#model = BaggingRegressor(base_estimator=CatBoostRegressor(silent=True, n_estimators=50,learning_rate=1),
#                           n_estimators=500, random_state=0) # testing
#model = BaggingRegressor(base_estimator=CatBoostRegressor(silent=True, n_estimators=100,learning_rate=0.5),
#                           n_estimators=100, random_state=0)
#model = VotingRegressor(estimators=[('a',reg1),('b',reg2)])
##############################################################################
sequences = sorted(glob.glob(folder + '/train_sequences/*source.csv'))
targets = sorted(glob.glob(folder + '/train_targets/*target.csv'))

all_source_data = []
all_target_data = []
key_features = ['Latitude','Longitude','Time','Movement_Type', 'Accuracy']
key_features2 = ['Latitude','Longitude','Time','MA']   
    
def rot_seq(s,t):
    source, target = s, t    
    source.loc[:,'Latitude'] = -1*source.loc[:,'Latitude']
    target.loc[:,'Latitude'] = -1*target.loc[:,'Latitude']
    source.loc[:,'Longitude'] = -1*source.loc[:,'Longitude']
    target.loc[:,'Longitude'] = -1*target.loc[:,'Longitude']
    return source, target

for i in tqdm(range(len(sequences))):
    source = pd.read_csv(sequences[i])
    target = pd.read_csv(targets[i])
    if source.shape[0] == 120:
        source = smooth(source)
        source=source[key_features]     
        target = target[key_features[:2]]

        source['MA'] = source.Accuracy * (source.Movement_Type)
        
        source=source[key_features2]       
        all_source_data.append(source)
        all_target_data.append(target)

        s2,t2=rot_seq(source,target)
        all_source_data.append(s2)
        all_target_data.append(t2)

all_source_data = np.concatenate(all_source_data, 0)
all_target_data = np.concatenate(all_target_data, 0)

X_train, X_test, y_train, y_test = train_test_split(all_source_data, all_target_data, 
                                                    test_size=0.3, random_state=42, shuffle=True)
model = MultiOutputRegressor(model)
model.fit(X_train, y_train)
val_preds = model.predict(X_test)
val_error = np.sqrt(mean_squared_error(y_test, val_preds))
print('validation error: ', val_error)

sub()