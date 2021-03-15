import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback, EarlyStopping
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras import layers
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering,AffinityPropagation,MeanShift,SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as dates
from keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import warnings
import torch
from torch import tensor
import math 
from tqdm import tqdm
import haversine
from numba import vectorize, jit, cuda
import numba.cuda 
warnings.filterwarnings('ignore')
from math import ceil
from torch.utils.data import DataLoader
from scipy.stats import iqr
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dictances
from dictances import bhattacharyya
import seaborn as sn
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import datetime
from sklearn.cluster import DBSCAN

def score(sub, ref):
    total_distance = 0
    for date, c_lat, c_lon in ref[['datetime', 'latitude', 'longitude']].values:
        row = sub.loc[sub.date < date].tail(1) # Prior to Oct 2 this was incorrectly .head(1)
        dists = []
        for a in range(6):
            dist = ((c_lat - row[f'A{a}_Latitude'].values[0])**2+(c_lon - row[f'A{a}_Longitude'].values[0])**2)**0.5 
            dists.append(dist)
        total_distance += min(dists)
    return total_distance

def loss_fn(crash_locs, amb_locs):    
  dists_split = crash_locs-amb_locs[0] 
  dists = (dists_split[:,0]**2 + dists_split[:,1]**2)**0.5  
  min_dists = dists
  for i in range(1, 6):
    # Update dists so they represent the dist to the closest ambulance
    dists_split = crash_locs-amb_locs[i]  
    dists = (dists_split[:,0]**2 + dists_split[:,1]**2)**0.5
    min_dists = torch.min(min_dists, dists)
  return min_dists.mean()

#LOAD TRAIN DATA
folder = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(folder  + "/train.csv")
train['datetime'] = pd.to_datetime(train['datetime'], errors='coerce')
#ADD WEEK DAY
train['dayname'] = train.datetime.dt.strftime("%A")
train['day'] = le.fit_transform(train.dayname)
#ADD MONTH 
train['month'] = train.datetime.dt.month
#ADD WEEK NUMBER
train['week'] = np.ceil(train.datetime.dt.day/7.0)
#ADD HOUR RANGES
i=0
train['hour']=0
for hour in range(8):    
      train['hour'][(train.datetime.dt.hour >= hour*3) &
                 (train.datetime.dt.hour < hour*3 + 3)] = i
      i=i+1

print((train['month'].value_counts().sort_values().sort_values()))
print((train['day'].value_counts().sort_values().sort_values()))
print((train['hour'].value_counts().sort_values().sort_values()))

#group months into 7 clusters
K = KMeans(n_clusters= 7)
ll = train[['month']]
K.fit(ll)
pred= K.predict(ll)
train['month_cluster'] = pred

#CREATE SUBMISSION DATAFRAME AND TEST RANGE
reference = train.loc[train.datetime > '2019-01-01']
dates = pd.date_range('2019-01-01', '2020-01-01', freq='3h')
sub = pd.DataFrame({
      'date':dates
      })
#ADD MONTH INDEX
sub['month'] = sub.date.dt.month
#MERGE
#ADD DAYS
sub['day'] = sub.date.dt.strftime("%A")
sub.day = le.fit_transform(sub.day)
#ADD HOUR RANGE INDEX
i=0
sub['hour']=0
for hour in range(8):    
      sub['hour'][(sub.date.dt.hour >= hour*3) &
                 (sub.date.dt.hour < hour*3 + 3)] = i
      i=i+1

#merge to assign month cluster number to the sub file
sub = pd.merge(sub, train, how='left', on=['month','day','hour'])
sub = sub.drop_duplicates(subset=['date'])
sub= sub.interpolate(method='nearest')

for i in range(6): 
     sub['A'+str(i)+'_Latitude'] =  0.0
     sub['A'+str(i)+'_Longitude'] = 0.0
    
r =torch.randn(6, 2) * 0.04
amb_locs = r + tensor([-1.27,36.85])
amb_locs.requires_grad_()

for m in tqdm(range(7)):  
          crash_locs = tensor(train[['latitude', 'longitude']][(train.month_cluster == m)].values)
          for i in range(1000):
             # if i % 100 == 0 : lr -= 1e-3
              lr = 0.05
              loss= loss_fn(crash_locs, amb_locs)  
              loss.backward()  
              amb_locs.data -= lr * amb_locs.grad.data
              amb_locs.grad = None          
          amb_locs_np =  amb_locs.cpu().detach().numpy()
          for u in range(6):
              sub['A'+str(u)+'_Latitude'][(sub.month_cluster== m)]  = amb_locs_np[u][0] 
              sub['A'+str(u)+'_Longitude'][(sub.month_cluster == m)]  = amb_locs_np[u][1]  
              
for d in tqdm(range(7)):  
          crash_locs = tensor(train[['latitude', 'longitude']][(train.day == d)].values)
          for i in range(1000):
             # if i % 100 == 0 : lr -= 1e-3
              lr = 0.05
              loss= loss_fn(crash_locs, amb_locs)  
              loss.backward()  
              amb_locs.data -= lr * amb_locs.grad.data
              amb_locs.grad = None          
          amb_locs_np =  amb_locs.cpu().detach().numpy()
          for u in range(6):
              sub['A'+str(u)+'_Latitude'][(sub.day== d)]  += amb_locs_np[u][0] 
              sub['A'+str(u)+'_Longitude'][(sub.day == d)]  += amb_locs_np[u][1]  
              
for h in tqdm(range(8)):      
          crash_locs = tensor(train[['latitude', 'longitude']][(train.hour == m)].values)
          for i in range(1000):
             # if i % 100 == 0 : lr -= 1e-3
              lr = 0.05
              loss= loss_fn(crash_locs, amb_locs)  
              loss.backward()  
              amb_locs.data -= lr * amb_locs.grad.data
              amb_locs.grad = None          
          amb_locs_np =  amb_locs.cpu().detach().numpy()
          for u in range(6):
              sub['A'+str(u)+'_Latitude'][(sub.hour== h)]  += amb_locs_np[u][0] 
              sub['A'+str(u)+'_Longitude'][(sub.hour == h)]  += amb_locs_np[u][1]  

for u in range(6):
    sub['A'+str(u)+'_Latitude']/=3
    sub['A'+str(u)+'_Longitude']/=3

s=score(sub, reference)
sub = sub.loc[(sub.date >= '2019-07-01') & (sub.date < '2020-01-01')]
sub=sub.drop(columns=['day','hour','month'])
print('Score:' + str(s))
sub.to_csv(folder+'/submission.csv')