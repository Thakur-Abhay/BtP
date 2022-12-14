#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import date, datetime, timedelta
# import library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import ruptures as rpt

def statistical_features(arr):
    vmin = np.amin(arr)
    vmax = np.amax(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    return vmin, vmax, mean, std

def shape_features(arr):
    skewness = skew(arr)
    kurt = kurtosis(arr)
    return skewness, kurt

user = 'DF'

import_path =r"C:\Users\abhay\Dropbox\My PC (LAPTOP-CPO2A5NR)\Desktop\7th sem\BTP\DF\DF"
export_path = r"C:\Users\abhay\Dropbox\My PC (LAPTOP-CPO2A5NR)\Desktop\7th sem\BTP\DF\DF"

rightEDAdatapath  = f'{import_path}/EDA.csv'
rightHRdatapath   = f'{import_path}/HR.csv'
rightTEMPdatapath = f'{import_path}/TEMP.csv'

rightHRdatapath = np.loadtxt(rightHRdatapath, delimiter = ',')
rightHRdatapath = np.repeat(rightHRdatapath, 4)
np.savetxt('hr_new.csv', rightHRdatapath, delimiter = ',')


edat =  pd.read_csv(f'{user}/DF/EDA.csv',  header = 2,  names = ['EDA'])
hrt =   pd.read_csv(f'hr_new.csv',      header = 12, names = ['HR'])
# tempt = pd.read_csv(f'{user}/DF/Temp.csv', header = 2,  names = ['TEMP'])

gt =    pd.read_csv(f'{user}/DF/EDA.csv',  nrows  = 1)


reference_time = gt.iloc[0,0]
min_len = min(len(edat), len(hrt))

eda = edat.iloc[:min_len, 0]
hrt = hrt.iloc[:min_len, 0]
# tempt = tempt.iloc[:min_len, 0]
df_original = pd.concat([eda, hrt], axis = 1)


array = df_original.values
data = pd.DataFrame(array)
data.columns = ['EDA','HR']

cols = [
    'EDA_Mean','EDA_Min','EDA_Max','EDA_Std', 'EDA_Kurtosis', 'EDA_Skew','EDA_Num_Peaks','EDA_Amphitude','EDA_Duration',
    'HR_Mean','HR_Min','HR_Max','HR_Std','HR_RMS'
]
df_features = pd.DataFrame(columns=cols)

index = 0



for i in range(0,len(data['EDA']), 20):
    df_partial = data.iloc[i:i+40,]
    plen = len(df_partial['EDA'])
    
    if plen < 40:
        continue
    
    eda = df_partial['EDA'].values
    hr = df_partial['HR'].values

    eda_min, eda_max, eda_mean, eda_std = statistical_features(eda)
    hr_min, hr_max, hr_mean, hr_std = statistical_features(hr)
    eda_skew, eda_kurtosis = shape_features(eda)
    
    hr_rms = np.sqrt(np.mean(np.square(np.ediff1d(hr))))

    peaks,properties = find_peaks(eda, width=5)
    num_Peaks = len(peaks)
    
    prominences = np.array(properties['prominences'])
    widths = np.array(properties['widths'])
    amphitude = np.sum(prominences)
    duration = np.sum(widths)

    df_features.loc[index] = [eda_mean, eda_min, eda_max, eda_std, eda_kurtosis, eda_skew, num_Peaks, amphitude, duration, hr_mean, hr_min, hr_max, hr_std,hr_rms]

    index = index+1



df_features.shape


# In[14]:


# cols = list(map(str, range(30, 0, -1)))
# df_lag_features = pd.DataFrame(columns=cols)
# df_lag_features = pd.concat([
#     df_features['HR_Mean'].shift(10),  df_features['HR_Mean'].shift(9),    df_features['HR_Mean'].shift(8),
#     df_features['HR_Mean'].shift(7),   df_features['HR_Mean'].shift(6),    df_features['HR_Mean'].shift(5),
#     df_features['HR_Mean'].shift(4),   df_features['HR_Mean'].shift(3),    df_features['HR_Mean'].shift(2),
# #     df_features['HR_Mean'].shift(1),   df_features['temp_Mean'].shift(10), df_features['temp_Mean'].shift(9),
# #     df_features['temp_Mean'].shift(8), df_features['temp_Mean'].shift(7),  df_features['temp_Mean'].shift(6),
# #     df_features['temp_Mean'].shift(5), df_features['temp_Mean'].shift(4),  df_features['temp_Mean'].shift(3),
# #     df_features['temp_Mean'].shift(2), df_features['temp_Mean'].shift(1),  df_features['EDA_Mean'].shift(10),
#     df_features['EDA_Mean'].shift(9),  df_features['EDA_Mean'].shift(8),   df_features['EDA_Mean'].shift(7),
#     df_features['EDA_Mean'].shift(6),  df_features['EDA_Mean'].shift(5),   df_features['EDA_Mean'].shift(4),
#     df_features['EDA_Mean'].shift(3),  df_features['EDA_Mean'].shift(2),   df_features['EDA_Mean'].shift(1)], axis=1)
# df_lag_features.columns = cols
# df_lag_features = df_lag_features.dropna()
df_features.head()


# In[19]:


df_lag = pd.read_csv('combined_lagEDA.csv')
train_set = df_lag.iloc[:,0:48]
labels = df_lag.iloc[:,48:49]

clf = RandomForestClassifier(n_estimators=100,max_depth=15)
train, test, train_labels, test_labels = train_test_split(train_set, labels, test_size=0.33, random_state=30)

clf.fit(train, train_labels.values.ravel())

y_pred = clf.predict(test)

f1score   = f1_score        (test_labels, y_pred, average = 'macro')
recall    = recall_score    (test_labels, y_pred, average = 'macro')
precision = precision_score (test_labels, y_pred, average = 'macro')
accuracy  = accuracy_score  (test_labels, y_pred)

print('acc =', accuracy)
print('pre =', precision)
print('recall =', recall) 
print('f1 =', f1score)


# In[16]:


# df_total = df_total.dropna()
# scalar = MinMaxScaler()
# x_scaled = scalar.fit_transform(df_total.iloc[:,0:48])
# data = pd.DataFrame(x_scaled)
# data = data.fillna(0)

# pred_t = clf.predict(data)
# pred_t = pd.DataFrame(pred_t)
# pred_t.columns=['pred']
# pred_t.head(30)
# pred_t.to_csv('pred'+user+'.csv')
ff=pd.read_csv("Combined_lagEDA.csv")


# In[27]:


ff.columns


# In[28]:


ff.drop(['HRR_Mean', 'HRR_Min', 'HRR_Max',
       'HRR_Std', 'HRR_RMS', 'TEMPR_Mean', 'TEMPR_Min', 'TEMPR_Max',
       'TEMPR_Std'],axis=1)


# In[29]:


df_lag = ff
train_set = df_lag.iloc[:,0:48]
labels = df_lag.iloc[:,48:49]

clf = RandomForestClassifier(n_estimators=100,max_depth=15)
train, test, train_labels, test_labels = train_test_split(train_set, labels, test_size=0.33, random_state=30)

clf.fit(train, train_labels.values.ravel())

y_pred = clf.predict(test)

f1score   = f1_score        (test_labels, y_pred, average = 'macro')
recall    = recall_score    (test_labels, y_pred, average = 'macro')
precision = precision_score (test_labels, y_pred, average = 'macro')
accuracy  = accuracy_score  (test_labels, y_pred)

print('acc =', accuracy)
print('pre =', precision)
print('recall =', recall) 
print('f1 =', f1score)


# In[30]:


# taking HR,EDA an temp as parameters
# acc = 0.9539810080350621
# pre = 0.9525827322740762
# recall = 0.9351157842946799
# f1 = 0.9427923136823679

# leaving EDA as a parameter
# acc = 0.9517896274653032
# pre = 0.9526652625044415
# recall = 0.9311397507677217
# f1 = 0.9403267326497146


# considering only temp as a standard for measurement
# acc = 0.9500852203554906
# pre = 0.9494298034922313
# recall = 0.9299593653387411
# f1 = 0.9383754731454076


#considering only EDA as parameter
# acc = 0.9488677867056245
# pre = 0.9499459968062167
# recall = 0.9272025353343221
# f1 = 0.9368519990281442


# In[24]:


ff.drop(['HRR_Mean', 'HRR_Min', 'HRR_Max',
       'HRR_Std', 'HRR_RMS'],axis=1)


# In[31]:


df_lag = ff
train_set = df_lag.iloc[:,0:48]
labels = df_lag.iloc[:,48:49]

clf = RandomForestClassifier(n_estimators=100,max_depth=15)
train, test, train_labels, test_labels = train_test_split(train_set, labels, test_size=0.33, random_state=30)

clf.fit(train, train_labels.values.ravel())

y_pred = clf.predict(test)

f1score   = f1_score        (test_labels, y_pred, average = 'macro')
recall    = recall_score    (test_labels, y_pred, average = 'macro')
precision = precision_score (test_labels, y_pred, average = 'macro')
accuracy  = accuracy_score  (test_labels, y_pred)

print('acc =', accuracy)
print('pre =', precision)
print('recall =', recall) 
print('f1 =', f1score)


# In[ ]:





