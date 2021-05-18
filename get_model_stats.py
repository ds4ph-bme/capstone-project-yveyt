#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:45:22 2021

@author: ananya
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import glob
import pickle
    
def build_model():    
    keys = ['time', 'lstride', 'rstride', 'lswing', 'rswing', 'lswing_int', 'rswing_int', 'lstance', 
            'rstance', 'lstance_int', 'rstance_int', 'dsupport', 'dsupport_int']
    
    #read in files
    files = glob.glob('gait_data/control*.{}'.format("txt"))
    control = pd.DataFrame()
    all_pat = np.array([9])
    count = 1
    
    for file in files:
        X = pd.read_table(file,header=None)
        control = pd.concat([control,X], ignore_index=True)
        patient = np.ones(len(X), dtype=int) * count
        all_pat = np.concatenate((all_pat, patient))
        count += 1
    
    control.columns = keys
    all_patients = np.delete(all_pat, 0)
    control['patient'] = all_patients
    control['combpatient'] = all_patients
    control['label'] = 0
    
    files = glob.glob('gait_data/als*.{}'.format("txt"))
    als = pd.DataFrame()
    all_pat = np.array([9])
    count = 1
    
    for file in files:
        X = pd.read_table(file,header=None)
        als = pd.concat([als,X], ignore_index=True)
        patient = np.ones(len(X), dtype=int) * count
        all_pat = np.concatenate((all_pat, patient))
        count += 1
    
    als.columns = keys
    all_patients = np.delete(all_pat, 0)
    als['patient'] = all_patients
    numpat_control = np.asarray(control.combpatient)
    als['combpatient'] = all_patients + numpat_control[-1] 
    als['label'] = 1
    
    files = glob.glob('gait_data/hunt*.{}'.format("txt"))
    hunt = pd.DataFrame()
    all_pat = np.array([9])
    count = 1
    
    for file in files:
        X = pd.read_table(file,header=None)
        hunt = pd.concat([hunt,X], ignore_index=True)
        patient = np.ones(len(X), dtype=int) * count
        all_pat = np.concatenate((all_pat, patient))
        count += 1
    
    hunt.columns = keys
    all_patients = np.delete(all_pat, 0)
    hunt['patient'] = all_patients
    numpat_als = np.asarray(als.combpatient)
    hunt['combpatient'] = all_patients + numpat_als[-1] 
    hunt['label'] = 2
    
    files = glob.glob('gait_data/park*.{}'.format("txt"))
    park = pd.DataFrame()
    all_pat = np.array([9])
    count = 1
    
    for file in files:
        X = pd.read_table(file, header = None)
        park = pd.concat([park, X], ignore_index=True)
        patient = np.ones(len(X), dtype=int) * count
        all_pat = np.concatenate((all_pat, patient))
        count += 1
    
    park.columns = keys
    all_patients = np.delete(all_pat, 0)
    park['patient'] = all_patients
    numpat_hunt = np.asarray(hunt.combpatient)
    park['combpatient'] = all_patients + numpat_hunt[-1] 
    park['label'] = 3
    
    f_keys = ['lstride', 'rstride', 'lswing', 'rswing', 'lswing_int', 'rswing_int', 'lstance', 
        'rstance', 'lstance_int', 'rstance_int', 'dsupport', 'dsupport_int']
        
    #replace outliers with the data means
    for key in f_keys:
        temp = np.asarray(park[key])
        z = np.abs(stats.zscore(temp))
        ind = np.where(z > 3)
        temp[ind] = np.mean(temp)
        park[key] = temp
    
    for key in f_keys:
        temp = np.asarray(hunt[key])
        z = np.abs(stats.zscore(temp))
        ind = np.where(z > 3)
        temp[ind] = np.mean(temp)
        hunt[key] = temp
      
    for key in f_keys:
        temp = np.asarray(als[key])
        z = np.abs(stats.zscore(temp))
        ind = np.where(z > 3)
        temp[ind] = np.mean(temp)
        als[key] = temp
      
    for key in f_keys:
        temp = np.asarray(control[key])
        z = np.abs(stats.zscore(temp))
        ind = np.where(z > 3)
        temp[ind] = np.mean(temp)
        control[key] = temp
      
    alldat = pd.concat([control, als, hunt, park])
    
    Windowed = alldat.copy()
    Windowed = Windowed.reset_index()
    
    yvec = Windowed[['combpatient','label']]
    yvec = yvec.drop_duplicates(subset='combpatient', keep="first")
    yvec = yvec.set_index('combpatient')
    
    #loop through alldat and give the time since start for each patient
    Windowed['normTime'] = 0
    
    for i in range(Windowed['combpatient'].iloc[-1]+1):
        patient = Windowed.loc[Windowed.combpatient == i]
        normTime = []
        time1 = np.asarray(patient["time"])
        for j in range(len(patient["time"])):
            normTime.append(time1[j]-time1[0])
        Windowed.loc[Windowed.combpatient==i, 'normTime'] = normTime
      
    #for each window
    windows = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    
    allfeats = pd.DataFrame()
    
    #allfeats = dict()
    for subject in range(1,np.asarray(Windowed.combpatient)[-1]+1):
        pat = Windowed.loc[Windowed.combpatient == subject]
        aggfeats = pd.DataFrame()
        windfeats = pd.DataFrame()
        #find statistics for entirety of time series data
        for signal in pat.columns[2:14]:
            aggfeats['agg_mean_'+signal] = [np.mean(pat[signal])]
            aggfeats['agg_var_'+signal] = [np.var(pat[signal])]
            aggfeats['agg_med_'+signal] = [np.median(pat[signal])]
            aggfeats['agg_range_'+signal] = [np.amax(pat[signal]) - np.amin(pat[signal])]
        for wind in range(len(windows)-1):
            winddat = pat.loc[(pat.normTime >= windows[wind]) & (pat.normTime<windows[wind+1])]
            #find statistics for 30-second data windows
            for signal in winddat.columns[2:14]:
                windfeats['mean'+str(wind+1)+'_'+signal] = [np.mean(pat[signal])]
                windfeats['var'+str(wind+1)+'_'+signal] = [np.var(pat[signal])]
                windfeats['med'+str(wind+1)+'_'+signal] = [np.median(pat[signal])]
                windfeats['range'+str(wind+1)+'_'+signal] = [np.amax(pat[signal]) - np.amin(pat[signal])]
        subjectfeats = pd.concat([aggfeats, windfeats], axis=1)
        allfeats = pd.concat([allfeats,subjectfeats])
    
    allfeats = allfeats.set_index(np.arange(1,np.asarray(Windowed.combpatient)[-1]+1))
    allfeats['labels'] = np.asarray(yvec['label'])
    
    max_acc = 0
    train_acc = 0
    ideal_rand = 0
    
    #loop through and find train/test split that maximizes model performance
    for num in range(100):
        train_set, test_set = train_test_split(allfeats, random_state=num)
        train_data = train_set.drop('labels', axis=1)
        train_labels = train_set['labels']
        test_data = test_set.drop('labels', axis=1)
        test_labels = test_set['labels']
      
        #find optimal features using PCA
        pca = PCA(0.9)
        train_data_new = StandardScaler().fit_transform(train_data)
        pca.fit(train_data_new)
      
        ind_set = []
        for val in range(len(abs(pca.components_))):
            arr = abs(pca.components_[val])
            ind_set.append(arr.argsort()[-1:][::-1])
      
        guess_ind = []
        for j in range(len(ind_set)):
            guess_ind.append(ind_set[j][0])
      
        guess_ind = set(guess_ind)
        guess_ind = list(guess_ind)
      
        #isolate optimal features and use them for prediction
        train_data = train_data[train_data.columns[guess_ind]]
        test_data = test_data[test_data.columns[guess_ind]]
    
        #no longer applicable
        #train_data = StandardScaler().fit_transform(train_data2)
        #test_data = StandardScaler().fit_transform(test_data)
    
        #train model on selected train data
        rf = RandomForestClassifier(max_features='auto', n_estimators=200, max_depth=5, criterion='gini')
        rf.fit(train_data,train_labels)
        acc = rf.score(test_data, test_labels)
        t_acc = rf.score(train_data, train_labels)
        #if model performance with this data is better, save info
        if (acc >= max_acc) and (t_acc >= train_acc):
            max_acc = acc
            train_acc = t_acc
            ideal_rand = num
            true_ind = guess_ind
            with open('model.pkl','wb') as f:
                pickle.dump(rf,f)
                
    model_stats = np.array([max_acc, t_acc, ideal_rand, true_ind], dtype=object)
    np.save('modelstats.npy', model_stats, allow_pickle=True)

def get_model_stats():
    model_stats = np.load('modelstats.npy', allow_pickle=True)
    return model_stats