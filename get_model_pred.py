#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:18:49 2021

@author: ananya
"""

import numpy as np
import pandas as pd
import pickle

def get_model_pred(X, true_ind):
  
    aggfeats = pd.DataFrame()
    
    windows = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

    #find statistics for entirety of time series data
    for signal in X.columns[1:13]:
        aggfeats['agg_mean_'+signal] = [np.mean(X[signal])]
        aggfeats['agg_var_'+signal] = [np.var(X[signal])]
        aggfeats['agg_med_'+signal] = [np.median(X[signal])]
        aggfeats['agg_range_'+signal] = [np.amax(X[signal]) - np.amin(X[signal])]
    
    #find statistics for 30-second data windows
    for wind in range(len(windows)-1):
        winddat = X.loc[(X.normTime >= windows[wind]) & (X.normTime<windows[wind+1])]
        for signal in winddat.columns[1:13]:
            aggfeats['mean'+str(wind+1)+'_'+signal] = [np.mean(X[signal])]
            aggfeats['var'+str(wind+1)+'_'+signal] = [np.var(X[signal])]
            aggfeats['med'+str(wind+1)+'_'+signal] = [np.median(X[signal])]
            aggfeats['range'+str(wind+1)+'_'+signal] = [np.amax(X[signal]) - np.amin(X[signal])]
        
    #train model on established best train data
    with open('model.pkl','rb') as f:
        rf = pickle.load(f)
    
    true_ind = np.asarray(true_ind, dtype=int)
    test_data = aggfeats[aggfeats.columns[true_ind]]
    #test_data = StandardScaler().fit_transform(test_data) #no longer applicable
    
    pred = rf.predict(test_data)
    
    if pred == 0:
        return "is healthy"
    elif pred == 1:
        return "has ALS"
    elif pred == 2:
        return "has Huntington's"
    elif pred == 3:
        return "has Parkinson's"
    else:
        return " "
