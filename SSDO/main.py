# -*- coding: utf-8 -*-

import click
from scipy.io import loadmat
from ssdo import SSDO
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest

@click.command()
@click.argument('dataset_name', type=click.Choice(['description_data','heatmap','description_plus_timeseries_data','line_chart_all_data']))
@click.argument('net_name', type=click.Choice(['IsolationForest','SSDO_with_iforest','SSDO_with_COP-Kmeans']))
@click.option('--n_labeled_normal', type=int, default=500 ,help='labeled normal number from 0 to 3510')
@click.option('--n_labeled_anomaly', type=int, default=100 ,help='labeled normal number from 0 to 870')
@click.option('--n_unlabeled', type=int, default=9660 ,help='labeled normal number from 0 to 9660')

def main(dataset_name,net_name,n_labeled_normal,n_labeled_anomaly,n_unlabeled):
    
    if dataset_name == 'description_data':
        load_fn = 'Last_use_data/description_data.mat'
            
    if dataset_name == 'heatmap':
        load_fn = 'Last_use_data/heatmap.mat'
        
    if dataset_name == 'description_plus_timeseries_data':
        load_fn = 'Last_use_data/description_plus_timeseries_data.mat'
     
    if dataset_name == 'line_chart_all_data':
        load_fn = 'Last_use_data/line_chart_all_data.mat'
        
    load_data = loadmat(load_fn)
    X = load_data['X'] 
    y = load_data['y'].ravel()

    idx_norm = y == 0
    idx_out = y == 1
    idx_un = y == 2 
    
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],test_size=0.4,random_state=8)
    X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],test_size=0.4,random_state=8)
    
    X_train = np.concatenate((X_train_norm[:n_labeled_normal], X_train_out[:n_labeled_anomaly],X[idx_un][:n_unlabeled]))
    X_test = np.concatenate((X_test_norm, X_test_out))
    y_train = np.concatenate((y_train_norm[:n_labeled_normal], y_train_out[:n_labeled_anomaly],y[idx_un][:n_unlabeled]))
    y_test = np.concatenate((y_test_norm, y_test_out))
    
    #-------------------------------------
    y_train_one = y_train.copy()
    y_train_one[y_train_one <= 0] = -1
    y_train_one[y_train_one >= 2] = 0
    #-------------------------------------
    
    if net_name == 'IsolationForest':
        prior_detector = IsolationForest(n_estimators=100, contamination=0.2, behaviour='new')
        prior_detector.fit(X_train) 
        tr_prior = prior_detector.decision_function(X_train) * -1
        te_pred  = prior_detector.decision_function(X_test) * -1
        tr_prior = tr_prior + abs(min(tr_prior))
        te_pred  = te_pred  + abs(min(te_pred)) 
        
    if net_name == 'SSDO_with_iforest': 
        prior_detector = IsolationForest(n_estimators=100, contamination=0.2, behaviour='new')
        prior_detector.fit(X_train) 
        tr_prior = prior_detector.decision_function(X_train) * -1
        te_prior = prior_detector.decision_function(X_test) * -1
        tr_prior = tr_prior + abs(min(tr_prior))
        te_prior = te_prior + abs(min(te_prior))
        
        detector = SSDO(unsupervised_prior='other')
        tr_pred = detector.fit_predict(X_train, y_train_one,prior=tr_prior)
        te_pred = detector.predict(X_test,prior=te_prior)
        y_test[y_test <= 0] = -1
     
        
    if net_name == 'SSDO_with_COP-Kmeans': 
        detector = SSDO()
        tr_pred = detector.fit_predict(X_train, y_train_one)
        te_pred = detector.predict(X_test)
        y_test[y_test <= 0] = -1
    
    print('dataset name:',dataset_name,'&','net name:',net_name,', AUC:', accuracy_score(y_test, te_pred))
   
if __name__ == '__main__':
    main()
    
    
   
    

