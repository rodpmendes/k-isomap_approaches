"""
Created on Thu Jun 15 05:03:53 2023

Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

def is_valid_mode(prediction_mode):
    return prediction_mode.upper() == 'DBSCAN' or prediction_mode.upper() == 'GMM'

def pred_gmm(data, target):
    num_clusters_gmm = len(np.unique(target))
    gmm = GaussianMixture(n_components=num_clusters_gmm, covariance_type='full')
    gmm_labels = gmm.fit_predict(data)
    
    return gmm_labels

def pred_dbscan(data, target):
    #HDBSCan (hierarquico - verificar implementação)
    
    #verify best params
    
    dbs = DBSCAN(eps=0.3, min_samples=4).fit(data)

    #n_clusters = len(set(self_labels)) - (1 if -1 in self_labels else 0)
    #n_noise = list(self_labels).count(-1)
    #print("clusters: %d" % n_clusters)
    #print("noise points: %d" % n_noise)

    return dbs.labels_


def predict(prediction_mode, data, target):
   
    if prediction_mode.upper() == 'GMM':
        self_labels = pred_gmm(data, target)
    elif prediction_mode.upper() == 'DBSCAN':
        self_labels = pred_dbscan(data, target)
        
    
    return self_labels