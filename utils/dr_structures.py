"""
Created on Thu Set 26 03:07:11 2023

Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""
from utils import data_structures as ds

def unsup_k_isomap():
    return {
              'name': 'Unsup K-ISOMAP',
              'learning': ds.learning.unsupervised,
              'transformation': ds.transformation.nonlinear,
              'structure': ds.structure.local_structure,
            
              'clf': []
           }

def sup_k_isomap():
    return {
              'name': 'Sup K-ISOMAP',
              'learning': ds.learning.supervised,
              'transformation': ds.transformation.nonlinear,
              'structure': ds.structure.local_structure,
            
              'clf': []
           }

def self_sup_k_isomap_gmm():
    return {
              'name': 'Self-Sup K-ISOMAP GMM',
              'learning': ds.learning.self_supervised,
              'transformation': ds.transformation.nonlinear,
              'structure': ds.structure.local_structure,
            
              'clf': []
           }

def self_sup_k_isomap_dbscan():
    return {
              'name': 'Self-Sup K-ISOMAP DBSCAN',
              'learning': ds.learning.self_supervised,
              'transformation': ds.transformation.nonlinear,
              'structure': ds.structure.local_structure,
            
              'clf': []
           }
    
def semi_sup_k_isomap_gmm():
    return {
              'name': 'Semi-Sup K-ISOMAP GMM',
              'learning': ds.learning.semi_supervised,
              'transformation': ds.transformation.nonlinear,
              'structure': ds.structure.local_structure,
            
              'clf': []
           }
    
def semi_sup_k_isomap_dbscan():
    return {
              'name': 'Semi-Sup K-ISOMAP DBSCAN',
              'learning': ds.learning.semi_supervised,
              'transformation': ds.transformation.nonlinear,
              'structure': ds.structure.local_structure,
            
              'clf': []
           }
    
def unsup_lda_comp_none():
    return {
              'name': 'LDAcompNone',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def sup_lda():
    return {
              'name': 'LDA',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def sup_pca():
    return {
              'name': 'SUP PCA',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def unsup_pca():
    return {
              'name': 'UNSUP PCA',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def unsup_t_sne():
    return {
              'name': 't-SNE',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def sup_nca():
    return {
              'name': 'NCA',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.local_structure,
            
              'clf': []
           }

def rfe_logist_reg():
    return {
              'name': 'RFE_LogistReg',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def rfe_linear_reg():
    return {
              'name': 'RFE_LinearReg',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def rfe_linear_svc():
    return {
              'name': 'RFE_LinearSVC',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }

def rfe_nonlinear_decision_trees():
    return {
              'name': 'RFE_NonLinearDecisionTrees',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.local_structure,
            
              'clf': []
           }
    
def rfe_nonlinear_random_forests():
    return {
              'name': 'RFE_NonLinearRandomForests',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def rfe_nonlinear_gradient_boosting_machines():
    return {
              'name': 'RFE_NonLinearGradientBoostingMachines',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def rfe_nonlinear_mpl():
    return {
              'name': 'RFE_NonLinearMLP',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def select_k_best():
    return {
              'name': 'SelectKBest',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }
    
def pls_regression():
    return {
              'name': 'PLS_Regression',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.linear,
              'structure': ds.structure.none.global_structure,
            
              'clf': []
           }

def deep_learning():
    return {
              'name': 'Deep_Learning',
              'learning': ds.learning.none.supervised,
              'transformation': ds.transformation.none.both_nonlinear_linear,
              'structure': ds.structure.none.local_structure,
            
              'clf': []
           }
    
def lle():
    return {
              'name': 'LLE',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.local_structure,
            
              'clf': []
           }
    
def umap():
    return {
              'name': 'UMAP',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def isomap():
    return {
              'name': 'ISOMAP',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def lap_eig():
    return {
              'name': 'Lap_Eig',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.local_structure,
            
              'clf': []
           }

def ltsa():
    return {
              'name': 'LTSA',
              'learning': ds.learning.none.unsupervised,
              'transformation': ds.transformation.none.nonlinear,
              'structure': ds.structure.none.both_local_global,
            
              'clf': []
           }
    
def get_dr_props(dr_name):
    
    switch = {
        'Unsup K-ISOMAP': unsup_k_isomap,
        'Sup K-ISOMAP': sup_k_isomap,
        'Self-Sup K-ISOMAP GMM': self_sup_k_isomap_gmm,
        'Self-Sup K-ISOMAP DBSCAN': self_sup_k_isomap_dbscan,
        'Semi-Sup K-ISOMAP GMM': semi_sup_k_isomap_gmm,
        'Semi-Sup K-ISOMAP DBSCAN': semi_sup_k_isomap_dbscan,
        
        'LDA': sup_lda,
        'SUP PCA': sup_pca,
        'UNSUP PCA': unsup_pca,
        'LDAcompNone': unsup_lda_comp_none,
        't-SNE': unsup_t_sne,
        'NCA' : sup_nca,
        'RFE_LogistReg': rfe_logist_reg,
        'RFE_LinearReg': rfe_linear_reg,
        'RFE_LinearSVC': rfe_linear_svc,
        'RFE_NonLinearDecisionTrees': rfe_nonlinear_decision_trees,
        'RFE_NonLinearRandomForests': rfe_nonlinear_random_forests,
        'RFE_NonLinearGradientBoostingMachines': rfe_nonlinear_gradient_boosting_machines,
        'RFE_NonLinearMLP': rfe_nonlinear_mpl,
        'SelectKBest': select_k_best,
        'PLS_Regression': pls_regression,
        'DeepLearning': deep_learning,
        'LLE': lle,
        'UMAP': umap,
        'ISOMAP': isomap,
        'LapEig': lap_eig,
        'LTSA': ltsa
        
    }

    return switch.get(dr_name, lambda: print("Invalid Dimensionality Reduction Technique"))()

