"""
Created on Thu Jun 15 05:03:53 2023

Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import pandas as pd
from enum import Enum
import sklearn.datasets as skdata

from utils import dr_structures as dr_struct

class learning(Enum):
    none = 0
    supervised = 1
    self_supervised = 2
    semi_supervised = 3
    unsupervised = 4

class transformation(Enum):
    none = 0
    nonlinear = 1
    linear = 2
    both_nonlinear_linear = 3
    
class structure(Enum):
    none = 0
    local_structure = 1
    global_structure = 2
    both_local_global = 3
       
class classifiers(Enum):
    none = 0
    knn = 1,
    svm = 2,
    naive_bayes = 3
    decision_tree = 4
    quadratic_discriminant = 5
    mpl_classifier = 6
    gaussian_process = 7
    random_forest_classifier = 8
    
def export_to_csv(table_results, filename):
    flatten_data_results = {
    'name': [],
    'classes': [],
    'samples_per_class': [],
    'samples_total': [],
    'dimensionality': [],
    
    'dr_name': [],
    'dr_learning': [],
    'dr_transformation': [],
    'dr_structure': [],
    
    'clf_name': [],
    'clf_accuracy': [],
    'clf_kappa': [],
    'clf_balanced_accuracy': []
    }
    
    for result in table_results:
        for dr in result['db']['dr']:
            for clf in dr['clf']:
                flatten_data_results['name'].append(result['db']['name'])
                flatten_data_results['classes'].append(result['db']['classes'])
                flatten_data_results['samples_per_class'].append(result['db']['samples_per_class'])
                flatten_data_results['samples_total'].append(result['db']['samples_total'])
                flatten_data_results['dimensionality'].append(result['db']['dimensionality'])
            
                flatten_data_results['dr_name'].append(dr['name'])
                flatten_data_results['dr_learning'].append(dr['learning'])
                flatten_data_results['dr_transformation'].append(dr['transformation'])
                flatten_data_results['dr_structure'].append(dr['structure'])
                
                flatten_data_results['clf_name'].append(clf['name'])
                flatten_data_results['clf_accuracy'].append(clf['accuracy'])
                flatten_data_results['clf_kappa'].append(clf['kappa'])
                flatten_data_results['clf_balanced_accuracy'].append(clf['balanced_accuracy'])
    
    df = pd.DataFrame(flatten_data_results)
    df.index += 1
    df.to_csv(filename, index=True)
    
def usage_example():
    table_results = []

    table_results = [{
        'db': {
                'name': 'iris',
                'classes': 3,
                'samples_per_class': 50,
                'samples_total': 150,
                'dimensionality': 4,
                
                'dr': 
                    [{
                        'name': 'Sup K-ISOMAP',
                        'learning': learning.supervised.name,
                        'transformation': transformation.nonlinear.name,
                        'structure': structure.local_structure.name,
                        
                        'clf': [{
                                    'name': 'KNN',
                                    'accuracy': 0.986,
                                    'kappa': 0.975,
                                    'balanced_accuracy': 0.9
                                },
                                {
                                    'name': 'SVM',
                                    'accuracy': 0.869,
                                    'kappa': 0.755,
                                    'balanced_accuracy': 0.679
                                },
                                {
                                    'name': 'Naive Bayes',
                                    'accuracy': 0.679,
                                    'kappa': 0.985,
                                    'balanced_accuracy': 0.799
                                }]
                    },
                    {
                        'name': 'LDA comp none',
                        'learning': learning.supervised.name,
                        'transformation': transformation.linear.name,
                        'structure': structure.global_structure.name,
                        
                        'clf': [{
                                    'name': 'KNN',
                                    'accuracy': 0.965,
                                    'kappa': 0.953,
                                    'balanced_accuracy': 0.899
                                },
                                {
                                    'name': 'SVM',
                                    'accuracy': 0.768,
                                    'kappa': 0.858,
                                    'balanced_accuracy': 0.799
                                },
                                {
                                    'name': 'Naive Bayes',
                                    'accuracy': 0.579,
                                    'kappa': 0.985,
                                    'balanced_accuracy': 0.999
                                }]
                    }]
            },
    },
    {
        'db': {
                'name': 'digits',
                'classes': 10,
                'samples_per_class': 180,
                'samples_total': 1797,
                'dimensionality': 64,
                'dr': 
                    [{
                        'name': 'Sup K-ISOMAP',
                        'learning': learning.supervised.name,
                        'transformation': transformation.nonlinear.name,
                        'structure': structure.local_structure.name,
                        
                        'clf': [{
                                    'name': 'KNN',
                                    'accuracy': 0.977,
                                    'kappa': 0.971,
                                    'balanced_accuracy': 0.7
                                },
                                {
                                    'name': 'SVM',
                                    'accuracy': 0.769,
                                    'kappa': 0.796,
                                    'balanced_accuracy': 0.789
                                },
                                {
                                    'name': 'Naive Bayes',
                                    'accuracy': 0.799,
                                    'kappa': 0.959,
                                    'balanced_accuracy': 0.919
                                }]
                    },
                    {
                        'name': 'LDA comp none',
                        'learning': learning.supervised.name,
                        'transformation': transformation.linear.name,
                        'structure': structure.global_structure.name,
                        
                        'clf': [{
                                    'name': 'KNN',
                                    'accuracy': 0.995,
                                    'kappa': 0.965,
                                    'balanced_accuracy': 0.999
                                },
                                {
                                    'name': 'SVM',
                                    'accuracy': 0.688,
                                    'kappa': 0.918,
                                    'balanced_accuracy': 0.899
                                },
                                {
                                    'name': 'Naive Bayes',
                                    'accuracy': 0.991,
                                    'kappa': 0.915,
                                    'balanced_accuracy': 0.904
                                }]
                    }]
            },          
    }]
        
    export_to_csv(table_results, 'results.csv')
    
def get_data_structure():
    table_results = [{
        'db': {
                'name': None,
                'classes': None,
                'samples_per_class': None,
                'samples_total': None,
                'dimensionality': None,
                
                'dr': None
            },
    }]
    
    table_results['db']['dr'] = get_dr_structure()
    
    return table_results
    
def get_dr_structure():
    dr_structure = [{
                        'name': None,
                        'learning': learning.none.name,
                        'transformation': transformation.none.name,
                        'structure': structure.none.name,
                        
                        'clf': None
                    }]
    dr_structure['clf'] = get_clf_structure()
    
    return dr_structure


def get_clf_structure():
    return [{
                'name': None,
                'accuracy': None,
                'kappa': None,
                'balanced_accuracy': None
            }]
        

def iris():
    db_name = 'iris'
    db = skdata.load_iris()
    db_props = {
                'name': db_name,
                'classes': len(db['target_names']),
                'samples_per_class': { 
                                            db['target_names'][0] : len(db['data'][db['target']==0]),
                                            db['target_names'][1] : len(db['data'][db['target']==1]), 
                                            db['target_names'][2] : len(db['data'][db['target']==2]) 
                                        },
                'samples_total': len(db['data']),
                'dimensionality': len(db['feature_names'])
                }
    return (db_name, db, db_props)

def digits():
    db_name = 'digits'
    db = skdata.load_digits()
    db_props = {
                  'name': db_name,
                  'classes': len(db['target_names']),
                  'samples_per_class': { 
                                            db['target_names'][0] : len(db['data'][db['target']==0]),
                                            db['target_names'][1] : len(db['data'][db['target']==1]), 
                                            db['target_names'][2] : len(db['data'][db['target']==2]),
                                            db['target_names'][3] : len(db['data'][db['target']==3]),
                                            db['target_names'][4] : len(db['data'][db['target']==4]),
                                            db['target_names'][5] : len(db['data'][db['target']==5]),
                                            db['target_names'][6] : len(db['data'][db['target']==6]),
                                            db['target_names'][7] : len(db['data'][db['target']==7]),
                                            db['target_names'][8] : len(db['data'][db['target']==8]),
                                            db['target_names'][9] : len(db['data'][db['target']==9]),
                                          },
                  'samples_total': len(db['data']),
                  'dimensionality': len(db['feature_names'])
               }
    return (db_name, db, db_props)
    
def prnn_crabs():
    db_name = 'prnn_crabs'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)
    
def balance_scale():
    # ??????
    # https://www.openml.org/d/11
    # number of features = 4 (site 5 ??)
    # number of numeric features = 4
    # number of symbolic features = 1 (??)
    
    db_name = 'balance-scale'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
                }
    return (db_name, db, db_props)
    
def parity5():
    # ??????
    # https://www.openml.org/d/40714
    # number of features = 5 (site 6 ??)
    # number of numeric features = 0
    # number of symbolic features = 5 (6 ??)
    
    db_name = 'parity5'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
                }
    return (db_name, db, db_props)

def hayes_roth():
    # ??????
    # https://www.openml.org/d/974
    # number of features = 4 (site 5 ??)
    # number of numeric features = 4
    # number of symbolic features = 1 (??)
    
    db_name = 'hayes-roth'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)
 
def rabe_131():
    # ??????
    # https://www.openml.org/d/874
    # number of features = 5 (site 6 ??)
    # number of numeric features = 5
    # number of symbolic features = 1 (??)
    
    db_name = 'rabe_131'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)
    
def servo():
    # ??????
    # https://www.openml.org/d/747
    # number of features = 4 (site 5 ??)
    # number of numeric features = 4 (5 ??)
    # number of symbolic features = 0
    
    db_name = 'servo'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def monks_problems_1():
    # ??????
    # https://www.openml.org/d/333
    # number of features = 6 (site 7 ??)
    # number of numeric features = 0
    # number of symbolic features = 6 (7 ??)
    
    db_name = 'monks-problems-1'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def bolts():
    # ??????
    # https://www.openml.org/d/857
    # number of features = 7 (site 8 ??)
    # number of numeric features = 7
    # number of symbolic features = 0 (1 ??)

    db_name = 'bolts'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def fri_c2_100_10():
    # ??????
    # https://www.openml.org/d/762
    # number of features = 10 (site 11 ??)
    # number of numeric features = 10
    # number of symbolic features = 1 (1 ??)

    db_name = 'fri_c2_100_10'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def threeOf9():
    # ??????
    # https://www.openml.org/d/40690
    # number of features = 9 (site 10 ??)
    # number of numeric features = 0
    # number of symbolic features = 9 (10 ??)
    
    db_name = 'threeOf9'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def fri_c3_100_5():
    # ??????
    # https://www.openml.org/d/916
    # number of features = 5 (site 6 ??)
    # number of numeric features = 5
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'fri_c3_100_5'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def baskball():
    # ??????
    # https://www.openml.org/d/731
    # number of features = 4 (site 5 ??)
    # number of numeric features = 4
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'baskball'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def newton_hema():
    # ??????
    # https://www.openml.org/d/784
    # number of features = 3 (site 4 ??)
    # number of numeric features = 3 (2 ??)
    # number of symbolic features = 0 (2 ??)
    
    db_name = 'newton_hema'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def strikes():
    # ??????
    # https://www.openml.org/d/770
    # number of features = 6 (site 7 ??)
    # number of numeric features = 6 
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'strikes'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def datatrieve():
    # ??????
    # https://www.openml.org/d/1075
    # number of features = 8 (site 9 ??)
    # number of numeric features = 8 
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'datatrieve'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def diggle_table_a2():
    # ??????
    # https://www.openml.org/d/818
    # number of features = 8 (site 9 ??)
    # number of numeric features = 8 (7 ??)
    # number of symbolic features = 0 (2 ??)
    
    db_name = 'diggle_table_a2'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def fl2000():
    # ??????
    # https://www.openml.org/d/988
    # number of features = 15 (site 16 ??)
    # number of numeric features = 15 (14 ??)
    # number of symbolic features = 0 (2 ??)
    
    db_name = 'fl2000'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def triazines():
    # ??????
    # https://www.openml.org/d/788
    # number of features = 60 (site 61 ??)
    # number of numeric features = 60
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'triazines'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def veteran():
    # ??????
    # https://www.openml.org/d/719
    # number of features = 7 (site 8 ??)
    # number of numeric features = 7 (3 ??)
    # number of symbolic features = 0 (5 ??)
    
    db_name = 'veteran'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def diabetes():
    # ??????
    # https://www.openml.org/d/40975
    # number of features = 6 (site 7 ??)
    # number of numeric features = 7 (0 ??)
    # number of symbolic features = 0 (7 ??)
    
    db_name = 'diabetes'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def car():
    # ??????
    # https://www.openml.org/d/996
    # number of features = 9 (site 10 ??)
    # number of numeric features = 9
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'car'
    db = skdata.fetch_openml(name=db_name, version=3)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def prnn_fglass():
    # ??????
    # https://www.openml.org/d/721
    # number of features = 10 (site 11 ??)
    # number of numeric features = 10
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'prnn_fglass'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def analcatdata_creditscore():
    # ??????
    # https://www.openml.org/d/461
    # number of features = 6 (site 7 ??)
    # number of numeric features = 6 (3 ??)
    # number of symbolic features = 0 (4 ??)
    
    db_name = 'analcatdata_creditscore'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def pwLinear():
    # ??????
    # https://www.openml.org/d/721
    # number of features = 10 (site 11 ??)
    # number of numeric features = 10
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'pwLinear'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def breast_cancer():
    # ??????
    # https://www.openml.org/d/13 (???)
    # number of features = 30
    # number of numeric features = 30
    # number of symbolic features = 0
    
    db_name = 'breast_cancer'
    db = skdata.load_breast_cancer()
    db_props = {
                 'name': db_name,
                 'classes': len(db['target_names']),
                 'samples_per_class': { 
                                            db['target_names'][0] : len(db['data'][db['target']==0]),
                                            db['target_names'][1] : len(db['data'][db['target']==1])
                                         },
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['feature_names'])
               }
    return (db_name, db, db_props)

def wine():
    # ??????
    # https://www.openml.org/d/42867
    # number of features = 13 (16 ??)
    # number of numeric features = 13
    # number of symbolic features = 0 (3 ??)
    
    db_name = 'wine'
    db = skdata.load_wine()
    db_props = {
                 'name': db_name,
                 'classes': len(db['target_names']),
                 'samples_per_class': { 
                                            db['target_names'][0] : len(db['data'][db['target']==0]),
                                            db['target_names'][1] : len(db['data'][db['target']==1])
                                         },
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['feature_names'])
               }
    return (db_name, db, db_props)

def backache():
    # ??????
    # https://www.openml.org/d/463
    # number of features = 31 (32 ??)
    # number of numeric features = 31 (5 ??)
    # number of symbolic features = 0 (27 ??)
    
    db_name = 'backache'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def heart_statlog():
    # ??????
    # https://www.openml.org/d/53
    # number of features = 13 (14 ??)
    # number of numeric features = 13
    # number of symbolic features = 0 (1 ??)
    
    db_name = 'heart-statlog'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

#
# add data bases
#
def tic_tac_toe():
    # ??????
    # https://www.openml.org/d/50
    # number of features = 9 (?? 10)
    # number of numeric features = 9 (?? 0)
    # number of symbolic features = 0 (?? 10)
    
    db_name = 'tic-tac-toe'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)


def visualizing_galaxy():
    # ??????
    # https://www.openml.org/d/925
    # number of features = 4 (?? 5)
    # number of numeric features = 4
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'visualizing_galaxy'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def sleuth_ex1605():
    # ??????
    # https://www.openml.org/d/755
    # number of features = 5 (?? 6)
    # number of numeric features = 5
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'sleuth_ex1605'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def mux6():
    # ??????
    # https://www.openml.org/d/40681
    # number of features = 6 (?? 7)
    # number of numeric features = 6 (?? 0)
    # number of symbolic features = 0 (?? 7)
    
    db_name = 'mux6'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def car_evaluation():
    # ??????
    # https://www.openml.org/d/40664
    # number of features = 21 (?? 22)
    # number of numeric features = 21 (?? 0)
    # number of symbolic features = 0 (?? 22)
    
    db_name = 'car-evaluation'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def blogger():
    # ??????
    # https://www.openml.org/d/1463
    # number of features = 5 (?? 6)
    # number of numeric features = 5 (?? 0)
    # number of symbolic features = 0 (?? 6)
    
    db_name = 'blogger'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def sa_heart():
    # ??????
    # https://www.openml.org/d/1498
    # number of features = 9 (?? 10)
    # number of numeric features = 9 (?? 8)
    # number of symbolic features = 0 (?? 2)
    
    db_name = 'sa-heart'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def pyrim():
    # ??????
    # https://www.openml.org/d/800
    # number of features = 27 (?? 28)
    # number of numeric features = 27
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'pyrim'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def spectf():
    # ??????
    # https://www.openml.org/d/337
    # number of features = 44 (?? 45)
    # number of numeric features = 44
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'SPECTF'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def kidney():
    # ??????
    # https://www.openml.org/d/945
    # number of features = 6 (?? 7)
    # number of numeric features = 6 (?? 3)
    # number of symbolic features = 0 (?? 4)
    
    db_name = 'kidney'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def diabetes_numeric():
    # ??????
    # https://www.openml.org/d/212
    # number of features = 2 (?? 3)
    # number of numeric features = 2 (?? 3)
    # number of symbolic features = 0
    
    db_name = 'diabetes_numeric'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def parkinsons():
    # ??????
    # https://www.openml.org/d/1488
    # number of features = 22 (?? 23)
    # number of numeric features = 22
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'parkinsons'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def rabe_131():
    # ??????
    # https://www.openml.org/d/874
    # number of features = 5 (?? 6)
    # number of numeric features = 5
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'rabe_131'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def corral():
    # ??????
    # https://www.openml.org/d/40669
    # number of features = 6 (?? 7)
    # number of numeric features = 6 (?? 0)
    # number of symbolic features = 0 (?? 7)
    
    db_name = 'corral'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def grub_damage():
    # ??????
    # https://www.openml.org/d/1026
    # number of features = 8 (?? 9)
    # number of numeric features = 8 (?? 2)
    # number of symbolic features = 0 (?? 7)
    
    db_name = 'grub-damage'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def haberman():
    # ??????
    # https://www.openml.org/d/43
    # number of features = 3 (?? 4)
    # number of numeric features = 3 (?? 2)
    # number of symbolic features = 0 (?? 2)
    
    db_name = 'haberman'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def tuning_svms():
    # ??????
    # https://www.openml.org/d/41976
    # number of features = 80 (?? 81)
    # number of numeric features = 80
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'TuningSVMs'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def prnn_synth():
    # ??????
    # https://www.openml.org/d/464
    # number of features = 2 (?? 3)
    # number of numeric features = 2
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'prnn_synth'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def visualizing_environmental():
    # ??????
    # https://www.openml.org/d/736
    # number of features = 3 (?? 4)
    # number of numeric features = 3
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'visualizing_environmental'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def mu284():
    # ??????
    # https://www.openml.org/d/880
    # number of features = 10 (?? 11)
    # number of numeric features = 10
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'mu284'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def ar4():
    # ??????
    # https://www.openml.org/d/1061
    # number of features = 29 (?? 30)
    # number of numeric features = 29
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'ar4'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def engine1():
    # ??????
    # https://www.openml.org/d/4340
    # number of features = 5 (?? 6)
    # number of numeric features = 5
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'Engine1'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def prnn_viruses():
    # ??????
    # https://www.openml.org/d/480
    # number of features = 18 (?? 19)
    # number of numeric features = 18 (?? 10)
    # number of symbolic features = 0 (?? 9)
    
    db_name = 'prnn_viruses'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def vineyard():
    # ??????
    # https://www.openml.org/d/713
    # number of features = 2 (?? 3)
    # number of numeric features = 2
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'vineyard'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def confidence():
    # ??????
    # https://www.openml.org/d/1015
    # number of features = 3 (?? 4)
    # number of numeric features = 3
    # number of symbolic features = 0 (?? 1)
    
    db_name = 'confidence'
    db = skdata.fetch_openml(name=db_name, version=2)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def olivetti_faces():
    # ??????
    # https://www.openml.org/d/41083
    # number of features = 4097
    # number of numeric features = 4096
    # number of symbolic features = 1
    
    db_name = 'Olivetti_Faces'
    db = skdata.fetch_openml(name=db_name, version=1)
    db_props = {
                 'name': db_name,
                 'classes': len(db['target'].cat.categories),
                 'samples_per_class': db['target'].value_counts().to_dict(),
                 'samples_total': len(db['data']),
                 'dimensionality': len(db['data'].columns)
               }
    return (db_name, db, db_props)

def get_db_props(db_name):
    
    switch = {
        'tic-tac-toe':tic_tac_toe,
        'visualizing_galaxy': visualizing_galaxy,
        'sleuth_ex1605': sleuth_ex1605,
        'mux6': mux6,
        'car-evaluation': car_evaluation,
        'blogger': blogger,
        'sa-heart': sa_heart,
        'pyrim': pyrim,
        'SPECTF': spectf,
        'kidney': kidney,
        'diabetes_numeric': diabetes_numeric,
        'parkinsons': parkinsons,
        'rabe_131': rabe_131,
        'corral': corral,
        'grub-damage': grub_damage,
        'haberman': haberman,
        'TuningSVMs': tuning_svms,
        'prnn_synth': prnn_synth,
        'visualizing_environmental': visualizing_environmental,
        'mu284': mu284,
        'ar4': ar4,
        'Engine1': engine1,
        'prnn_viruses': prnn_viruses,
        'vineyard': vineyard,
        'confidence': confidence,
        
        
        'iris': iris,
        'digits': digits,
        'prnn_crabs': prnn_crabs,
        'balance-scale': balance_scale,
        'parity5': parity5,
        'hayes-roth': hayes_roth,
        'rabe_131': rabe_131,
        'servo': servo,
        'monks-problems-1': monks_problems_1,
        'bolts': bolts, 
        'fri_c2_100_10': fri_c2_100_10,
        'threeOf9': threeOf9,
        'fri_c3_100_5': fri_c3_100_5,
        'baskball': baskball,
        'newton_hema': newton_hema,
        'strikes': strikes,
        'datatrieve': datatrieve,
        'diggle_table_a2': diggle_table_a2,
        'fl2000': fl2000,
        'triazines': triazines,
        'veteran': veteran,
        'diabetes': diabetes,
        'car': car,
        'prnn_fglass': prnn_fglass,
        'analcatdata_creditscore': analcatdata_creditscore,
        'pwLinear': pwLinear,
        'breast_cancer': breast_cancer,
        'wine': wine,
        'backache': backache,
        'heart-statlog': heart_statlog,
        
        'Olivetti_Faces': olivetti_faces
    }

    return switch.get(db_name, lambda: print("Invalid data base"))()

def get_data_list():
    data_list = []
    
    data_list.append( get_db_props('wine') )
    
    data_list.append( get_db_props('pwLinear') )
    data_list.append( get_db_props('breast_cancer') )
    
    data_list.append( get_db_props('backache') )
    data_list.append( get_db_props('heart-statlog') )
    
    
    
    
    
    data_list.append( get_db_props('Olivetti_Faces') ) 
    data_list.append( get_db_props('digits') )
    
    
    
    data_list.append( get_db_props('tic-tac-toe') )
    data_list.append( get_db_props('visualizing_galaxy') )
    data_list.append( get_db_props('sleuth_ex1605') )
    data_list.append( get_db_props('car-evaluation') )
    data_list.append( get_db_props('blogger') )
    data_list.append( get_db_props('sa-heart') )
    data_list.append( get_db_props('pyrim') )
    data_list.append( get_db_props('SPECTF') )
    data_list.append( get_db_props('kidney') )
    data_list.append( get_db_props('diabetes_numeric') )
    data_list.append( get_db_props('parkinsons') )
    data_list.append( get_db_props('rabe_131') )
    data_list.append( get_db_props('corral') )
    data_list.append( get_db_props('grub-damage') )
    data_list.append( get_db_props('haberman') )
    data_list.append( get_db_props('TuningSVMs') )
    data_list.append( get_db_props('prnn_synth') )
    data_list.append( get_db_props('visualizing_environmental') )
    data_list.append( get_db_props('mu284') )
    data_list.append( get_db_props('ar4') )
    
    
    data_list.append( get_db_props('vineyard') )
    data_list.append( get_db_props('confidence') )
    
    
    
    # data bases with problens in 
    # LTSA
    # # # data_list.append( get_db_props('rabe_131') )
    # # # data_list.append( get_db_props('hayes-roth') )
    # # # data_list.append( get_db_props('diabetes') )
    
    # data bases with problems in 
    # RFE_NonLinearGradientBoostingMachines
    #data_list.append( get_db_props('mux6') )
    
    # data bases with problems in
    # Quadratic Discriminant 
    #data_list.append( get_db_props('Engine1') )
    
    # data bases with problems in
    # Decision Tree
    #data_list.append( get_db_props('prnn_viruses') )
    
    data_list.append( get_db_props('iris') )
    data_list.append( get_db_props('prnn_crabs') )
    data_list.append( get_db_props('balance-scale') )
    #erro data_list.append( get_db_props('parity5') ) 
    data_list.append( get_db_props('servo') )
    data_list.append( get_db_props('monks-problems-1') )
    data_list.append( get_db_props('bolts') )
    data_list.append( get_db_props('fri_c2_100_10') )
    data_list.append( get_db_props('threeOf9') )
    data_list.append( get_db_props('fri_c3_100_5') )
    data_list.append( get_db_props('baskball') )
    data_list.append( get_db_props('newton_hema') )
    data_list.append( get_db_props('strikes') )
    data_list.append( get_db_props('datatrieve') )
    data_list.append( get_db_props('diggle_table_a2') )
    data_list.append( get_db_props('fl2000') )
    #erro data_list.append( get_db_props('triazines') )
    
    data_list.append( get_db_props('veteran') )
    data_list.append( get_db_props('car') )
    data_list.append( get_db_props('prnn_fglass') )
    #erro data_list.append( get_db_props('analcatdata_creditscore') )
    
    
    
    return data_list



def get_table_results(db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc):
    db_result = {
        'db': {
                'name': db_props['name'],
                'classes': db_props['classes'],
                'samples_per_class': db_props['samples_per_class'],
                'samples_total': db_props['samples_total'],
                'dimensionality': db_props['dimensionality'],
                
                'dr': []
            },
    }
    
    dr_props = dr_struct.get_dr_props(dr_technique)
    dr_technique = {
                        'name': dr_props['name'],
                        'learning': dr_props['learning'],
                        'transformation': dr_props['transformation'],
                        'structure': dr_props['structure'],
                        
                        'clf': dr_props['clf']
                    }
    
    for idx, cls_label in enumerate(classifiers_labels):
        clf = {
                    'name': cls_label,
                    'accuracy': acc[idx],
                    'kappa': kappa[idx],
                    'balanced_accuracy': balanced_acc[idx]
                }
        dr_technique['clf'].append(clf)    
    
    db_result['db']['dr'].append(dr_technique)
    
    return db_result



def add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc):
    db_result = {
        'db': {
                'name': db_props['name'],
                'classes': db_props['classes'],
                'samples_per_class': db_props['samples_per_class'],
                'samples_total': db_props['samples_total'],
                'dimensionality': db_props['dimensionality'],
                
                'dr': []
            },
    }
    
    dr_props = dr_struct.get_dr_props(dr_technique)
    dr_technique = {
                        'name': dr_props['name'],
                        'learning': dr_props['learning'],
                        'transformation': dr_props['transformation'],
                        'structure': dr_props['structure'],
                        
                        'clf': dr_props['clf']
                    }
    
    for idx, cls_label in enumerate(classifiers_labels):
        clf = {
                    'name': cls_label,
                    'accuracy': acc[idx],
                    'kappa': kappa[idx],
                    'balanced_accuracy': balanced_acc[idx]
                }
        dr_technique['clf'].append(clf)    
    
    db_result['db']['dr'].append(dr_technique)
    
    table_results.append(db_result)