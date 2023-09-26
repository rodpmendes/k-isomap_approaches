"""
Created on Thu Set 26 02:58:43 2023

Semi Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import unsupervised_dr as unsup_dr
import supervised_dr as sup_dr
import semi_supervised_dr as semisup_dr
import self_supervised_dr as selfsup_dr

from utils import myClassifiers as mycls
from utils import data_structures as ds
from utils.plot_utils import GroupedBarChartPlot, HeatMapChartPlot

import numpy as np
from numpy import sqrt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def main():
        
    # Set global TSNE parameters
    PERPLEXITY = 20
    SEED = 1                  # Random seed
    MOMENTUM = 0.9
    LEARNING_RATE = 30
    NUM_ITERS = 1             # Num iterations to train for
    TSNE = False              # If False, Symmetric SNE
    NUM_PLOTS = 5             # Num. times to plot in training

    # numpy RandomState for reproducibility
    rng = np.random.RandomState(SEED)

    # Load data
    data_list = ds.get_data_list()

    # Heat Map Plot Configuration
    rows = 4
    column = 3
    index = 1
    
    # result objects
    acc_values = []
    kappa_values = []
    table_results = []
    round_decimals = 2

    for db_name, db, db_props in data_list:
        print(f'db_name: {db_name}')
        
        acc_values = []
        kappa_values = []
        balanced_acc_values = []

        X = db['data']
        y = db['target']
        
        # Treat catregorical features (required for some OpenML datasets)
        if not isinstance(X, np.ndarray):
            cat_cols = X.select_dtypes(['category']).columns
            X[cat_cols] = X[cat_cols].apply(lambda x: x.cat.codes)
            # Convert to numpy array
            X = X.to_numpy()
            y = y.to_numpy()

        if type(y[0]) == str:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Normalize data
        X = preprocessing.scale(X)
    
        # Fit Unsup K-ISOMAP
        dr_technique = 'Unsup K-ISOMAP'
        n = X.shape[0]
        nn = round(sqrt(n))
        Y = unsup_dr.GeodesicIsomap(X, nn, 2)
        classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
        
        acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
        kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
        balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            
        ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
        print(f'dr_technique: {dr_technique}')


        # Fit Sup K-ISOMAP
        dr_technique = 'Sup K-ISOMAP'
        n = X.shape[0]
        nn = round(sqrt(n))
        Y = sup_dr.GeodesicIsomap(X, nn, 2, y)
        classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
        
        acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
        kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
        balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            
        ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
        print(f'dr_technique: {dr_technique}')    
        
        
        
        # Fit Semi-Sup K-ISOMAP GMM
        dr_technique = 'Semi-Sup K-ISOMAP GMM'
        n = X.shape[0]
        nn = round(sqrt(n))
        Y = semisup_dr.GeodesicIsomap(X, nn, 2, y, prediction_mode="GMM", proportion=0.1)
                        
        classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
        
        acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
        kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
        balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            
        ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
        print(f'dr_technique: {dr_technique}') 
        
        
        # Fit Semi-Sup K-ISOMAP DBSCAN
        dr_technique = 'Semi-Sup K-ISOMAP DBSCAN'
        n = X.shape[0]
        nn = round(sqrt(n))
        Y = semisup_dr.GeodesicIsomap(X, nn, 2, y, prediction_mode="DBSCAN", proportion=0.1)
                        
        classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
        
        acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
        kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
        balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            
        ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
        print(f'dr_technique: {dr_technique}')
        
        
        # Fit Self-Sup K-ISOMAP GMM
        dr_technique = 'Self-Sup K-ISOMAP GMM'
        n = X.shape[0]
        nn = round(sqrt(n))
        Y = selfsup_dr.GeodesicIsomap(X, nn, 2, y, prediction_mode="GMM")
                        
        classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
        
        acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
        kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
        balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            
        ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
        print(f'dr_technique: {dr_technique}')
        
        
        # Fit Self-Sup K-ISOMAP DBSCAN
        dr_technique = 'Self-Sup K-ISOMAP DBSCAN'
        n = X.shape[0]
        nn = round(sqrt(n))
        Y = selfsup_dr.GeodesicIsomap(X, nn, 2, y, prediction_mode="DBSCAN")
                        
        classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
        
        acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
        kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
        balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            
        ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
        print(f'dr_technique: {dr_technique}')
        
        
        if 1==2:
            HeatMapChartPlot(acc_values, classifiers_labels, 'Accuracy', db_name, plot=(rows, column, index))
            if index == 12:
                index = 1
            else:
                index += 1
            
            HeatMapChartPlot(kappa_values, classifiers_labels, 'Kappa', db_name, plot=(rows, column, index))
            if index == 12:
                index = 1
            else:
                index += 1
            
            HeatMapChartPlot(balanced_acc_values, classifiers_labels, 'Balanced Acc', db_name, plot=(rows, column, index))
            if index == 12:
                index = 1
            else:
                index += 1

            GroupedBarChartPlot(acc_values, classifiers_labels, db_name, 'Accuracy')
            GroupedBarChartPlot(kappa_values, classifiers_labels, db_name, 'Kappa')
            GroupedBarChartPlot(balanced_acc_values, classifiers_labels, db_name, 'Balanced Acc')

        #input()
        

        if 1 == 3:

            # # Obtain matrix of joint probabilities p_ij
            # P = mycls.p_joint(X, y, PERPLEXITY)

            # # Fit SNE
            # Y = mycls.estimate_sne(X, y, P, rng, num_iters=NUM_ITERS, q_fn=mycls.q_tsne if TSNE else mycls.q_joint, grad_fn=mycls.tsne_grad if TSNE else mycls.symmetric_sne_grad, learning_rate=LEARNING_RATE, momentum=MOMENTUM, plot=NUM_PLOTS)
            # acc, acc_labels, kappa = mycls.myClassification(Y, y)
            # plot_values.append( (np.round(acc, round_decimals), ['SNE']) )
            # kappa_values.append( (np.round(kappa, round_decimals), ['SNE']) )
            

            #UnsupervisedLDANonComponents
            # Fit LDAcompNone
            dr_technique = 'LDAcompNone'
            Y = unsup_dr.UnsupervisedLDANonComponents(X, y) #y is not optional ??
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit LLE
            dr_technique = 'LLE'
            Y = unsup_dr.UnsupervisedLLE(X)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')

        
            # Fit UMAP
            dr_technique = 'UMAP'
            Y = unsup_dr.UnsupervisedUMAP(X)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
        
        
            # Fit ISOMAP
            dr_technique = 'ISOMAP'
            Y = unsup_dr.UnsupervisedISOMAP(X)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Laplacian Eigenmaps
            dr_technique = 'LapEig'
            Y = unsup_dr.UnsupervisedLapEig(X)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            #
            # ?? problens with multiple databases
            # nan depending on n_neighbors or standardized data
            #
            # # Fit LTSA
            # dr_technique = 'LTSA'
            # Y = unsup_dr.UnsupervisedLTSA(X, nn, 2) 
            # classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            # acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            # kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            # balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            # add_table_results(results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            # print(f'dr_technique: {dr_technique}')
            
        
            # Fit LDA
            dr_technique = 'LDA'
            Y = sup_dr.SupervisedLDA(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit t-SNE
            dr_technique = 't-SNE'
            Y = unsup_dr.UnsupervisedTSNE(X)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised PCA
            dr_technique = 'SUP PCA'
            Y = sup_dr.SupervisedPCA(X, y, 2)
            Y = Y.T
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y.real, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            # Fit Unsupervised PCA
            dr_technique = 'UNSUP PCA'
            Y = unsup_dr.UnsupervisedPCA(X, 2)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y.real.T, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised NCA
            dr_technique = 'NCA'
            Y = sup_dr.SupervisedNCA(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Logist Regression
            dr_technique = 'RFE_LogistReg'
            Y = sup_dr.SupervisedLinearRFE_LogistReg(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Linear regression
            dr_technique = 'RFE_LinearReg'
            Y = sup_dr.SupervisedLinearRFE_LinearReg(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Linear SVC
            dr_technique = 'RFE_LinearSVC'
            Y = sup_dr.SupervisedLinearRFE_LinearSVC(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Decision Trees
            dr_technique = 'RFE_NonLinearDecisionTrees'
            Y = sup_dr.SupervisedNonLinearRFE_DecisionTree(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Random Forests
            dr_technique = 'RFE_NonLinearRandomForests'
            Y = sup_dr.SupervisedNonLinearRFE_RandomForest(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Gradient Boosting Machines
            dr_technique = 'RFE_NonLinearGradientBoostingMachines'
            Y = sup_dr.SupervisedNonLinearRFE_GradientBoostingMachines(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised RFE with Neural Networks
            # dr_technique = 'RFE_NonLinearMLP'
            # Y = dr.SupervisedNonLinearRFE_MLP(X, y)
            # classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            # acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            # kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            # ds.add_table_results(results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            
            
            #
            # Fit Supervised SelectKBest
            # not a Dimensionality Reduction
            # ?? why it is here ??
            #
            # dr_technique = 'SelectKBest'
            # Y = sup_dr.SupervisedSelectKBest(X, y)
            # classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            # acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            # kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            # ds.add_table_results(results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            # print(f'dr_technique: {dr_technique}')
            
            
            # Fit Supervised PLS Regression
            dr_technique = 'PLS_Regression'
            Y = sup_dr.SupervisedPLSRegression(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            
            # Fit Supervised Deep Learning (Test size 60%)
            dr_technique = 'DeepLearning'
            Y = sup_dr.SupervisedDeepLearning(X, y)
            classifiers_labels, acc, kappa, balanced_acc = mycls.myClassification(Y, y)
            acc_values.append( (np.round(acc, round_decimals), [dr_technique]) )
            kappa_values.append( (np.round(kappa, round_decimals), [dr_technique]) )
            balanced_acc_values.append( (np.round(balanced_acc, round_decimals), [dr_technique]) )
            ds.add_table_results(table_results, db_props, dr_technique, classifiers_labels, acc, kappa, balanced_acc)
            print(f'dr_technique: {dr_technique}')
            
            
            #HeatMapChartPlot(acc_values, classifiers_labels, 'Accuracy', db_name, plot=(rows, column, index))
            #index += 1
            
            #HeatMapChartPlot(kappa_values, classifiers_labels, 'Kappa', db_name, plot=(rows, column, index))
            #index += 1

    ds.export_to_csv(table_results, 'table_results_20092023.csv')

    #GroupedBarChartPlot(acc_values, acc_labels)
    #GroupedBarChartPlot(kappa_values, acc_labels)

    input();




if __name__ == "__main__":
    main()
