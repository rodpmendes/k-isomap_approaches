"""
Created on Thu Jun 15 05:03:53 2023

Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import numpy as np
import scipy as sp
from numpy import sqrt


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from sklearn.feature_selection import RFE
from sklearn.cross_decomposition import PLSRegression


# Linear
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

#Non Linear
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.neighbors as sknn
from numpy.linalg import norm
import networkx as nx

from scipy.sparse.linalg import eigsh
import torch

import time

# Supervised PCA implementation (variation from paper Supervised Principal Component Analysis - Pattern Recognition)
def SupervisedPCA(dados, labels, d):

    dados = dados.T

    m = dados.shape[0]      # number of samples
    n = dados.shape[1]      # number of features

    I = np.eye(n)
    U = np.ones((n, n))
    H = I - (1/n)*U

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                L[i, j] = 1

    Q1 = np.dot(dados, H)
    Q2 = np.dot(H, dados.T)
    Q = np.dot(np.dot(Q1, L), Q2)

    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(Q)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados)
    
    return novos_dados

def SupervisedLDA(data, labels):
    # Local Fisher Discriminant Analysis (LFDA)
    # use number of components
    # ?? search about references ??
    
    components = GetNumberOfComponents(labels)
    model = LinearDiscriminantAnalysis(n_components=components)
    data_lda = model.fit_transform(data, labels)
    
    return data_lda
    
def GetNumberOfComponents(labels):
    components = len(np.unique(labels))
    
    if components > 2:
        components = 2
    else:
        components = 1
    
    return components


def SupervisedNCA(data, labels):

    model = NeighborhoodComponentsAnalysis(random_state=42)
    data_nca = model.fit_transform(data, labels)
    
    return data_nca

def SupervisedLinearRFE_LogistReg(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = LogisticRegression()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedLinearRFE_LinearReg(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = LinearRegression()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedLinearRFE_LinearSVC(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = LinearSVC()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedNonLinearRFE_DecisionTree(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = DecisionTreeClassifier()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedNonLinearRFE_RandomForest(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = RandomForestClassifier()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedNonLinearRFE_GradientBoostingMachines(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = GradientBoostingClassifier()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedNonLinearRFE_MLP(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    clf = MLPClassifier()

    model = RFE(estimator=clf, n_features_to_select=n_components)
    data_rfe = model.fit_transform(data, labels)
    
    return data_rfe

def SupervisedSelectKBest(data, labels):
    #
    # Feature Selection
    # not a dimensionality reduction technique
    #
    
    model = SelectKBest(score_func=f_classif, k=2)
    data_kbest = model.fit_transform(data, labels)
    
    return data_kbest


# ISOMAP-KL implementation
def GeodesicIsomap(dados, k, d, target, use_np = False, use_gpu = False):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m), dtype=np.float32)
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    print('###### Computes the means and covariance matrices for each patch')
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            
            if use_np:
                start_time = time.time()
                v, w = np.linalg.eigh(np.cov(amostras.T))
                np_time = time.time() - start_time
                print('i', i, 'np_time', np_time)
            elif use_gpu and torch.cuda.is_available():
                start_time = time.time()
                torch_gpu_matrix = torch.tensor(np.cov(amostras.T)).to("cuda")
                torch_v, torch_w = torch.linalg.eigh(torch_gpu_matrix)
                v = torch_v.cpu().numpy()
                w = torch_w.cpu().numpy()
                torch_gpu_time = time.time() - start_time
                print('i', i, 'torch_gpu_time', torch_gpu_time)
            else:
                start_time = time.time()
                torch_cpu_matrix = torch.tensor(np.cov(amostras.T))
                torch_v, torch_w = torch.linalg.eigh(torch_cpu_matrix)
                v = torch_v.cpu().numpy()
                w = torch_w.cpu().numpy()
                torch_cpu_time = time.time() - start_time
                print('i', i, 'torch_cpu_time', torch_cpu_time)               
            
            #v_is_equal = np.allclose(v, np.sort(torch_v.cpu().numpy()))
            #w_is_equal = np.allclose(w, np.sort(torch_w.cpu().numpy()))
            #print('v is equal', v_is_equal)
            #print('w is equal', w_is_equal)
            
            #is_symmetric = np.allclose(amostras, amostras.T)
            # Calculate the sparsity ratio
            #sparsity = 1.0 - (np.count_nonzero(amostras) / amostras.size)

            # if is_symmetric and sparsity < .3:
            #     v, w = np.linalg.eigh(np.cov(amostras.T))
            # elif sparsity > .3:
            #     v, w = np.linalg.eigsh(np.cov(amostras.T), k=5)
            # else:
            #     v, w = np.linalg.eig(np.cov(amostras.T))
            
            
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            
            maiores_autovetores = w[:, ordem[::-1]]     # Esse é o oficial!
            
            #maiores_autovetores = w[:, ordem[-1:]]      # Pega apenas os 2 primeiros (teste)
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    print('###### Defines the patch-based matrix (graph)')
    B = A.copy()
    for i in range(n):
        start_time = time.time()
        for j in range(n):
            start_time_j = time.time()
            if B[i, j] > 0:
                #delta = 0
                #delta = np.zeros(m)
                #for k in range(m):
                    #delta = delta + np.linalg.norm(matriz_pcs[i, :, k] - matriz_pcs[j, :, k])
                    #delta[k] = np.linalg.norm(matriz_pcs[i, :, k] - matriz_pcs[j, :, k])
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                #B[i, j] = delta
                if target[i] == target[j]:
                    B[i, j] = min(delta)
                else:
                    #B[i, j] = sum(delta)
                    #B[i, j] = np.sum(delta)/len(delta)  # Mean curvature
                    B[i, j] = max(delta)
                    #B[i, j] = min(delta)
                    #B[i, j] = min(delta)*max(delta)    # Gaussian curvature
                    #B[i, j] = max(delta) - min(delta)
                    #print(B[i, j])
            patch_based_time_j = time.time() - start_time_j
            print('j', j, 'patch_based_time_j', patch_based_time_j)
            
        patch_based_time = time.time() - start_time
        print('i', i, 'patch_based_time', patch_based_time)
        
    
    # Computes geodesic distances in B
    start_time = time.time()
    #G = nx.from_numpy_matrix(B)
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    #print(np.isnan(B).any())
    #print(np.isinf(B).any())
    # Pode gerar nan ou inf na matriz B
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])   # encontra o maior elemento que não seja inf
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    geodesic_distance_time = time.time() - start_time
    print('geodesic_distance_time', geodesic_distance_time)
    
    
    return output


def SupervisedPLSRegression(data, labels):
    n_components = GetNumberOfComponents(labels)
    
    model = PLSRegression(n_components=n_components)

    data_transformed, labels_transformed = model.fit_transform(data, labels)
    
    return data_transformed



def SupervisedDeepLearning(data, labels):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    encoding_dim = 2  # Number of neurons in the bottleneck layer (the reduced dimensionality)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.6, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data)
    X_test_scaled = scaler.transform(X_test)

    # Define the autoencoder model
    input_dim = data.shape[1]
    
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder_layer = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoder_layer = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder_layer)

    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder_layer)

    # Compile the autoencoder model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder on the training data
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=16, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))

    # Use the trained autoencoder to transform the data
    transformed_data = autoencoder.predict(X_train_scaled)
    
    return transformed_data









# ISOMAP-KL implementation
def GeodesicIsomap_v2(dados, k, d, target, use_np = False, use_gpu = False):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m), dtype=np.float32)
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    print('###### Computes the means and covariance matrices for each patch')
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            
            if use_np:
                start_time = time.time()
                v, w = np.linalg.eigh(np.cov(amostras.T))
                np_time = time.time() - start_time
                print('i', i, 'np_time', np_time)
            elif use_gpu and torch.cuda.is_available():
                start_time = time.time()
                torch_gpu_matrix = torch.tensor(np.cov(amostras.T)).to("cuda")
                torch_v, torch_w = torch.linalg.eigh(torch_gpu_matrix)
                v = torch_v.cpu().numpy()
                w = torch_w.cpu().numpy()
                torch_gpu_time = time.time() - start_time
                print('i', i, 'torch_gpu_time', torch_gpu_time)
            else:
                start_time = time.time()
                torch_cpu_matrix = torch.tensor(np.cov(amostras.T))
                torch_v, torch_w = torch.linalg.eigh(torch_cpu_matrix)
                v = torch_v.cpu().numpy()
                w = torch_w.cpu().numpy()
                torch_cpu_time = time.time() - start_time
                print('i', i, 'torch_cpu_time', torch_cpu_time)               
            
            #v_is_equal = np.allclose(v, np.sort(torch_v.cpu().numpy()))
            #w_is_equal = np.allclose(w, np.sort(torch_w.cpu().numpy()))
            #print('v is equal', v_is_equal)
            #print('w is equal', w_is_equal)
            
            #is_symmetric = np.allclose(amostras, amostras.T)
            # Calculate the sparsity ratio
            #sparsity = 1.0 - (np.count_nonzero(amostras) / amostras.size)

            # if is_symmetric and sparsity < .3:
            #     v, w = np.linalg.eigh(np.cov(amostras.T))
            # elif sparsity > .3:
            #     v, w = np.linalg.eigsh(np.cov(amostras.T), k=5)
            # else:
            #     v, w = np.linalg.eig(np.cov(amostras.T))
            
            
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            
            maiores_autovetores = w[:, ordem[::-1]]     # Esse é o oficial!
            
            #maiores_autovetores = w[:, ordem[-1:]]      # Pega apenas os 2 primeiros (teste)
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    print('###### Defines the patch-based matrix (graph)')
    B = A.copy()
    for i in range(n):
        start_time = time.time()
        for j in range(n):
            start_time_j = time.time()
            if B[i, j] > 0:
                #delta = 0
                #delta = np.zeros(m)
                #for k in range(m):
                    #delta = delta + np.linalg.norm(matriz_pcs[i, :, k] - matriz_pcs[j, :, k])
                    #delta[k] = np.linalg.norm(matriz_pcs[i, :, k] - matriz_pcs[j, :, k])
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                #B[i, j] = delta
                if target[i] == target[j]:
                    B[i, j] = min(delta)
                else:
                    #B[i, j] = sum(delta)
                    #B[i, j] = np.sum(delta)/len(delta)  # Mean curvature
                    B[i, j] = max(delta)
                    #B[i, j] = min(delta)
                    #B[i, j] = min(delta)*max(delta)    # Gaussian curvature
                    #B[i, j] = max(delta) - min(delta)
                    #print(B[i, j])
            #patch_based_time_j = time.time() - start_time_j
            #print('j', j, 'patch_based_time_j', patch_based_time_j)
            
        patch_based_time = time.time() - start_time
        print('i', i, 'patch_based_time', patch_based_time)
        
    
    # Computes geodesic distances in B
    start_time = time.time()
    #G = nx.from_numpy_matrix(B)
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    #print(np.isnan(B).any())
    #print(np.isinf(B).any())
    # Pode gerar nan ou inf na matriz B
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])   # encontra o maior elemento que não seja inf
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    geodesic_distance_time = time.time() - start_time
    print('geodesic_distance_time', geodesic_distance_time)
    
    
    return output