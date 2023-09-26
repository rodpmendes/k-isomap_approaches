"""
Created on Thu Jul 20 20:05:40 2023

Unsupervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import numpy as np
from numpy import sqrt
import umap                 # install with: pip install umap
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import networkx as nx
import sklearn.neighbors as sknn
import scipy as sp

def UnsupervisedLLE(data):
    lle = LocallyLinearEmbedding(n_components=2)
    lle_data = lle.fit_transform(data)
    return lle_data

def UnsupervisedUMAP(data):
    model = umap.UMAP(n_components=2)
    umap_data  = model.fit_transform(data)
    #umap_data = umap_data.T # verify dimension of T ??
    return umap_data

def UnsupervisedTSNE(data, dimensionOfEmbeddedSpace=2):

    numberOfNeighbors = GetNumberOfNeighbors(data.shape[0])
    model = TSNE(n_components=dimensionOfEmbeddedSpace, perplexity=numberOfNeighbors)
    tsne_data = model.fit_transform(data, None)
    
    return tsne_data

def GetNumberOfNeighbors(n):
    return round(sqrt(n))


def UnsupervisedISOMAP(data):
    model = Isomap(n_neighbors=20, n_components=2)
    isomap_data = model.fit_transform(data)
    #isomap_data = isomap_data.T # verify dimension of T ??
    
    return isomap_data


def UnsupervisedLapEig(data):
    model = SpectralEmbedding(n_neighbors=20, n_components=2)
    lap_eig_data = model.fit_transform(data)
    #lap_eig_data = lap_eig_data.T # verify dimension of T ??
    return lap_eig_data


def UnsupervisedLTSA(data, nn, n):
    model = LocallyLinearEmbedding(n_neighbors=nn, n_components=n, method='ltsa')
    ltsa_data = model.fit_transform(data)
    #ltsa_data = ltsa_data.T # verify dimension of T ??
    return ltsa_data


def UnsupervisedLDANonComponents(data, labels):
    model = LinearDiscriminantAnalysis(n_components=None)
    data_lda = model.fit_transform(data, labels) #labels is not optional ??
    
    return data_lda

# PCA implementation
def UnsupervisedPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


# K-ISOMAP implementation
def GeodesicIsomap(dados, k, d):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
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
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                B[i, j] = np.linalg.norm(delta)                
               
    # Computes geodesic distances in B
    #D = sksp.graph_shortest_path(B, directed=False)
    G = nx.from_numpy_matrix(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Remove infs or nans
    maximo = np.nanmax(B[B != np.inf])   
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
    
    return output