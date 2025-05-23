"""
Created on Thu Jun 21 06:08:53 2023

Semi Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import numpy as np
import scipy as sp
from numpy import sqrt

import sklearn.neighbors as sknn
import networkx as nx

import sklearn.datasets as skdata
import networkx as nx

import utils.clustering as clustering

# ISOMAP-KL implementation
def GeodesicIsomap(dados, k, d, target, prediction_mode="GMM", proportion=0.1):
    
    if not clustering.is_valid_mode(prediction_mode):
        print('prediction not implemented')
        return
    
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

    
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                B[i, j] = np.linalg.norm(delta)  


    # Apply Gaussian Mixture Model (GMM) or DBSCAN or another mode implemented 
    self_labels = clustering.predict(prediction_mode, dados, target)

    
    #Semi-Supervised
    selected_edges = select_edges(B, proportion, "betweenness_centrality")
    for tup in selected_edges:
        i, j = tup[0]
        delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                
        if self_labels[i] == self_labels[j]:
            B[i, j] = min(delta)
            #print('non-equal', i, j)
        else:
            B[i, j] = max(delta) 
            #print('equal', i, j)
            
    # for i in range(n):
    #     for j in range(n):
    #         if B[i, j] > 0 and any((i,j) == tup[0] for tup in selected_edges):
    #             delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                
    #             if self_labels[i] == self_labels[j]:
    #                 B[i, j] = min(delta)
    #                 #print('non-equal', i, j)
    #             else:
    #                 B[i, j] = max(delta) 
    #                 #print('equal', i, j)
    
    # Computes geodesic distances in B
    G = nx.from_numpy_matrix(B)
    #G = nx.from_numpy_array(B) #G = nx.from_numpy_array(B) # pip install networkx==3.1
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
    
    return output

def select_edges(cov_matrix, proportion=.1, selection_mode="betweenness_centrality"):
    
    if selection_mode=="betweenness_centrality":
        G = nx.from_numpy_matrix(cov_matrix)
        edge_betweenness = nx.edge_betweenness_centrality(G)
        #for edge, centrality in edge_betweenness.items():
        #    print(f"Betweenness Centrality = {centrality:.6f} - Aresta {edge}")
        
        # betweenness centrality em ordem decrescente
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

        # número de arestas a serem selecionadas (proporção do total)
        num_edges_to_select = int(len(sorted_edges) * proportion)

        # Seleciona a proporção do parâmetro com maiores betweenness centrality
        selected_edges = sorted_edges[:num_edges_to_select]
    else:
        selected_edges = None

    return selected_edges

def test_GeodesicIsomap():
    db = skdata.load_iris()
    X = db['data']
    y = db['target']
    n = X.shape[0]
    nn = round(sqrt(n))


    # Treat catregorical features
    if not isinstance(X, np.ndarray):
        cat_cols = X.select_dtypes(['category']).columns
        X[cat_cols] = X[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        X = X.to_numpy()
        target = y.to_numpy()
        
    from sklearn import preprocessing
    # Data standardization (to deal with variables having different units/scales)
    X = preprocessing.scale(X)

    Y = GeodesicIsomap(X, k=5, d=nn, target=y)



if __name__ == "__main__":
    test_GeodesicIsomap()
