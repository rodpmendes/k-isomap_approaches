import numpy as np
import sklearn.neighbors as sknn

dados = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

n = dados.shape[0]
m = dados.shape[1]


matriz_pcs = np.zeros((n, m, m))

k=2
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