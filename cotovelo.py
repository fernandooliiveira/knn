import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def pca_cotovelo(bi):
    n_components = range(1, 15)
    explained_variance = []

    for n in n_components:
        print(n)
        pca = PCA(n_components=n)
        pca.fit(bi)
        explained_variance.append(np.sum(pca.explained_variance_ratio_))

    # Plote a curva da variância explicada
    plt.figure(figsize=(8, 4))
    plt.plot(n_components, explained_variance, marker='o', linestyle='-')
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Variância')
    plt.title('Método do Cotovelo para determinar o número de PCs')
    plt.grid(True)
    plt.show()

def kmeans_cotovelo(pca_ap):
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pca_ap)
        wcss.append(kmeans.inertia_)

    # Plote a curva do método do cotovelo
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Quadrados das Distâncias Intra-Cluster')
    plt.show()
