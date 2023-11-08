import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def cotovelo(bi):
    n_components = range(1, 101)
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
    plt.ylabel('Variância Acumulada Explicada')
    plt.title('Método do Cotovelo para determinar o número de PCs')
    plt.grid(True)
    plt.show()
