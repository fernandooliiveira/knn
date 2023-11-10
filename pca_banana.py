import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def pca_banana(pca, mt, title):
    # pca = PCA(n_components=2)
    pca_result = pca.fit_transform(mt)

    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]

    colors = np.where(pc2 >= 0, 'blue', 'red')

    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, c=colors)

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.title(title)
    # plt.show()
    return pca_result


