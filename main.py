# __all__ = []
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cotovelo import pca_cotovelo, kmeans_cotovelo
from sklearn.neighbors import KNeighborsClassifier

from pca_banana import pca_banana

PATH_CASCA_MACA = 'data/casca_maca.mat'
PATH_CASCA_MARMELO = 'data/casca_marmelo.mat'
PATH_CASCA_NANICA = 'data/casca_nanica.mat'
PATH_CASCA_PRATA = 'data/casca_prata.mat'
PATH_POLPA_MACA = 'data/polpa_maca.mat'
PATH_POLPA_NANICA = 'data/polpa_nanica.mat'
PATH_POLPA_PRATA = 'data/polpa_prata.mat'

hyp_casca_maca = scipy.io.loadmat(PATH_CASCA_MACA).get('casca_maca')
hyp_casca_marmelo = scipy.io.loadmat(PATH_CASCA_MARMELO).get('casca_marmelo')
hyp_casca_nanica = scipy.io.loadmat(PATH_CASCA_NANICA).get('casca_nanica')
hyp_casca_prata = scipy.io.loadmat(PATH_CASCA_PRATA).get('casca_prata')
hyp_polpa_maca = scipy.io.loadmat(PATH_POLPA_MACA).get('polpa_maca')
hyp_polpa_nanica = scipy.io.loadmat(PATH_POLPA_NANICA).get('polpa_nanica')
hyp_polpa_prata = scipy.io.loadmat(PATH_POLPA_PRATA).get('polpa_prata')


# generate_file_hyp(hyp_casca_maca, 'CASCA_MACA')
# generate_file_hyp(hyp_casca_marmelo, 'CASCA_MARMELO')
# generate_file_hyp(hyp_casca_nanica, 'CASCA_NANICA')
# generate_file_hyp(hyp_casca_prata, 'CASCA_PRATA')
# generate_file_hyp(hyp_polpa_maca, 'POLPA_MACA')
# generate_file_hyp(hyp_polpa_nanica, 'POLPA_NANICA')
# generate_file_hyp(hyp_polpa_prata, 'POLPA_PRATA')

# assinatura_espectral_polpa(hyp_polpa_maca, hyp_polpa_nanica, hyp_polpa_prata)
# assinatura_espectral_casca(hyp_casca_maca, hyp_casca_nanica, hyp_casca_prata, hyp_casca_marmelo)

def calc_bidimensional(mt, title):
    print(title)
    print(mt.shape)
    height, width, lamb = mt.shape
    return mt.reshape(height * width, lamb)


bi_casca_maca = calc_bidimensional(hyp_casca_maca, 'CASCA MACA: ')
bi_casca_marmelo = calc_bidimensional(hyp_casca_marmelo, 'CASCA MARMELO: ')
bi_casca_nanica = calc_bidimensional(hyp_casca_nanica, 'CASCA NANICA: ')
bi_casca_prata = calc_bidimensional(hyp_casca_prata, 'CASCA PRATA: ')
bi_polpa_maca = calc_bidimensional(hyp_polpa_maca, 'POLPA MACA: ')
bi_polpa_nanica = calc_bidimensional(hyp_polpa_nanica, 'POLPA NANICA: ')
bi_polpa_prata = calc_bidimensional(hyp_polpa_prata, 'POLPA PRATA: ')

# pca_cotovelo(bi_casca_maca)
pca = PCA(n_components=2)
# pca_casca_maca = pca_banana(pca, bi_casca_maca, 'Gráfico de Dispersão Casca Maca')
pca_casca_maca = pca.fit_transform(bi_casca_maca)

# pca2 = PCA(n_components=2)
# pca_casca_marmelo = pca_banana(bi_casca_marmelo, 'Gráfico de Dispersão Casca Marmelo')
# pca_casca_marmelo = pca2.fit_transform(bi_casca_marmelo)


# KMEANS
# kmeans_cotovelo(pca_casca_maca)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(pca_casca_maca)
labels = kmeans.labels_

# Visualizando os resultados
# plt.scatter(pca_casca_maca[:, 0], pca_casca_maca[:, 1], c=labels, cmap='viridis', edgecolor='k', s=40)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200,
#             label='Centróides')
# plt.title('Resultado do K-Means com Dois Clusters')
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.legend()
# plt.show()

# KNN
X_train, X_test, labels_train, labels_test = train_test_split(
    pca_casca_maca, labels, test_size=0.2, random_state=42
)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, labels_train)
# knn_classifier.predict(X_test)

labels_pred_test = knn_classifier.predict(X_test)
accuracy_test = accuracy_score(labels_test, labels_pred_test)


labels_pred_train = knn_classifier.predict(pca_casca_maca)
accuracy_train = accuracy_score(labels_train, labels_pred_train)

# labels_pred = knn_classifier.predict(X_train)
# labels_pred = knn_classifier.predict(pca_casca_maca)
# labels_pred = knn_classifier.predict(pca_casca_marmelo)

# height, width, _ = hyp_casca_maca.shape
#
# # Reshape das labels preditas para a forma da imagem original
# labels_pred_reshaped = labels_pred.reshape((height, width))
# # labels_pred_reshaped = labels.reshape((height, width))
#
# # Crie uma matriz de zeros para as bandas R, G, B (assumindo que sua imagem é RGB)
# result_image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Especifique as cores que você deseja atribuir a cada cluster
# cluster_colors = {0: [255, 0, 0], 1: [0, 255, 0]}
#
# # Pinte a imagem de acordo com as labels preditas
# for cluster_label, color in cluster_colors.items():
#     result_image[labels_pred_reshaped == cluster_label] = color

# Exiba a imagem resultante
# plt.imshow(result_image)
# plt.title("Resultado do KNN na Imagem Hiperespectral")
# plt.show()

# Calcule a acurácia comparando as previsões com os rótulos reais
# accuracy = accuracy_score(labels_test, labels_pred)
# Imprima as acurácias
print(f'Acurácia do modelo KNN nos dados de teste: {accuracy_test:.2f}')
print(f'Acurácia do modelo KNN nos dados de treino: {accuracy_train:.2f}')

# Imprima a acurácia
# print(f'Acurácia do modelo KNN: {accuracy:.2f}')

# plt.imshow(bi_casca_maca[:, 0].reshape(231, 320), cmap='viridis')
# plt.title(f'Banda {0} Original')
# plt.show()
# labels_pred = knn_classifier.predict(X_test)
#
# accuracy = accuracy_score(labels_test, labels_pred)
# print(f'Acurácia do modelo KNN: {accuracy}')
# pca_casca_marmelo = pca_banana(bi_casca_marmelo, 'Gráfico de Dispersão Casca Marmelo')
# pca_casca_nanica = pca_banana(bi_casca_nanica, 'Gráfico de Dispersão Casca Nanica')
# pca_casca_prata = pca_banana(bi_casca_prata, 'Gráfico de Dispersão Casca Prata')
# pca_polpa_maca = pca_banana(bi_polpa_maca, 'Gráfico de Dispersão Polpa Maca')
# pca_polpa_nanica = pca_banana(bi_polpa_nanica, 'Gráfico de Dispersão Polpa Nanica')
# pca_polpa_prata = pca_banana(bi_polpa_prata, 'Gráfico de Dispersão Polpa Prata')
