# __all__ = []

import scipy.io
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from hyp_print import generate_file_hyp, assinatura_espectral_polpa, assinatura_espectral_casca


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
# bi_casca_marmelo = calc_bidimensional(hyp_casca_marmelo, 'CASCA MARMELO: ')
# bi_casca_nanica = calc_bidimensional(hyp_casca_nanica, 'CASCA NANICA: ')
# bi_casca_prata = calc_bidimensional(hyp_casca_prata, 'CASCA PRATA: ')
# bi_polpa_maca = calc_bidimensional(hyp_polpa_maca, 'POLPA MACA: ')
# bi_polpa_marmelo = calc_bidimensional(hyp_polpa_nanica, 'POLPA NANICA: ')
# bi_polpa_prata = calc_bidimensional(hyp_polpa_prata, 'POLPA PRATA: ')


# pca = PCA(n_components=3)
# pca.fit(bi_casca_maca)
# print(pca.explained_variance_ratio_)

# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(bi_casca_maca)
#
# pc1 = pca_result[:, 0]
# pc2 = pca_result[:, 1]
#
# colors = np.where(pc2 >= 0, 'blue', 'red')
#
# plt.figure(figsize=(8, 6))
# plt.scatter(pc1, pc2, c=colors)
#
# plt.xlabel('PC1')
# plt.ylabel('PC2')
#
# plt.title('Gráfico de Dispersão Casca Maca')
# plt.show()


# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(bi_casca_maca)
#
# # Extrair os valores dos PCs
# pc1 = pca_result[:, 0]
# pc2 = pca_result[:, 1]
# pc3 = pca_result[:, 2]
#
# # Plotar um gráfico 3D com cada PC colorido de forma diferente
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for i in range(len(pc1)):
#     if i % 3 == 0:
#         color = 'blue'  # PC1 em azul
#     elif i % 3 == 1:
#         color = 'red'  # PC2 em vermelho
#     else:
#         color = 'yellow'  # PC3 em amarelo
#
#     ax.scatter(pc1[i], pc2[i], pc3[i], c=color)
#
# # Configurar os rótulos dos eixos
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
#
# # Definir título
# plt.title('Gráfico 3D dos PCs (PC1 em azul, PC2 em vermelho, PC3 em amarelo)')
#
# # Mostrar o gráfico
# plt.show()
#
# print()

pca = PCA(n_components=3)
pca_result = pca.fit_transform(bi_casca_maca)

# Extrair os valores dos PCs
pc1 = pca_result[:, 0]
pc2 = pca_result[:, 1]
pc3 = pca_result[:, 2]

# Criar cores com base nos PCs
colors = ['blue' if pc == 0 else 'red' if pc == 1 else 'yellow' for pc in range(3)]

# Plotar um gráfico de dispersão em 2D com cada PC em uma cor diferente
plt.figure(figsize=(8, 6))
plt.scatter(pc1, pc2, c=colors)

# Configurar os rótulos dos eixos
plt.xlabel('PC1')
plt.ylabel('PC2')

# Definir título
plt.title('Gráfico de Dispersão Casca Maca')

# Mostrar o gráfico
plt.show()
