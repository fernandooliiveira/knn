# __all__ = []

import scipy.io
# import numpy as np

PATH_CASCA_MACA = 'data/casca_maca.mat'
PATH_CASCA_MARMELO = 'data/casca_marmelo.mat'
PATH_CASCA_NANICA = 'data/casca_nanica.mat'
PATH_CASCA_PRATA = 'data/casca_prata.mat'
PATH_POLPA_MACA = 'data/polpa_maca.mat'
PATH_POLPA_NANICA = 'data/polpa_nanica.mat'
PATH_POLPA_PRATA = 'data/polpa_prata.mat'


# Função para reorganizar matrizes tridimensionais em bidimensionais
# def reorganizar_matrizes(matrizes_tridimensionais):
#     matrizes_bidimensionais = []
#
#     for matriz_tridimensional in matrizes_tridimensionais:
#         print(matrizes_tridimensionais.shape)
#         altura, largura, comprimento_de_onda = matrizes_tridimensionais.shape
#
#         # Reorganize a matriz tridimensional em uma matriz bidimensional
#         x = matriz_tridimensional.reshape(comprimento_de_onda, altura * largura)
#         print(x)
#     # matriz_bidimensional = matriz_tridimensional.reshape(comprimento_de_onda, altura * largura)
#
#     # matrizes_bidimensionais.append(matriz_bidimensional)
#
#     return matrizes_bidimensionais

def calc_bidimensional(mt, title):
    print(title)
    print(mt.shape)
    height, width, lamb = mt.shape
    return mt.reshape(height * width, lamb)


# scipy.io.loadmat(file_name, mdict=None, appendmat=True, **kwargs)
# loaded = scipy.io.loadmat(PATH_CASCA_MACA).get('casca_maca')
bi_casca_maca = calc_bidimensional(scipy.io.loadmat(PATH_CASCA_MACA).get('casca_maca'), 'CASCA MACA: ')
bi_casca_marmelo = calc_bidimensional(scipy.io.loadmat(PATH_CASCA_MARMELO).get('casca_marmelo'), 'CASCA MARMELO: ')
bi_casca_nanica = calc_bidimensional(scipy.io.loadmat(PATH_CASCA_NANICA).get('casca_nanica'), 'CASCA NANICA: ')
# bi_casca_maca = calc_bidimensional(scipy.io.loadmat(PATH_CASCA_MACA).get('casca_maca'), 'CASCA MACA: ')
# bi_casca_maca = calc_bidimensional(scipy.io.loadmat(PATH_CASCA_MACA).get('casca_maca'), 'CASCA MACA: ')
# print(bi_casca_maca)
# print(loaded.get('casca_maca'))

# cascaMaca = calc_bidimensional(loaded)
# print(cascaMaca)

# mat_data = scipy.io.loadmat(mat_file_path)
#
# dados = mat_data['casca_maca']
#
# altura, largura, comprimento_de_onda = dados.shape
#
# x = dados.reshape(altura * largura, comprimento_de_onda)
# print(x)
