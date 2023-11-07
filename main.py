# __all__ = []

import scipy.io
import numpy as np
from kneed import KneeLocator

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

variancia_total = np.var(bi_casca_maca, axis=0).sum()
elbow = KneeLocator(range(1, 230 + 1), variancia_total * np.arange(1, 230 + 1), curve="convex", direction="decreasing")
n_componentes_otimo = elbow.elbow

print()
