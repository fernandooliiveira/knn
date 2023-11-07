import scipy
from matplotlib import pyplot as plt


def generate_file_hyp(mt, title):
    banda_selecionada = mt[:, :, 200]
    plt.imshow(banda_selecionada, cmap='gray')
    plt.colorbar()
    plt.title(title)
    file_name = title + '.png'
    plt.savefig(file_name)
    # plt.show()


def assinatura_espectral_polpa(maca, naninca, prata):
    x, y = 150, 100

    assinatura1 = maca[x, y, :]
    assinatura2 = naninca[x, y, :]
    assinatura3 = prata[x, y, :]

    plt.plot(assinatura1, label='POLPA MACA', color='red')
    plt.plot(assinatura2, label='POLPA NANICA', color='green')
    plt.plot(assinatura3, label='POLPA PRATA', color='blue')

    plt.title(f'Assinaturas Espectrais do Pixel ({x}, {y})')
    plt.xlabel('Banda Espectral')
    plt.ylabel('Valor de Reflectância')
    plt.legend()
    # plt.savefig('ASS_POLPA.png')
    plt.show()


def assinatura_espectral_casca(maca, naninca, prata, marmelo):
    x, y = 50, 50

    assinatura1 = maca[x, y, :]
    assinatura2 = naninca[x, y, :]
    assinatura3 = prata[x, y, :]
    assinatura4 = marmelo[x, y, :]

    plt.plot(assinatura1, label='CASCA MACA', color='red')
    plt.plot(assinatura2, label='CASCA NANICA', color='green')
    plt.plot(assinatura3, label='CASCA PRATA', color='blue')
    plt.plot(assinatura4, label='CASCA MARMELO', color='orange')

    plt.title(f'Assinaturas Espectrais do Pixel ({x}, {y})')
    plt.xlabel('Banda Espectral')
    plt.ylabel('Valor de Reflectância')
    plt.legend()
    # plt.savefig('ASS_CASCA.png')
    plt.show()
