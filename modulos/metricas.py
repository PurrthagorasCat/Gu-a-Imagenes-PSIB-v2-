"""
Métricas compartidas entre módulos.
"""

import numpy as np
from skimage.metrics import structural_similarity


# ------------------------------------------------------------
# Calidad de imagen respecto de una referencia
# ------------------------------------------------------------

def psnr(referencia, imagen, rango=255):
    """PSNR en dB: 10·log10(rango² / ECM). Infinito si las imágenes son iguales."""
    ecm = np.mean((referencia.astype(np.float64) - imagen.astype(np.float64)) ** 2)
    return np.inf if ecm == 0 else 10 * np.log10(rango ** 2 / ecm)


def ssim(referencia, imagen, rango=255):
    """Índice de similitud estructural entre imágenes en escala de grises (1 = idénticas)."""
    return structural_similarity(referencia, imagen, data_range=rango)


# Pendiente: Dice, Jaccard/IoU
# Pendiente: MAE, RMSE