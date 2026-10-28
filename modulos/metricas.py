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


# ------------------------------------------------------------
# Segmentación respecto de una máscara de referencia
# ------------------------------------------------------------

def dice(a, b):
    """Dice = 2·|A ∩ B| / (|A| + |B|) entre dos máscaras binarias (1 = iguales, 0 = no se tocan)."""
    suma = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / suma if suma else 1.0


def iou(a, b):
    """IoU (Jaccard) = |A ∩ B| / |A ∪ B| entre dos máscaras binarias."""
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 1.0


# ------------------------------------------------------------
# Error de una medición respecto de una referencia
# ------------------------------------------------------------

def mae(medido, referencia):
    """Error absoluto medio. Ignora los pares con algún NaN."""
    error = np.asarray(medido, float) - np.asarray(referencia, float)
    return np.nanmean(np.abs(error))


def rmse(medido, referencia):
    """Raíz del error cuadrático medio. Ignora los pares con algún NaN."""
    error = np.asarray(medido, float) - np.asarray(referencia, float)
    return np.sqrt(np.nanmean(error ** 2))
