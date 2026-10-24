"""
Módulo 1 — Fundamentos, espacios de color y mejoramiento digital.
"""

import cv2                       # OpenCV: redimensionado y conversión entre espacios de color
import numpy as np               # Cálculo numérico (arrays, unique, round)
import matplotlib.pyplot as plt  # Gráficos para mostrar las imágenes


# ------------------------------------------------------------
# 1.1 Dimensiones, resolución espacial, cuantización y color
# ------------------------------------------------------------

def formato_miles(n):
    """Formatea un entero con punto como separador de miles."""
    # Python usa coma para los miles (ej. 1,234); se reemplaza por punto
    return f"{n:,}".replace(",", ".")


def contar_colores(imagen):
    """Cantidad de colores (o niveles de gris) distintos presentes en la imagen."""
    if imagen.ndim == 2:
        # Imagen de un canal: cada valor distinto es un nivel de gris
        return len(np.unique(imagen))
    # Cada color (R, G, B) se empaqueta en un único entero de 24 bits (R en los 8 bits
    # altos, G en los del medio y B en los bajos) para contarlos rápido con unique.
    # Se pasa a uint32 antes de desplazar porque en uint8 los bits se perderían
    codigos = (imagen[..., 0].astype(np.uint32) << 16) | (imagen[..., 1].astype(np.uint32) << 8) | imagen[..., 2]
    return len(np.unique(codigos))


def imprimir_dimensiones(imagen, nombre="Imagen"):
    """Imprime dimensiones, canales, profundidad de bits y cantidades derivadas."""
    alto, ancho = imagen.shape[:2]
    canales = 1 if imagen.ndim == 2 else imagen.shape[2]   # Las imágenes en gris no tienen 3.er eje
    bits_canal = np.iinfo(imagen.dtype).bits               # Bits del tipo de dato (uint8 → 8)
    bits_pixel = bits_canal * canales

    print(nombre)
    print(f"  Dimensiones (ancho x alto) : {ancho} x {alto} px  ({formato_miles(ancho * alto)} píxeles)")
    print(f"  Canales                    : {canales}")
    print(f"  Tipo de dato               : {imagen.dtype}")
    print(f"  Profundidad de bits        : {bits_canal} bits por canal ({2 ** bits_canal} niveles), "
          f"{bits_pixel} bits por píxel")
    print(f"  Colores representables     : {formato_miles(2 ** bits_pixel)}")
    print(f"  Colores distintos presentes: {formato_miles(contar_colores(imagen))}")
    print(f"  Rango de valores presente  : {imagen.min()} a {imagen.max()}")
    print(f"  Memoria sin comprimir      : {imagen.nbytes / 1024:.1f} KB".replace(".", ","))
    print()


def submuestrear(imagen, factor):
    """Reduce la resolución a `factor` (entre 0 y 1) por eje, por vecino más cercano."""
    return cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def ampliar_a(imagen, alto, ancho):
    """Lleva la imagen al tamaño (alto, ancho) repitiendo píxeles, solo para visualizar."""
    # OpenCV recibe el tamaño como (ancho, alto), al revés que shape
    return cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_NEAREST)


def cuantizar(imagen, bits):
    """Cuantización uniforme de 8 bits a `bits` bits por canal."""
    niveles = 2 ** bits
    ancho_intervalo = 256 // niveles
    # División entera para ve a qué intervalo pertenece cada valor
    indice = imagen // ancho_intervalo
    # Se lleva cada intervalo a su nivel dentro de 0–255 (con 2 bits: 0, 85, 170, 255)
    return np.round(indice * (255 / (niveles - 1))).astype(np.uint8)


def rgb_a_hsv(imagen):
    """Convierte de RGB a HSV y devuelve (H en grados 0–360, S, V)."""
    hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
    # OpenCV guarda H entre 0 y 179 (grados / 2) para que entre en 8 bits; se pasa a
    # float antes de multiplicar para que no desborde el uint8
    h_grados = hsv[..., 0].astype(np.float32) * 2
    return h_grados, hsv[..., 1], hsv[..., 2]


# ------------------------------------------------------------
# Visualización
# ------------------------------------------------------------

def mostrar_fila(imagenes, titulos, titulo_figura=None, cmap="gray", rango=(0, 255),
                 barra_color=False, ancho_max=18, alto_max=5):
    """
    Muestra varias imágenes una al lado de la otra.

    Las imágenes de un canal se dibujan con un rango fijo (por defecto 0–255) para
    que matplotlib no estire el contraste de cada una por separado.
    `cmap` y `rango` pueden ser un valor para todas o una lista con uno por imagen.
    """
    n = len(imagenes)
    cmaps = cmap if isinstance(cmap, list) else [cmap] * n
    rangos = rango if isinstance(rango, list) else [rango] * n
    relacion = imagenes[0].shape[1] / imagenes[0].shape[0] 
    alto = min(alto_max, ancho_max / (n * relacion))
    ancho_figura = n * alto * relacion * (1.15 if barra_color else 1) 
    fig, ejes = plt.subplots(1, n, figsize=(ancho_figura, alto + 0.9), squeeze=False, layout="constrained")

    for eje, img, titulo, cm, rg in zip(ejes[0], imagenes, titulos, cmaps, rangos):
        if img.ndim == 3:
            eje.imshow(img)
        else:
            vmin, vmax = rg if rg is not None else (None, None)
            im = eje.imshow(img, cmap=cm, vmin=vmin, vmax=vmax)
            if barra_color:
                fig.colorbar(im, ax=eje, fraction=0.046, pad=0.03)
        eje.set_title(titulo)
        eje.axis("off")

    if titulo_figura:
        fig.suptitle(titulo_figura, fontweight="bold")
    plt.show()


# Pendiente: Transformaciones de intensidad y ecualización (global, CLAHE)
# Pendiente: Ruido y filtrado espacial
# Pendiente: Detección de bordes
# Pendiente: TFD-2D, densidad espectral de potencia y filtrado en frecuencia
