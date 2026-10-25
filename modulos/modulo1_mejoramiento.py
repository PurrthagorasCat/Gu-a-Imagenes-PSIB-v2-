"""
Módulo 1 — Fundamentos, espacios de color y mejoramiento digital.
"""
 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
 
# ------------------------------------------------------------
# 1.1 Dimensiones, resolución espacial, cuantización y color
# ------------------------------------------------------------

def formato_miles(n):
    """Formatea un entero con punto como separador de miles."""
    # Python usa coma para los miles (ej. 1,234); se reemplaza por punto
    return f"{n:,}".replace(",", ".")

 
def formato_decimal(x, decimales=1):
    """Formatea un número con coma decimal."""
    return f"{x:.{decimales}f}".replace(".", ",")
 

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
# 1.2 Transformaciones de intensidad, ecualización y ruido
# ------------------------------------------------------------
 
def a_gris(imagen):
    """RGB a escala de grises (luminancia Y = 0,299 R + 0,587 G + 0,114 B)."""
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
 
 
def histograma(imagen):
    """Cantidad de píxeles en cada nivel de gris (0 a 255)."""
    return np.bincount(imagen.ravel(), minlength=256)
 
 
def tabla_estadisticas(imagenes, nombres):
    """Media, desvío, extremos, saturación y niveles usados de cada imagen."""
    filas = []
    for img, nombre in zip(imagenes, nombres):
        filas.append({
            "Imagen": nombre,
            "Media": img.mean(),
            "Desvío": img.std(),
            "Mín": int(img.min()),
            "Máx": int(img.max()),
            "% en 0": 100 * np.mean(img == 0),
            "% en 255": 100 * np.mean(img == 255),
            "Niveles usados": len(np.unique(img)),
        })
    return pd.DataFrame(filas).set_index("Imagen").round(2)
 
 
def ajustar_brillo(imagen, constante):
    """g = f + constante, saturando en [0, 255]."""
    g = imagen.astype(np.int16) + constante      # int16 evita el desborde de uint8
    return np.clip(g, 0, 255).astype(np.uint8)
 
 
def ajustar_contraste(imagen, factor):
    """g = factor · f, saturando en [0, 255]."""
    g = np.round(imagen.astype(np.float32) * factor)
    return np.clip(g, 0, 255).astype(np.uint8)
 
 
def escalar_rango(imagen, f1, f2):
    """Lleva [f1, f2] a [0, 255]; lo menor que f1 pasa a 0 y lo mayor que f2, a 255."""
    g = (imagen.astype(np.float32) - f1) * 255 / (f2 - f1)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)
 
 
def lut_ecualizacion(imagen):
    """Función de transformación de la ecualización global: CDF normalizada a [0, 255]."""
    cdf = histograma(imagen).cumsum()
    cdf_min = cdf[cdf > 0][0]                    # CDF del primer nivel presente
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255)
    return np.clip(lut, 0, 255).astype(np.uint8)
 
 
def ecualizar(imagen):
    """Ecualización global del histograma (equivale a cv2.equalizeHist)."""
    return lut_ecualizacion(imagen)[imagen]      # cada píxel f pasa a T(f)
 
 
def ecualizar_clahe(imagen, limite=2.0, grilla=8):
    """
    Ecualización local adaptativa (CLAHE): ecualiza por bloques de una grilla
    grilla x grilla, recortando cada histograma a `limite` veces su altura media.
    """
    clahe = cv2.createCLAHE(clipLimit=limite, tileGridSize=(grilla, grilla))
    return clahe.apply(imagen)
 
 
def ruido_gaussiano(imagen, sigma, rng):
    """Ruido aditivo: g = f + n, con n ~ N(0, sigma²) en niveles de gris."""
    g = imagen + rng.normal(0, sigma, imagen.shape)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)
 
 
def ruido_sal_pimienta(imagen, densidad, rng):
    """Ruido impulsivo: una fracción `densidad` de píxeles pasa a 0 (pimienta) o a 255 (sal)."""
    g = imagen.copy()
    u = rng.random(imagen.shape)
    g[u < densidad / 2] = 0
    g[(u >= densidad / 2) & (u < densidad)] = 255
    return g
 
 
def ruido_speckle(imagen, sigma, rng):
    """Ruido multiplicativo: g = f · (1 + n), con n ~ N(0, sigma²)."""
    g = imagen * (1 + rng.normal(0, sigma, imagen.shape))
    return np.clip(np.round(g), 0, 255).astype(np.uint8)
 
 
def recorte_central(imagen, fraccion=0.4):
    """Recorte centrado que ocupa `fraccion` del ancho y del alto."""
    alto, ancho = imagen.shape[:2]
    h, w = int(alto * fraccion), int(ancho * fraccion)
    y0, x0 = (alto - h) // 2, (ancho - w) // 2
    return imagen[y0:y0 + h, x0:x0 + w]
 
 
# ------------------------------------------------------------
# Visualización
# ------------------------------------------------------------
 
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # series, siempre en este orden
GRIS_TEXTO = "#52514e"
GRIS_EJES = "#b5b4ad"
GRIS_GRILLA = "#e4e3dd"
 
 
def _ejes_recesivos(eje):
    """Ejes y grilla en gris claro, para que no compitan con los datos."""
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(GRIS_EJES)
    eje.tick_params(colors=GRIS_TEXTO, labelsize=9)
    eje.grid(axis="y", color=GRIS_GRILLA, linewidth=0.8)
    eje.set_axisbelow(True)
 
def mostrar_fila(imagenes, titulos, titulo_figura=None, cmap="gray", rango=(0, 255),
                 barra_color=False, ancho_max=18, alto_max=5):
    """
    Muestra varias imágenes una al lado de la otra.
 
    Las imágenes de un canal se dibujan con un rango fijo (por defecto 0–255).
    Si no se fija, matplotlib estira el contraste de cada imagen entre su mínimo
    y su máximo, y eso ocultaría las diferencias de niveles entre imágenes.
 
    `cmap` y `rango` pueden ser un único valor para todas o una lista con uno
    por imagen. En las imágenes RGB se ignoran.
    """
    n = len(imagenes)
    cmaps = cmap if isinstance(cmap, list) else [cmap] * n
    rangos = rango if isinstance(rango, list) else [rango] * n
 
    relacion = imagenes[0].shape[1] / imagenes[0].shape[0]   # ancho / alto
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
 
 
def mostrar_con_histogramas(imagenes, titulos, titulo_figura=None, ancho_columna=4.5):
    """Imágenes en escala de grises (arriba) con su histograma (abajo)."""
    n = len(imagenes)
    relacion = max(img.shape[1] / img.shape[0] for img in imagenes)
    alto_img = ancho_columna / relacion
    fig, ejes = plt.subplots(2, n, figsize=(ancho_columna * n, alto_img + 2.8), squeeze=False,
                             height_ratios=[alto_img, 2.2], layout="constrained")
 
    for j, (img, titulo) in enumerate(zip(imagenes, titulos)):
        eje_img, eje_h = ejes[0, j], ejes[1, j]
        eje_img.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje_img.set_title(titulo)
        eje_img.axis("off")
 
        h = histograma(img)
        eje_h.bar(np.arange(256), h, width=1.0, color=COLORES[0], edgecolor="none")
        tope = max(h[1:255].max(), 1) * 1.15     # escala sin los niveles 0 y 255
        eje_h.set_ylim(0, tope)
        eje_h.set_xlim(-2, 257)
        eje_h.yaxis.set_major_formatter(FuncFormatter(lambda v, _: formato_miles(int(v))))
 
        # Si 0 o 255 se salen de escala (saturación), se informa su porcentaje
        for nivel, x, alineacion in [(0, 4, "left"), (255, 251, "right")]:
            if h[nivel] > tope:
                pct = formato_decimal(100 * h[nivel] / img.size)
                eje_h.text(x, tope * 0.96, f"{pct} % en {nivel}", ha=alineacion, va="top",
                           fontsize=9, color=GRIS_TEXTO)
 
        eje_h.set_title(f"media {img.mean():.0f}  |  desvío {img.std():.0f}", fontsize=10, color=GRIS_TEXTO)
        eje_h.set_xlabel("Nivel de gris", color=GRIS_TEXTO)
        if j == 0:
            eje_h.set_ylabel("Píxeles", color=GRIS_TEXTO)
        _ejes_recesivos(eje_h)
 
    if titulo_figura:
        fig.suptitle(titulo_figura, fontweight="bold")
    plt.show()
 
 
def graficar_transformaciones(curvas, titulo=None):
    """Grafica funciones de transformación g = T(f); `curvas` = {nombre: T evaluada en 0…255}."""
    f = np.arange(256)
    fig, eje = plt.subplots(figsize=(5, 5), layout="constrained")
    eje.plot(f, f, color=GRIS_EJES, linestyle="--", linewidth=1.2, label="Identidad")
    for (nombre, t), color in zip(curvas.items(), COLORES):
        eje.plot(f, t, color=color, linewidth=2, label=nombre)
 
    eje.set_xlim(0, 255)
    eje.set_ylim(-3, 258)
    eje.set_aspect("equal")
    eje.set_xlabel("Nivel de entrada f", color=GRIS_TEXTO)
    eje.set_ylabel("Nivel de salida g = T(f)", color=GRIS_TEXTO)
    eje.legend(frameon=False, loc="lower right", fontsize=9)
    _ejes_recesivos(eje)
    eje.grid(axis="x", color=GRIS_GRILLA, linewidth=0.8)
    if titulo:
        eje.set_title(titulo, fontweight="bold")
    plt.show()
 
 
# Pendiente: Detección de bordes
# Pendiente: TFD-2D, densidad espectral de potencia y filtrado en frecuencia