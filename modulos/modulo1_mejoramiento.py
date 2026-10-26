"""
Módulo 1 — Fundamentos, espacios de color y mejoramiento digital.

1.1 a 1.3 son las funciones de los notebooks de análisis. La sección "Mejoramiento del ROI
para la segmentación" es la que usa el pipeline: de ahí sale la imagen que segmenta el módulo 2.
"""
 
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from skimage.filters import gaussian, threshold_multiotsu
 
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
    # División entera para ver a qué intervalo pertenece cada valor
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
# 1.3 Detección de bordes
# ------------------------------------------------------------
 
# Kernels (dirección x, dirección y)
KERNELS = {
    "Roberts": (np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])),
    "Prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])),
}
 
 
def magnitud_gradiente(imagen, operador):
    """|∇f| = √(Gx² + Gy²) con el operador de Roberts, Prewitt o Sobel."""
    f = imagen.astype(np.float32)
    if operador == "Sobel":
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    else:
        kx, ky = KERNELS[operador]
        gx = cv2.filter2D(f, cv2.CV_32F, kx.astype(np.float32))
        gy = cv2.filter2D(f, cv2.CV_32F, ky.astype(np.float32))
    return np.hypot(gx, gy)
 
 
def laplaciano(imagen):
    """|∇²f| con el kernel de 3x3 de 4 vecinos."""
    return np.abs(cv2.Laplacian(imagen.astype(np.float32), cv2.CV_32F, ksize=1))
 
 
def canny(imagen, umbral_bajo, umbral_alto):
    """Canny completo: suavizado gaussiano (5x5, σ = 1,4) y luego cv2.Canny, que no suaviza por sí mismo."""
    suavizada = cv2.GaussianBlur(imagen, (5, 5), 1.4)
    return cv2.Canny(suavizada, umbral_bajo, umbral_alto, L2gradient=True)
 
 
def umbrales_canny(imagen, percentil=90, proporcion=0.4):
    """Umbral alto = percentil de |∇f| (Sobel) de la imagen suavizada; umbral bajo = proporción del alto."""
    suavizada = cv2.GaussianBlur(imagen, (5, 5), 1.4)
    alto = float(np.percentile(magnitud_gradiente(suavizada, "Sobel"), percentil))
    return proporcion * alto, alto
 
 
def detectar_bordes(imagen, umbral_bajo, umbral_alto):
    """Magnitud de Roberts, Sobel, Prewitt y Laplaciano, y bordes binarios de Canny."""
    mapas = {op: magnitud_gradiente(imagen, op) for op in ("Roberts", "Sobel", "Prewitt")}
    mapas["Laplaciano"] = laplaciano(imagen)
    mapas["Canny"] = canny(imagen, umbral_bajo, umbral_alto)
    return mapas
 
 
def para_mostrar(mapa, percentil=99):
    """Escala un mapa a 0–255 saturando en un percentil, para que pocos bordes fuertes no opaquen al resto."""
    tope = max(np.percentile(mapa, percentil), 1e-6)
    return np.clip(mapa / tope * 255, 0, 255).astype(np.uint8)
 
 
# ------------------------------------------------------------
# 1.3 Dominio frecuencial
# ------------------------------------------------------------
 
def espectro(imagen):
    """TFD-2D centrada (frecuencia cero en el centro)."""
    return np.fft.fftshift(np.fft.fft2(imagen.astype(np.float64)))
 
 
def reconstruir(espectro_centrado):
    """TFD-2D inversa de un espectro centrado (parte real)."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(espectro_centrado)))
 
 
def normalizar(imagen, percentiles=(0, 100)):
    """Escala lineal a 0–255 entre dos percentiles (para mostrar reconstrucciones)."""
    bajo, alto = np.percentile(imagen, percentiles)
    return np.clip((imagen - bajo) / max(alto - bajo, 1e-12) * 255, 0, 255).astype(np.uint8)
 
 
def frecuencias_radiales(forma):
    """Frecuencia radial ρ (ciclos/píxel) de cada punto de un espectro centrado."""
    alto, ancho = forma
    u = np.fft.fftshift(np.fft.fftfreq(ancho))
    v = np.fft.fftshift(np.fft.fftfreq(alto))
    U, V = np.meshgrid(u, v)
    return np.hypot(U, V)
 
 
def psd(imagen):
    """PSD = |F|² / (M·N), centrada. Se resta la media para que la componente continua no domine."""
    f = imagen.astype(np.float64)
    return np.abs(espectro(f - f.mean())) ** 2 / f.size
 
 
def perfil_radial(mapa, bins=100):
    """Promedio de un mapa centrado en anillos de igual frecuencia radial, de 0 a 0,5 ciclos/píxel."""
    rho = frecuencias_radiales(mapa.shape).ravel()
    bordes = np.linspace(0, 0.5, bins + 1)
    idx = np.digitize(rho, bordes) - 1
    ok = (idx >= 0) & (idx < bins)
    suma = np.bincount(idx[ok], weights=mapa.ravel()[ok], minlength=bins)
    cuenta = np.bincount(idx[ok], minlength=bins)
    return (bordes[:-1] + bordes[1:]) / 2, suma / np.maximum(cuenta, 1)
 
 
def filtro_pasabajos(forma, corte, tipo="gaussiano"):
    """H(u, v) centrado. Ideal: 1 si ρ ≤ corte. Gaussiano: exp(−ρ² / 2·corte²). Corte en ciclos/píxel."""
    rho = frecuencias_radiales(forma)
    if tipo == "ideal":
        return (rho <= corte).astype(np.float64)
    return np.exp(-rho ** 2 / (2 * corte ** 2))
 
 
def filtrar_frecuencia(imagen, corte, tipo="gaussiano", pasaaltos=False, relleno=32):
    """
    Filtrado en frecuencia: relleno por reflexión → TFD → H(u, v) → TFD inversa → recorte.
    El relleno evita artefactos en los bordes, porque la TFD supone la imagen periódica.
    Devuelve valores reales (con signo en el pasaaltos).
    """
    f = cv2.copyMakeBorder(imagen.astype(np.float64), relleno, relleno, relleno, relleno,
                           cv2.BORDER_REFLECT_101)
    H = filtro_pasabajos(f.shape, corte, tipo)
    if pasaaltos:
        H = 1 - H
    g = reconstruir(espectro(f) * H)
    return g[relleno:-relleno, relleno:-relleno]
 
 
def a_uint8(imagen):
    """Redondea y satura a 0–255."""
    return np.clip(np.round(imagen), 0, 255).astype(np.uint8)
 
 
def respuesta_kernel(kernel, n=512):
    """|H(f)| de un kernel espacial sobre el eje horizontal, de 0 a 0,5 ciclos/píxel."""
    H = np.abs(np.fft.fft2(kernel, s=(n, n)))
    return np.arange(n // 2 + 1) / n, H[0, :n // 2 + 1]
 
 
def sigma_gaussiano_opencv(k):
    """σ que usa OpenCV para un kernel gaussiano de K x K cuando se le pasa σ = 0."""
    return 0.3 * ((k - 1) * 0.5 - 1) + 0.8


# ------------------------------------------------------------
# Mejoramiento del ROI para la segmentación (pipeline)
# ------------------------------------------------------------
# Es la imagen que recibe el módulo 2 en cada frame. Se trabaja sobre el recorte del ROI en
# BGR (como lo lee OpenCV). `esc` es la escala del ROI (radio del iris / 32 px): los tamaños
# de los filtros se adaptan a la resolución del video.

def ventana_suavizado(esc, base=5):
    """Lado impar del kernel gaussiano: `base` px para un iris de 32 px de radio, proporcional a esc."""
    n = max(3, int(round(base * esc)))
    return n if n % 2 else n + 1


def suavizar_roi(recorte_bgr, esc=None, color=False):
    """
    Primera etapa del mejoramiento: escala de grises y suavizado gaussiano.

    Entrada: recorte del ROI (BGR), esc (None si todavía no se conoce la escala),
             color (True: también Lab suavizado, para K-means)
    Salida: dict con gris y suave_ubicacion; si se da esc, también ventana y suave (y lab_suave)
    """
    gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    mejora = {
        "gris": gris,
        # Para ubicar la pupila: σ proporcional al ROI, porque se usa antes de conocer la escala
        "suave_ubicacion": cv2.GaussianBlur(gris, (0, 0), max(1.0, 0.008 * min(alto, ancho))).astype(np.float32),
    }
    if esc is not None:
        k = ventana_suavizado(esc)
        mejora["ventana"] = k
        mejora["suave"] = cv2.GaussianBlur(gris, (k, k), 0)   # Menos ruido antes de umbralizar
        if color:
            mejora["lab_suave"] = cv2.GaussianBlur(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2LAB), (k, k), 0)
    return mejora


def realzar_pupila(mejora, esc, centro=None, radio_iris=None):
    """
    Segunda etapa: rango de interés [0, F2], como el escalado de 1.2 pero sin estirar. Lo más
    claro que F2 queda en F2, así el único borde fuerte que queda es el de la pupila.

    F2 es el umbral inferior de Otsu multinivel (pupila / iris oscuro / iris claro y piel),
    calculado en un vecindario de ±1,25 radios de iris alrededor de `centro`: en un ROI grande
    la pupila es una fracción chica de los píxeles y el histograma completo no la separa. Se
    recalcula en cada frame, así se adapta al cambio de luz del estímulo. Sin `centro` se
    usa todo el ROI.

    Entrada: mejora (de suavizar_roi, con esc), esc, centro (x, y) de la pupila en el recorte,
             radio_iris (px del recorte)
    Salida: copia de mejora con umbral (F2; None si no hay contraste), vecindario (x0, y0, x1, y1)
            y realzada (float de 0 a 1, suavizada con σ = esc)
    """
    suave = mejora["suave"]
    alto, ancho = suave.shape
    x0, y0, x1, y1 = 0, 0, ancho, alto
    if centro is not None and radio_iris:
        x = int(np.clip(round(centro[0]), 0, ancho - 1))
        y = int(np.clip(round(centro[1]), 0, alto - 1))
        m = int(1.25 * radio_iris)
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), x + m, y + m
    try:
        umbral = int(threshold_multiotsu(suave[y0:y1, x0:x1], classes=3)[0])
    except ValueError:
        umbral = None   # Vecindario casi uniforme (ojo tapado, imagen saturada): no hay pupila
    gris = mejora["gris"].astype(float)
    recortada = gris if umbral is None else np.minimum(gris, umbral)
    # El suavizado ensancha el borde, así la snake lo encuentra desde más lejos
    realzada = gaussian(recortada / 255.0, sigma=esc, preserve_range=True)
    return dict(mejora, umbral=umbral, vecindario=(x0, y0, min(x1, ancho), min(y1, alto)), realzada=realzada)


def mejorar_roi(recorte_bgr, esc, centro=None, radio_iris=None, color=False):
    """Las dos etapas juntas (grises, suavizado y rango de interés), para una imagen suelta."""
    return realzar_pupila(suavizar_roi(recorte_bgr, esc, color), esc, centro, radio_iris)


 
 
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
 
 
 
def superponer(imagen, capas):
    """Imagen en grises con máscaras coloreadas encima; `capas` = [(máscara, color), ...]."""
    salida = np.dstack([imagen] * 3) / 255
    for mascara, color in capas:
        salida[mascara] = np.array(to_rgb(color))
    return salida
 
 
def graficar_curvas(curvas, titulo, xlabel, ylabel, log_y=False, referencia=None):
    """Curvas sobre un mismo eje; `curvas` = {nombre: (x, y)}. La de `referencia` va en gris."""
    fig, eje = plt.subplots(figsize=(8, 4.5), layout="constrained")
    colores = iter(COLORES)
    for nombre, (x, y) in curvas.items():
        if nombre == referencia:
            eje.plot(x, y, color=GRIS_TEXTO, linewidth=2, linestyle="--", label=nombre)
        else:
            eje.plot(x, y, color=next(colores), linewidth=2, label=nombre)
    if log_y:
        eje.set_yscale("log")
    eje.set_xlabel(xlabel, color=GRIS_TEXTO)
    eje.set_ylabel(ylabel, color=GRIS_TEXTO)
    eje.set_title(titulo, fontweight="bold")
    eje.legend(frameon=False, fontsize=9)
    _ejes_recesivos(eje)
    eje.grid(axis="x", color=GRIS_GRILLA, linewidth=0.8)
    plt.show()


def realzada_para_ver(mejora):
    """Imagen realzada estirada a 0–255 entre su mínimo y su máximo (como escalar_rango), solo para mostrarla."""
    r = mejora["realzada"]
    bajo, alto = r.min(), r.max()
    return np.clip(np.round((r - bajo) / max(alto - bajo, 1e-12) * 255), 0, 255).astype(np.uint8)


def mostrar_mejoramiento(recorte_bgr, mejora, titulo=None, destino=None, mostrar=True):
    """
    Pasos del mejoramiento del ROI: original, grises, suavizado, rango de interés y el
    histograma del vecindario donde se calcula F2 (recuadro punteado).

    Entrada: recorte del ROI (BGR), mejora (de mejorar_roi), titulo, destino (ruta para guardar
             la figura, opcional), mostrar (False: no abre la ventana)
    Salida: nada
    """
    k, umbral = mejora["ventana"], mejora["umbral"]
    x0, y0, x1, y1 = mejora["vecindario"]
    relacion = recorte_bgr.shape[1] / recorte_bgr.shape[0]
    alto = 2.8
    fig = plt.figure(figsize=(3 * alto * relacion + 0.6, 2 * alto + 1.2), layout="constrained")
    arriba, abajo = fig.subfigures(2, 1)   # Filas independientes: el histograma no corre las imágenes
    ejes = dict(zip(["roi", "gris", "suave"], arriba.subplots(1, 3)))
    ejes.update(zip(["realzada", "hist"], abajo.subplots(1, 2, width_ratios=[1, 2])))
    paneles = [("roi", cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB), "ROI"),
               ("gris", mejora["gris"], "1. Escala de grises"),
               ("suave", mejora["suave"], f"2. Suavizado gaussiano {k} x {k} px"),
               ("realzada", realzada_para_ver(mejora),
                "3. Rango de interés: sin contraste" if umbral is None else f"3. Rango de interés [0, {umbral}]")]
    for clave, img, texto in paneles:
        ejes[clave].imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=255)
        ejes[clave].set_title(texto)
        ejes[clave].axis("off")
    ejes["suave"].add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                      edgecolor=COLORES[1], linestyle="--", linewidth=1.5))

    # Histograma del vecindario: F2 separa la pupila (niveles bajos) del resto
    eje_h = ejes["hist"]
    h = histograma(mejora["suave"][y0:y1, x0:x1])
    eje_h.bar(np.arange(256), h, width=1.0, color=COLORES[0], edgecolor="none")
    if umbral is not None:
        eje_h.axvline(umbral, color=COLORES[1], linewidth=1.5)
        eje_h.text(umbral + 4, h.max() * 0.95, f"F2 = {umbral} (Otsu multinivel)", color=COLORES[1],
                   va="top", fontsize=9)
    eje_h.set_xlim(-2, 257)
    eje_h.yaxis.set_major_formatter(FuncFormatter(lambda v, _: formato_miles(int(v))))
    eje_h.set_title("Histograma del vecindario de la pupila (suavizado)", fontsize=10, color=GRIS_TEXTO)
    eje_h.set_xlabel("Nivel de gris", color=GRIS_TEXTO)
    eje_h.set_ylabel("Píxeles", color=GRIS_TEXTO)
    _ejes_recesivos(eje_h)

    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    if destino:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=130)
    if mostrar:
        plt.show()
    else:
        plt.close(fig)