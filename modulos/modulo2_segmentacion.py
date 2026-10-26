"""
Módulo 2 — Segmentación y morfología matemática de la pupila (todo en un archivo).

Objetivo: aislar de forma robusta la pupila en un frame de trabajo y extender el
algoritmo a todo el video.
  2.1 Segmentación: Otsu, Otsu multinivel, K-means (K=2..5), Snake, Watershed.
  2.2 Morfología:   erosión, dilatación, apertura, cierre y combinaciones,
                    esqueleto (eje medio y contorno), métricas Dice / IoU contra
                    una referencia anotada a mano.
  Video:            seguimiento de la pupila frame a frame (Snake + morfología +
                    control de calidad) y curva del diámetro en el tiempo.

No se usa ningún algoritmo de detección automática del ojo ni modelos
entrenados: el ROI se marca A MANO al comienzo y TODO el análisis (video,
frame de trabajo, métodos 2.1 y 2.2, referencias) se hace dentro de ese ROI.

Este módulo no mejora la imagen: segmenta la que sale del módulo 1
(modulo1_mejoramiento: escala de grises, suavizado y rango de interés).
En el video, cada frame pasa primero por el módulo 1 y después por la
segmentación.

Carpetas (relativas a la raíz del proyecto):
  videos/                        -> videos de entrada
  anotaciones_manuales/          -> roi.json (ROI marcados) y referencias.json
                                    (contornos de la pupila anotados a mano)
  resultados/resultados_modulo2/
      resultados_video/          -> serie temporal (CSV + gráfico), video con el
                                    contorno superpuesto, frames extraídos
      resultados_imagen/         -> figuras y métricas de 2.1 y 2.2 sobre el
                                    frame de trabajo

Uso (desde la raíz del proyecto o con el botón Run):
  python modulos/modulo2_segmentacion.py                       # video más reciente de videos/
  python modulos/modulo2_segmentacion.py registro_A1.MOV       # un video de videos/
  python modulos/modulo2_segmentacion.py --roi X Y ANCHO ALTO  # ROI sin ventana
  python modulos/modulo2_segmentacion.py --usar-roi-guardado   # reutilizar el ROI de roi.json
  python modulos/modulo2_segmentacion.py --conservar-resultados
  python modulos/modulo2_segmentacion.py --frame dilatada contraida   # frames para 2.1/2.2
  python modulos/modulo2_segmentacion.py --factor 1            # resolución original (más lento)
  python modulos/modulo2_segmentacion.py --imagen ojo.png      # 2.1/2.2 sobre una imagen suelta
  python modulos/modulo2_segmentacion.py --anotar resultados/resultados_modulo2/resultados_video/frames/registro_A1_frame_0123.png
Desde otro script (p.ej. pipeline_completo.py):
  from modulos import modulo2_segmentacion as m2
  filas, r_iris = m2.ejecutar_modulo2("videos/registro_A1.MOV")
"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt   # las figuras se guardan en archivos y se cierran (no se muestran)
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion as _be, binary_dilation as _bd
from scipy.signal import medfilt
from skimage.draw import polygon2mask
from skimage.filters import threshold_otsu, threshold_multiotsu, sobel
from skimage.measure import find_contours
from skimage.morphology import disk, skeletonize
from skimage.segmentation import active_contour, watershed
from sklearn.cluster import KMeans

try:
    from . import modulo1_mejoramiento as m1
except ImportError:   # ejecutado como script: python modulos/modulo2_segmentacion.py
    import modulo1_mejoramiento as m1


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
RAIZ = Path(__file__).resolve().parent.parent          # raíz del proyecto
CARPETA_VIDEOS = RAIZ / "videos"
CARPETA_ANOTACIONES = RAIZ / "anotaciones_manuales"
CARPETA_RESULTADOS = RAIZ / "resultados" / "resultados_modulo2"
OUT_VIDEO = CARPETA_RESULTADOS / "resultados_video"
OUT_IMAGEN = CARPETA_RESULTADOS / "resultados_imagen"
ARCHIVO_ROI = CARPETA_ANOTACIONES / "roi.json"
ARCHIVO_REFERENCIAS = CARPETA_ANOTACIONES / "referencias.json"
ARCHIVO_FRAMES = CARPETA_ANOTACIONES / "frames_referencia.json"   # frame sobre el que se marcó cada ROI

NOMBRE_VIDEO = None       # p.ej. "registro_A1.MOV"; None = el más reciente de videos/
BORRAR_RESULTADOS = True  # borrar resultados/resultados_modulo2 antes de empezar (--conservar-resultados)
PEDIR_ROI_SIEMPRE = True  # marcar el ROI en cada corrida (--usar-roi-guardado para reutilizar roi.json)
ALTO_OVERLAY = 480        # alto del video overlay (sólo para visualizar)

R_REF_PARAM = 32          # radio de iris (px) para el que se ajustaron los parámetros
LADO_SOBRE_IRIS = 2.5     # lado menor del ROI ≈ 2.5 radios de iris (iris + margen)
R_MAX_PROCESO = 64        # radio de iris máximo (px) con el que se procesa (ver factor_auto)

# Factor de escala de los parámetros (tamaños de filtros y elementos
# estructurantes) para el recorte actual; se fija a partir del ROI y de la pupila.
ESC = 1.0

# Todos los tiempos se expresan en SEGUNDOS y se convierten a frames con los fps
# del video: así el comportamiento es el mismo a 30, 60 o 120 fps.
T_MEDIANA = 0.23          # ventana del filtro de mediana de la serie
T_ATIPICOS = 0.30         # ventana de la mediana móvil para detectar atípicos
T_ESTABLE = 0.10          # el frame de trabajo no puede tener frames inválidos a ±T_ESTABLE


def a_frames(t, fps, impar=False, minimo=1):
    """Convierte un tiempo en segundos a cantidad de frames."""
    n = max(minimo, int(round(t * fps)))
    return n + 1 if impar and n % 2 == 0 else n


def _ruta(path, carpeta=None):
    """Ruta absoluta: tal cual si existe; si no, relativa a `carpeta` o a la raíz."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path.resolve()
    for base in ([carpeta] if carpeta else []) + [RAIZ]:
        if (base / path).exists():
            return base / path
    return (carpeta or RAIZ) / path


# Operaciones morfológicas binarias (sobre scipy)
def binary_erosion(m, se):  return _be(m, se)
def binary_dilation(m, se): return _bd(m, se)
def binary_opening(m, se):  return binary_dilation(binary_erosion(m, se), se)
def binary_closing(m, se):  return binary_erosion(binary_dilation(m, se), se)


# ============================================================================
# ROI MANUAL
# ============================================================================
@dataclass
class ROI:
    x0: int               # esquina superior izquierda (imagen original)
    y0: int
    w: int                # ancho y alto (px)
    h: int
    origen: str = "manual"

    @property
    def r_ref(self):
        """Radio de iris de referencia, deducido del tamaño del ROI."""
        return min(self.w, self.h) / LADO_SOBRE_IRIS

    @property
    def esc(self):
        """Factor para escalar los parámetros (tamaños en px) de la segmentación."""
        return self.r_ref / R_REF_PARAM


def px(valor, esc, minimo=1):
    """Tamaño en px (ajustado para R_REF_PARAM) -> tamaño en px para este ROI."""
    return max(minimo, int(round(valor * esc)))


def impar(valor, esc, minimo=3):
    n = px(valor, esc, minimo)
    return n if n % 2 else n + 1


# ----------------------------------------------------------------------------
# Obtener el ROI
# ----------------------------------------------------------------------------
def seleccionar_roi(bgr, titulo="ROI: encerrar el IRIS con margen y apretar Enter (Esc = cancelar)"):
    H, W = bgr.shape[:2]
    f = min(1.0, 900 / max(H, W))                 # que la ventana entre en pantalla
    vista = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else bgr
    print("  >>> Se abrió una ventana para marcar el ROI (si no la ves, puede haber quedado "
          "detrás de otras ventanas). Encerrá el iris con margen y apretá Enter.", flush=True)
    try:
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(titulo, cv2.WND_PROP_TOPMOST, 1)    # adelante de todo
        x, y, w, h = cv2.selectROI(titulo, vista, showCrosshair=True)
        cv2.destroyWindow(titulo)
    except cv2.error:
        raise RuntimeError("No hay entorno gráfico para seleccionar el ROI: "
                           "indicalo con --roi X Y ANCHO ALTO")
    if w == 0 or h == 0:
        raise RuntimeError("Selección del ROI cancelada")
    return ROI(int(round(x / f)), int(round(y / f)), int(round(w / f)), int(round(h / f)))


def _leer_roi_json():
    if ARCHIVO_ROI.exists():
        return json.loads(ARCHIVO_ROI.read_text(encoding="utf-8"))
    return {}


def guardar_roi(nombre, roi):
    CARPETA_ANOTACIONES.mkdir(parents=True, exist_ok=True)
    datos = _leer_roi_json()
    datos[nombre] = [roi.x0, roi.y0, roi.w, roi.h]
    ARCHIVO_ROI.write_text(json.dumps(datos, indent=2), encoding="utf-8")


def _a_roi(v, origen):
    if len(v) == 3:                     # formato viejo: círculo del iris (x, y, r)
        x, y, r = v
        lado = LADO_SOBRE_IRIS * r
        return ROI(int(round(x - lado / 2)), int(round(y - lado / 2)),
                   int(round(lado)), int(round(lado)), origen)
    return ROI(*[int(round(t)) for t in v], origen=origen)


def guardar_frame_referencia(nombre, indice):
    """Frame del video sobre el que se marcó el ROI. Se usa también para la
    escala y para las figuras del mejoramiento (por si el primero no sirve,
    p.ej. porque el video empieza con un parpadeo)."""
    CARPETA_ANOTACIONES.mkdir(parents=True, exist_ok=True)
    datos = json.loads(ARCHIVO_FRAMES.read_text(encoding="utf-8")) if ARCHIVO_FRAMES.exists() else {}
    datos[nombre] = int(indice)
    ARCHIVO_FRAMES.write_text(json.dumps(datos, indent=2), encoding="utf-8")


def frame_referencia_guardado(nombre):
    """Frame de referencia guardado para el video `nombre`, o None."""
    if not ARCHIVO_FRAMES.exists():
        return None
    return json.loads(ARCHIVO_FRAMES.read_text(encoding="utf-8")).get(nombre)


def roi_guardado(nombre):
    """Busca el ROI por nombre de archivo. Un frame extraído de un video
    ("video_frame_0123.png") usa el ROI de su video ("video.mp4")."""
    datos = _leer_roi_json()
    if nombre in datos:
        return _a_roi(datos[nombre], "roi.json")
    if "_frame_" in nombre:
        madre = nombre.split("_frame_")[0]
        for k, v in datos.items():
            if Path(k).stem == madre:
                return _a_roi(v, f"roi.json ({k})")
    return None


def obtener_roi(bgr, nombre, rect=None, nuevo=False):
    """ROI de la imagen/video `nombre`: rect=(x, y, w, h) -> ese; si no, el
    guardado en roi.json (salvo nuevo=True); si no hay -> ventana (y se guarda)."""
    if rect is not None:
        roi = ROI(*[int(round(v)) for v in rect], origen="--roi")
        guardar_roi(nombre, roi)
        return roi
    if not nuevo:
        roi = roi_guardado(nombre)
        if roi is not None:
            return roi
    roi = seleccionar_roi(bgr)
    guardar_roi(nombre, roi)
    return roi


# ----------------------------------------------------------------------------
# Recorte (sin reescalar) y cambio de coordenadas
# ----------------------------------------------------------------------------
def factor_auto(roi, r_iris=None):
    """Factor de reducción por defecto: se procesa a resolución original salvo
    que el iris quede de más de R_MAX_PROCESO px de radio (p.ej. video 4K);
    en ese caso se reduce lo justo. Medido en el video 4K: con el iris a 64 px
    cada frame se procesa ~8 veces más rápido y el diámetro cambia < 1 %."""
    return min(1.0, R_MAX_PROCESO / (r_iris or roi.r_ref))


def recortar(bgr, roi, factor=1.0, r_iris=None):
    """Recorte EXACTO del ROI. factor = 1: resolución original; factor < 1: el
    recorte se reduce (INTER_AREA, que promedia y no genera aliasing) para
    procesar más rápido. Si el ROI sale del borde de la imagen, lo que falta se
    completa replicando el borde.
    r_iris: radio de iris estimado (px de la imagen original) a partir de la
    pupila medida; si no se da, se deduce del tamaño del ROI (menos robusto)."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = roi.x0, roi.y0, roi.x0 + roi.w, roi.y0 + roi.h
    if x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H:
        crop = bgr[y0:y1, x0:x1].copy()
    else:
        pad = max(0, -x0, -y0, x1 - W, y1 - H)
        big = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop = big[y0 + pad:y1 + pad, x0 + pad:x1 + pad].copy()
    if factor < 1:
        tam = (max(1, int(round(roi.w * factor))), max(1, int(round(roi.h * factor))))
        crop = cv2.resize(crop, tam, interpolation=cv2.INTER_AREA)
        factor = tam[0] / roi.w
    M = np.float32([[factor, 0, -factor * x0], [0, factor, -factor * y0]])
    r_ref = (r_iris or roi.r_ref) * factor
    info = dict(roi=roi, escala=factor, M=M, esc=r_ref / R_REF_PARAM, r_ref=r_ref,
                x0=x0, y0=y0, lado=max(roi.w, roi.h))
    return crop, info


def a_recorte(pts, info):
    """Puntos (x, y) de la imagen original -> coordenadas del recorte."""
    p = np.asarray(pts, np.float64)
    return p * info["escala"] + info["M"][:, 2]


def a_original(pts, info):
    """Puntos (x, y) del recorte -> coordenadas de la imagen original."""
    p = np.asarray(pts, np.float64)
    return (p - info["M"][:, 2]) / info["escala"]


# ============================================================================
# 2.1 y 2.2 SOBRE UNA IMAGEN (frame de trabajo)
# ============================================================================
# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def leer_bgr(path):
    path = _ruta(path)
    # cv2.imread falla en Windows con rutas no ASCII (p.ej. "Señales"):
    # se lee el archivo como bytes con numpy y se decodifica con OpenCV
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR) if path.exists() else None
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr


# ----------------------------------------------------------------------------
# ROI manual del ojo (paso previo) -> ver la sección ROI MANUAL
# ----------------------------------------------------------------------------
def figura_roi(bgr, info, crop, destino):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    grosor = max(1, int(round(max(bgr.shape[:2]) / 400)))
    r = info["roi"]
    cv2.rectangle(rgb, (r.x0, r.y0), (r.x0 + r.w, r.y0 + r.h), (0, 255, 0), 2 * grosor)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].set_title(f"ROI manual (verde): x={r.x0}, y={r.y0}, {r.w}x{r.h} px ({r.origen})", fontsize=8)
    ax[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax[1].set_title(f"Recorte del ROI ({crop.shape[1]}x{crop.shape[0]} px, "
                    f"{100 * info['escala']:.0f} % de la resolución original)")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(destino, dpi=120); plt.close(fig)


def cargar(path, roi_fijo=None, rect=None, nuevo_roi=False, factor=None, r_iris=None):
    """Lee la imagen completa y la recorta EXACTAMENTE con el ROI manual (sin
    reescalar). Si se pasa `roi_fijo` (p.ej. el ROI del video, desde
    procesar_video) se usa ese; si no, el de --roi, el de roi.json o se
    pide marcarlo con el mouse."""
    bgr_full = leer_bgr(path)
    if roi_fijo is None:
        roi_fijo = obtener_roi(bgr_full, Path(path).name, rect=rect, nuevo=nuevo_roi)
    print(f"{Path(path).name}: ROI x={roi_fijo.x0} y={roi_fijo.y0} {roi_fijo.w}x{roi_fijo.h} px "
          f"({roi_fijo.origen})")
    if r_iris is None:
        # escala a partir de la PUPILA (no del tamaño del ROI)
        f_rap = min(1.0, 600 / max(roi_fijo.w, roi_fijo.h))
        c0, _ = recortar(bgr_full, roi_fijo, f_rap)
        r_iris = estimar_pupila(m1.suavizar_roi(c0)["suave_ubicacion"])[2] / f_rap / PUPILA_SOBRE_IRIS
        if r_iris <= 0:
            r_iris = roi_fijo.r_ref
    if factor is None:
        factor = factor_auto(roi_fijo, r_iris)
    bgr, info = recortar(bgr_full, roi_fijo, factor, r_iris)
    figura_roi(bgr_full, info, bgr, OUT_IMAGEN / f"{Path(path).stem}_00_roi.png")
    cv2.imencode(".png", bgr)[1].tofile(str(OUT_IMAGEN / f"{Path(path).stem}_roi.png"))
    return bgr, info


def estimar_pupila(suave, cerca_de=None, radio=None):
    """Estimación simple (sin parámetros de escala) de la pupila en el ROI:
    punto más oscuro de la imagen suavizada (dentro de la elipse inscripta en
    el ROI), umbral a mitad de camino entre ese punto y la mediana del ROI, y
    mayor círculo inscripto en la zona oscura que contiene al punto (así no la
    agrandan las pestañas o sombras finas pegadas). Devuelve (x, y, r).
    suave: la imagen "suave_ubicacion" del módulo 1 (m1.suavizar_roi)."""
    g = suave
    h, w = g.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    elipse = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 <= 1
    if cerca_de is not None and radio:       # en video: cerca de la última posición válida
        elipse &= np.hypot(xx - cerca_de[0], yy - cerca_de[1]) <= radio
        if not elipse.any():
            elipse = np.ones_like(elipse)
    gm = np.where(elipse, g, 1e9)
    _, _, (x, y), _ = cv2.minMaxLoc(gm)
    vmin = float(g[y, x])
    t = vmin + 0.4 * (float(np.median(g[elipse])) - vmin)
    oscuro = ndi.binary_fill_holes(componente_con_semilla(g <= t, (x, y)))
    return circulo_inscripto(oscuro)


# Cociente típico pupila/iris (radios) usado para pasar del tamaño de la
# pupila medida al tamaño de iris con el que se escalan los parámetros. Sólo
# importa el orden de magnitud: un error de ±50 % no cambia los resultados.
PUPILA_SOBRE_IRIS = 0.45


def circulo_inscripto(mask):
    """(x, y, r) del mayor círculo inscripto en la máscara."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, r, _, (x, y) = cv2.minMaxLoc(dist)
    return float(x), float(y), float(r)


def region_analisis(shape, semilla):
    """Región de análisis = el ROI completo (el recorte ya es el ROI)."""
    h, w = shape[:2]
    return (slice(0, h), slice(0, w)), semilla


def componente_con_semilla(mask, semilla):
    """Se queda con la componente conexa que contiene a la semilla
    (o la más cercana si la semilla cayó en un hueco, p.ej. un reflejo)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    x, y = semilla
    l = lab[y, x]
    if l == 0:
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        l = lab[iy[y, x], ix[y, x]]
    return lab == l


def pegar(mask_roi, sl, shape):
    full = np.zeros(shape[:2], bool)
    full[sl] = mask_roi
    return full


# ----------------------------------------------------------------------------
# 2.1 Métodos de segmentación
# ----------------------------------------------------------------------------
def seg_otsu(suave, sl, s_roi):
    """suave: gris suavizado por el módulo 1 (mejora["suave"])."""
    g = suave[sl]
    # (a) Otsu: umbral óptimo global sobre todo el ROI (máscara tal cual, sin
    #     post-procesar: incluye pestañas, sombras, etc. -> material para 2.2)
    t = threshold_otsu(g)
    m = g < t
    # (b) Otsu multinivel (3 clases: pupila / iris oscuro+párpado / iris claro+piel)
    #     + componente conexa de la semilla
    t_multi = threshold_multiotsu(g, classes=3)[0]
    m_multi = componente_con_semilla(g < t_multi, s_roi)
    return m, t, m_multi, t_multi


def seg_kmeans(lab_suave, sl, s_roi, K, w_croma=0.5):
    """Espacio de características: L (intensidad) + a,b (croma), ponderando el croma.
    lab_suave: Lab suavizado por el módulo 1 (mejora["lab_suave"])."""
    r = lab_suave[sl].astype(np.float32)
    feats = r.reshape(-1, 3).copy()
    feats[:, 1:] *= w_croma
    # en ROIs grandes (resolución original) se ajusta con una muestra de
    # píxeles y después se clasifican TODOS: mismo resultado, mucho más rápido
    rng = np.random.default_rng(0)
    muestra = feats[rng.choice(len(feats), min(len(feats), 30000), replace=False)]
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(muestra)
    etiquetas = km.predict(feats).reshape(r.shape[:2])
    c_pupila = np.argmin(km.cluster_centers_[:, 0])      # cluster con menor L
    m = componente_con_semilla(etiquetas == c_pupila, s_roi)
    return m, etiquetas


def seg_snake(realzada, sl, centro, r0, n=200, init=None):
    """Contorno activo inicializado como círculo que rodea a la pupila.
    realzada: imagen del módulo 1 con el rango de interés [0, F2] (mejora["realzada"]):
    todo lo más claro que la pupila queda "plano", así el único borde fuerte es
    la frontera pupila/iris y la snake no es atraída por el párpado o el limbo.
    alpha: elasticidad (tiende a encoger), beta: rigidez (suavidad),
    w_edge: atracción a bordes, gamma: paso temporal.
    init: contorno inicial (fila, col) dado (p.ej. la pupila del video); si no,
    un círculo de radio r0 alrededor de `centro`."""
    img = realzada[sl]
    if init is None:
        t = np.linspace(0, 2 * np.pi, n)
        cx, cy = centro
        init = np.c_[cy + r0 * np.sin(t), cx + r0 * np.cos(t)]  # (fila, col)
    # las fuerzas internas (elasticidad alpha, rigidez beta) dependen de la
    # escala: en resolución original se dividen por ESC² para que la snake no
    # colapse antes de llegar al borde
    snake = active_contour(img, init, alpha=0.1 / ESC ** 2, beta=1.0 / ESC ** 2, gamma=0.01,
                           w_line=0, w_edge=1, max_num_iter=2500)
    m = polygon2mask(img.shape, snake)
    return m, init, snake


def seg_watershed(suave, sl, s_roi, m_semilla):
    """Watershed controlado por marcadores sobre el gradiente.
    Marcadores: 1 = núcleo de pupila, 2 = fondo (iris/esclerótica),
    3 = reflejos corneales (muy brillantes).
    suave: gris suavizado por el módulo 1 (mejora["suave"])."""
    g = suave[sl] / 255.0
    grad = sobel(g)
    # el núcleo se toma de Otsu + apertura (corta los "puentes" hacia las
    # pestañas/pliegue del párpado) -> marcador confiable dentro de la pupila
    nucleo = ndi.binary_fill_holes(m_semilla)
    nucleo = componente_con_semilla(binary_opening(nucleo, disk(px(3, ESC))), s_roi)
    marc = np.zeros(g.shape, np.int32)
    marc[~binary_dilation(nucleo, disk(px(4, ESC)))] = 2
    brillo = g > np.percentile(g, 99)                     # destellos
    marc[brillo] = 3
    marc[binary_erosion(nucleo, disk(px(2, ESC))) & ~brillo] = 1
    etiquetas = watershed(grad, marc)
    return etiquetas == 1, etiquetas, grad, marc


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def dice(a, b):
    i = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 2 * i / s if s else 1.0


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def cargar_referencia(nombre, shape, info_roi):
    """Las referencias se guardan en coordenadas de la IMAGEN ORIGINAL (así no
    dependen de cómo se recorte) y se llevan al recorte con su transformación."""
    p = ARCHIVO_REFERENCIAS
    if not p.exists():
        return None
    datos = json.loads(p.read_text(encoding="utf-8"))
    if nombre not in datos:
        return None
    pts = a_recorte(datos[nombre], info_roi).astype(np.float32)
    if len(pts) < 15:      # pocos puntos -> ajusta elipse (pupila ≈ elipse)
        (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(pts)
        m = np.zeros(shape[:2], np.uint8)
        cv2.ellipse(m, ((cx, cy), (ax1, ax2), ang), 1, -1)
        return m.astype(bool)
    return polygon2mask(shape[:2], pts[:, ::-1])          # (fila, col)


# ----------------------------------------------------------------------------
# 2.2 Morfología
# ----------------------------------------------------------------------------
def mayor_componente(m):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    tam = ndi.sum(m, lab, range(1, n + 1))
    return lab == (np.argmax(tam) + 1)


def variantes_morfologicas(m, r, semilla):
    se = disk(r)
    v = {
        "original": m,
        "erosion": binary_erosion(m, se),
        "dilatacion": binary_dilation(m, se),
        "apertura": binary_opening(m, se),
        "cierre": binary_closing(m, se),
        "cierre+apertura": binary_opening(binary_closing(m, se), se),
        "apertura+cierre": binary_closing(binary_opening(m, se), se),
    }
    # Combinación propuesta:
    #   1) apertura  -> corta puentes finos hacia pestañas / pliegue del párpado
    #   2) CC con la semilla -> descarta lo que quedó desconectado
    #   3) cierre    -> tapa muescas en el borde causadas por reflejos
    #   4) relleno de huecos -> reflejos totalmente dentro de la pupila
    x = componente_con_semilla(binary_opening(m, se), semilla)
    v["apertura+CC+cierre+relleno"] = ndi.binary_fill_holes(binary_closing(x, se))
    return v


def esqueletos(m):
    medial = skeletonize(m)                                 # eje medio de la región
    borde = m & ~binary_erosion(m, disk(1))                 # contorno
    contorno = skeletonize(borde)                           # contorno de 1 px
    return medial, contorno


def analizar_morfologia(rgb_r, m, ref_r, nombre, stem, metodo, semilla):
    filas = []
    sc = lambda x: (dice(x, ref_r), iou(x, ref_r)) if ref_r is not None else (np.nan, np.nan)
    # barrido del radio del elemento estructurante (equivalente a 1..4 px para
    # un iris de 32 px, escalado al tamaño real del ROI)
    radios = sorted({px(r, ESC) for r in range(1, 5)})
    for r in radios:
        for k, x in variantes_morfologicas(m, r, semilla).items():
            if k != "original" or r == radios[0]:
                filas.append((nombre, f"2.2 [{metodo}] r={r}px", k, *sc(x)))
    r_opt = px(2, ESC)
    if ref_r is not None:
        r_opt = max(radios, key=lambda r: dice(variantes_morfologicas(m, r, semilla)["apertura+CC+cierre+relleno"], ref_r))
    var = variantes_morfologicas(m, r_opt, semilla)
    final = var["apertura+CC+cierre+relleno"]
    medial, contorno = esqueletos(final)

    fig, ax = plt.subplots(2, 5, figsize=(15, 6.5))
    ax = ax.ravel()
    for a, (k, x) in zip(ax, var.items()):
        a.imshow(x, cmap="gray"); a.set_title(f"{k}\nDice={sc(x)[0]:.3f}", fontsize=8)
    ax[8].imshow(final, cmap="gray"); ax[8].imshow(np.ma.masked_where(~medial, medial), cmap="autumn")
    ax[8].set_title("Esqueleto (eje medio)", fontsize=8)
    ax[9].imshow(rgb_r); ax[9].imshow(np.ma.masked_where(~contorno, contorno), cmap="autumn")
    ax[9].set_title("Esqueleto del contorno", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{nombre}: morfología sobre '{metodo}' (EE = disco r={r_opt} px)")
    fig.tight_layout(); fig.subplots_adjust(hspace=0.4)
    tag = metodo.split(" (")[0].replace(" ", "_").replace("=", "")
    fig.savefig(OUT_IMAGEN / f"{stem}_22_morfologia_{tag}.png", dpi=130); plt.close(fig)
    return filas


# ----------------------------------------------------------------------------
# Pipeline por imagen
# ----------------------------------------------------------------------------
def overlay(rgb, mask, color=(255, 0, 0), ref=None):
    o = rgb.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    g = px(1, ESC / 1.5)
    cv2.drawContours(o, cnts, -1, color, g)
    if ref is not None:
        cnts, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cnts, -1, (0, 255, 0), g)
    return o


def procesar(path, roi_fijo=None, rect=None, nuevo_roi=False, factor=None, r_iris=None,
             pupila_video=None):
    """pupila_video: contorno (x, y) de la pupila en coords. de la imagen
    original, encontrado por procesar_video en este frame. Si se da, se usa
    como punto de partida (semilla, contorno inicial de la Snake y marcador del
    Watershed); los métodos se siguen aplicando igual sobre todo el ROI."""
    global ESC
    nombre = Path(path).name
    stem = Path(path).stem
    bgr, info_roi = cargar(path, roi_fijo, rect, nuevo_roi, factor, r_iris)
    ESC = info_roi["esc"]
    # Módulo 1: imagen mejorada (rango de interés calculado en todo el ROI)
    mejora = m1.mejorar_roi(bgr, ESC, color=True)
    m1.mostrar_mejoramiento(bgr, mejora, titulo=f"{nombre}: mejoramiento (módulo 1)",
                            destino=OUT_IMAGEN / f"{stem}_20_mejoramiento.png", mostrar=False)
    gris = mejora["gris"]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # solo para las figuras
    pupila_v = None
    if pupila_video is not None and len(pupila_video) >= 5:
        xy = a_recorte(pupila_video, info_roi)                       # (x, y) del recorte
        pupila_v = polygon2mask(gris.shape, xy[:, ::-1])
    if pupila_v is not None and pupila_v.any():
        cyv, cxv = ndi.center_of_mass(pupila_v)
        semilla = (int(round(cxv)), int(round(cyv)))
        print(f"  punto de partida: pupila encontrada por el video en este frame "
              f"(radio ≈ {np.sqrt(pupila_v.sum() / np.pi):.1f} px del recorte)")
    else:
        # imagen suelta: punto de partida = estimación simple de la pupila
        # (zona oscura "gruesa" del ROI), como un disco
        x, y, r = estimar_pupila(mejora["suave_ubicacion"])
        semilla = (int(round(x)), int(round(y)))
        yy, xx = np.mgrid[0:gris.shape[0], 0:gris.shape[1]]
        pupila_v = np.hypot(xx - x, yy - y) <= max(r, 2)
        print(f"  punto de partida: estimación simple de la pupila (radio ≈ {r:.1f} px del recorte)")
    sl, s_roi = region_analisis(gris.shape, semilla)
    ref = cargar_referencia(nombre, gris.shape, info_roi)
    ref_r = ref[sl] if ref is not None else None
    rgb_r = rgb[sl]

    # ---- 2.1 -------------------------------------------------------------
    m_otsu, t_roi, m_multi, t_multi = seg_otsu(mejora["suave"], sl, s_roi)
    # estimación gruesa de centro/radio de la pupila para inicializar la snake:
    # centro y radio del mayor círculo inscripto en la zona oscura (máximo de la
    # transformada de distancia). No le afectan las pestañas o la sombra del
    # párpado "pegadas" a la pupila, porque son finas.
    if pupila_v is not None:
        cx0, cy0, r_est = circulo_inscripto(pupila_v)
    else:
        cx0, cy0, r_est = circulo_inscripto(ndi.binary_fill_holes(m_multi))
    km = {K: seg_kmeans(mejora["lab_suave"], sl, s_roi, K) for K in range(2, 6)}
    init_v = None
    if pupila_v is not None:
        # contorno de partida agrandado un 25 % (igual que en el seguimiento del video)
        cs = find_contours(pupila_v.astype(float), 0.5)
        c = max(cs, key=len)
        c = c[np.linspace(0, len(c) - 1, 200, endpoint=False).astype(int)]
        cen = c.mean(0)
        init_v = cen + 1.25 * (c - cen)
    m_snake, init, snake = seg_snake(mejora["realzada"], sl, (cx0, cy0), 1.2 * r_est, init=init_v)
    # marcador de pupila del Watershed: el punto de partida (pupila del video o estimación)
    m_ws, et_ws, grad, marc = seg_watershed(mejora["suave"], sl, s_roi,
                                            pupila_v if pupila_v is not None else m_multi)

    metodos = {f"Otsu (T={t_roi:.0f})": m_otsu,
               f"Otsu multinivel (T={t_multi:.0f})": m_multi}
    for K, (m, _) in km.items():
        metodos[f"K-means K={K}"] = m
    metodos["Snake"] = m_snake
    metodos["Watershed"] = m_ws

    filas = []
    for k, m in metodos.items():
        d = dice(m, ref_r) if ref_r is not None else np.nan
        j = iou(m, ref_r) if ref_r is not None else np.nan
        filas.append((nombre, "2.1", k, d, j))

    # Figura 2.1
    n = len(metodos) + 1
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6.5))
    ax = ax.ravel()
    ax[0].imshow(rgb_r); ax[0].set_title("ROI (verde = ref.)", fontsize=8)
    if ref_r is not None:
        ax[0].imshow(overlay(rgb_r, np.zeros_like(ref_r), ref=ref_r))
    for a, (k, m), f in zip(ax[1:], metodos.items(), filas):
        a.imshow(overlay(rgb_r, m, ref=ref_r))
        a.set_title(f"{k}\nDice={f[3]:.2f} IoU={f[4]:.2f}", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{nombre}: métodos de segmentación sobre el ROI {gris.shape[1]}x{gris.shape[0]} px "
                 "(rojo = método, verde = referencia)")
    fig.tight_layout(); fig.subplots_adjust(hspace=0.4); fig.savefig(OUT_IMAGEN / f"{stem}_21_metodos.png", dpi=130); plt.close(fig)

    # Figura de diagnóstico: Otsu global, clusters, snake, watershed
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    ax[0, 0].imshow(m_otsu, cmap="gray"); ax[0, 0].set_title(f"Otsu en el ROI (T={t_roi:.0f})")
    ax[0, 1].hist(mejora["suave"][sl].ravel(), 64, color="gray"); ax[0, 1].axvline(t_roi, color="r")
    ax[0, 1].set_title("Histograma ROI suavizado + umbral Otsu")
    for i, K in enumerate([3, 5]):
        ax[0, 2 + i].imshow(km[K][1], cmap="tab10"); ax[0, 2 + i].set_title(f"K-means K={K} (clusters)")
    ax[1, 0].imshow(mejora["realzada"][sl], cmap="gray")
    ax[1, 0].plot(init[:, 1], init[:, 0], "--b", lw=1, label="inicial")
    ax[1, 0].plot(snake[:, 1], snake[:, 0], "-r", lw=1.5, label="final")
    ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("Snake (imagen realzada, módulo 1)")
    ax[1, 1].imshow(grad, cmap="magma"); ax[1, 1].set_title("Gradiente (Sobel)")
    ax[1, 2].imshow(marc, cmap="tab10", vmin=0, vmax=9); ax[1, 2].set_title("Marcadores 1=pupila 2=fondo 3=reflejo")
    ax[1, 3].imshow(et_ws, cmap="tab10", vmin=0, vmax=9); ax[1, 3].set_title("Watershed (cuencas)")
    for a in ax.ravel():
        if a is not ax[0, 1]:
            a.axis("off")
    fig.tight_layout(); fig.subplots_adjust(hspace=0.4); fig.savefig(OUT_IMAGEN / f"{stem}_21_diagnostico.png", dpi=130); plt.close(fig)

    # ---- 2.2 -------------------------------------------------------------
    # Se aplica a (a) el mejor método por píxel (Otsu/K-means/Watershed), donde la
    # morfología tiene más trabajo (puentes a pestañas, huecos por reflejos), y
    # (b) la Snake, que ya sale regularizada por sus fuerzas internas.
    candidatos = dict(metodos)
    por_pixel = {k: v for k, v in candidatos.items() if k != "Snake"}
    puntaje = (lambda k, d=None: dice(candidatos[k], ref_r)) if ref_r is not None else (lambda k: 0)
    mejor_px = max(por_pixel, key=puntaje) if ref_r is not None else "Watershed"
    k_multi = next(k for k in candidatos if "multinivel" in k)
    for metodo in dict.fromkeys([k_multi, mejor_px, "Snake"]):
        filas += analizar_morfologia(rgb_r, candidatos[metodo], ref_r, nombre, stem, metodo, s_roi)
    return filas


def guardar_metricas(filas, archivo, mostrar=True):
    import csv
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["imagen", "etapa", "metodo", "dice", "iou"]); w.writerows(filas)
    if mostrar:
        for f in filas:
            print(f"{f[0]:15s} {f[1]:28s} {f[2]:30s} Dice={f[3]:.3f} IoU={f[4]:.3f}")


# ============================================================================
# VIDEO
# ============================================================================
# ----------------------------------------------------------------------------
# Lectura del video (con rutas no ASCII en Windows)
# ----------------------------------------------------------------------------
@contextmanager
def abrir_video(path):
    """Abre el video y GARANTIZA liberar los recursos al salir (aunque haya un
    error o se corte el proceso con Ctrl+C).

    En Windows, cv2.VideoCapture no abre rutas con caracteres no ASCII (p.ej.
    "Señales"). Se prueba en este orden, de más barato a más caro:
      1. ruta corta 8.3 de Windows (suele ser ASCII);
      2. si sólo la CARPETA tiene acentos y el nombre del archivo no: abrir con
         la ruta relativa desde la carpeta del video (no copia nada);
      3. último recurso: copia temporal, que se borra SIEMPRE al salir.
    En Linux y macOS no hace falta nada de esto."""
    path = _ruta(path, CARPETA_VIDEOS)
    if not path.exists():
        raise FileNotFoundError(path)
    tmp_dir = None
    cap = None
    try:
        if sys.platform == "win32" and not str(path).isascii():
            corta = _ruta_corta_windows(path)
            if corta and corta.isascii():
                cap = cv2.VideoCapture(corta)
            elif path.name.isascii():
                cwd = os.getcwd()
                try:
                    os.chdir(path.parent)
                    cap = cv2.VideoCapture(path.name)
                finally:
                    os.chdir(cwd)
            else:
                tmp_dir = Path(tempfile.mkdtemp())
                copia = tmp_dir / ("video" + path.suffix)
                print(f"  (copiando el video a {tmp_dir} por el nombre con acentos)")
                shutil.copy(path, copia)
                cap = cv2.VideoCapture(str(copia))
        else:
            cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el video {path}")
        yield cap
    finally:
        if cap is not None:
            cap.release()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _ruta_corta_windows(path):
    """Nombre corto 8.3 de Windows (p.ej. C:\\Users\\Facu\\DESKTOP\\PROCES~1\\...)."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024)
        return buf.value if n else None
    except Exception:
        return None


def fps_de(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and np.isfinite(fps) and fps > 0 else 30.0


def leer_frames(cap, max_frames=None, buffer=8):
    """Lee los frames en un hilo aparte: mientras se procesa un frame ya se
    está decodificando el siguiente (decodificar un frame 4K tarda ~0.1 s)."""
    import queue
    import threading
    q = queue.Queue(maxsize=buffer)
    fin = object()
    parar = threading.Event()

    def lector():
        i = 0
        while not parar.is_set() and (max_frames is None or i < max_frames):
            ok, f = cap.read()
            if not ok:
                break
            q.put((i, f))
            i += 1
        q.put(fin)

    hilo = threading.Thread(target=lector, daemon=True)
    hilo.start()
    try:
        while True:
            item = q.get()
            if item is fin:
                return
            yield item
    finally:
        parar.set()
        while hilo.is_alive():           # vaciar la cola para que el hilo termine
            try:
                q.get_nowait()
            except queue.Empty:
                hilo.join(0.05)


# ----------------------------------------------------------------------------
# Inicialización: escala fija
# ----------------------------------------------------------------------------
def inicializar(path, rect=None, nuevo_roi=False, indice=0):
    """ROI manual del video: --roi, el guardado en roi.json o, si no hay, se
    marca con el mouse sobre el frame `indice` (y se guarda en roi.json)."""
    with abrir_video(path) as cap:
        if indice:
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, f = cap.read()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {indice} del video")
    roi = obtener_roi(f, Path(path).name, rect=rect, nuevo=nuevo_roi)
    print(f"ROI: x={roi.x0} y={roi.y0} {roi.w}x{roi.h} px ({roi.origen}); "
          f"escala de parámetros = {roi.esc:.2f}" + (f"; frame de referencia {indice}" if indice else ""))
    return roi


def calibrar_escala(path, roi, n=10, t=2.0, inicio=0):
    """Tamaño de referencia para TODOS los parámetros (radio de iris, en px del
    video original), calculado a partir de la PUPILA y no del tamaño del ROI:
    se mide la pupila (estimación simple, ver estimar_pupila) en n frames de
    los t segundos que siguen al frame `inicio` y se toma la mediana
    (descarta parpadeos)."""
    radios = []
    with abrir_video(path) as cap:
        fps = fps_de(cap)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f_rap = min(1.0, 600 / max(roi.w, roi.h))          # sólo para que sea rápido
        fin = min(total, inicio + a_frames(t, fps))
        for i in np.unique(np.linspace(inicio, max(inicio + 1, fin) - 1, n).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                continue
            crop, _ = recortar(f, roi, f_rap)
            x, y, r = estimar_pupila(m1.suavizar_roi(crop)["suave_ubicacion"])
            h, w = crop.shape[:2]
            # se descartan las estimaciones pegadas al borde del ROI (pelo, sombras)
            if r > 2 and ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2 < 0.8:
                radios.append(r / f_rap)
    if not radios:
        print("  [aviso] no se pudo medir la pupila: la escala se deduce del ROI")
        return roi.r_ref
    r_p = float(np.median(radios))
    r_iris = r_p / PUPILA_SOBRE_IRIS
    print(f"Escala: radio de pupila {'al comienzo' if not inicio else f'desde el frame {inicio}'} "
          f"≈ {r_p:.1f} px -> radio de iris de referencia "
          f"≈ {r_iris:.0f} px (mediana de {len(radios)} frames)")
    return r_iris


# ----------------------------------------------------------------------------
# Segmentación de un frame
# ----------------------------------------------------------------------------
def circulo(cx, cy, r, n=120):
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[cy + r * np.sin(t), cx + r * np.cos(t)]       # (fila, col)


def remuestrear(c, n=120):
    """Puntos equiespaciados sobre el contorno: la snake es inestable con
    puntos irregulares (los de find_contours) y colapsa a un punto."""
    c = c[:-1] if np.allclose(c[0], c[-1]) else c
    cc = np.vstack([c, c[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(cc, axis=0).T))]
    s = np.linspace(0, d[-1], n, endpoint=False)
    return np.c_[np.interp(s, d, cc[:, 0]), np.interp(s, d, cc[:, 1])]


def segmentar_frame(mejora, centro, esc, init=None, r0=None):
    """Snake + morfología sobre el recorte del ROI ya mejorado por el módulo 1.
    mejora: salida de m1.mejorar_roi (o de m1.suavizar_roi + m1.realzar_pupila):
            usa "suave" y el umbral F2 para la zona oscura y "realzada" para la snake.
    centro: posición (x, y) estimada de la pupila (semilla) en el recorte.
    esc:    escala de los parámetros para este ROI (ver la sección ROI MANUAL).
    init:   contorno inicial (fila, col) en coords. del recorte, o None.
    r0:     radio estimado de la pupila (para el círculo inicial si init=None).
    Devuelve la máscara de la pupila, o None si no se encontró."""
    segmentar_frame.snake = None
    g, t_multi, img = mejora["suave"], mejora["umbral"], mejora["realzada"]
    if t_multi is None:                # sin contraste (ojo tapado, imagen saturada)
        return None
    h, w = g.shape
    semilla = (int(np.clip(round(centro[0]), 0, w - 1)), int(np.clip(round(centro[1]), 0, h - 1)))
    se = disk(px(2, esc))
    oscuro = componente_con_semilla(ndi.binary_fill_holes(g < t_multi), semilla)
    if init is None:
        if oscuro.sum() < 10:
            return None
        if r0:                         # radio estimado de la pupila (estimar_pupila)
            cx, cy, r = centro[0], centro[1], r0
        else:                          # círculo inscripto en la zona oscura
            cx, cy, r = circulo_inscripto(oscuro)
        init = circulo(cx, cy, 1.3 * r)
    else:
        init = remuestrear(init)
        # el contorno anterior se agranda un poco para que la snake pueda
        # "cerrarse" sobre el borde también si la pupila se dilató
        cen = init.mean(0)
        init = cen + 1.25 * (init - cen)
    # las fuerzas internas (elasticidad alpha, rigidez beta) dependen de la
    # escala: en resolución original se dividen por esc² para que la snake no
    # colapse antes de llegar al borde
    snake = active_contour(img, init, alpha=0.1 / esc ** 2, beta=1.0 / esc ** 2, gamma=0.01,
                           w_line=0, w_edge=1, max_num_iter=2500)
    mask = polygon2mask(img.shape, snake)
    mask = componente_con_semilla(binary_opening(mask, se), semilla) if mask.any() else mask
    mask = ndi.binary_fill_holes(binary_closing(mask, se))
    segmentar_frame.snake = snake                    # subpíxel
    return mask


def ancho_horizontal(contorno, paso=0.1):
    """Máxima cuerda horizontal de un polígono (fila, col) con precisión
    subpíxel: para cada fila (cada `paso` px) se intersecan los lados."""
    y, x = contorno[:, 0], contorno[:, 1]
    y2, x2 = np.roll(y, -1), np.roll(x, -1)
    mejor = 0.0
    for yy in np.arange(y.min(), y.max(), paso):
        cruza = (y - yy) * (y2 - yy) < 0
        if cruza.sum() >= 2:
            xs = x[cruza] + (yy - y[cruza]) * (x2[cruza] - x[cruza]) / (y2[cruza] - y[cruza])
            mejor = max(mejor, xs.max() - xs.min())
    return mejor


def calidad(mask, mejora, r_ref):
    """Devuelve (válido, motivo, medidas). Rangos fisiológicos con el iris
    de referencia r_ref (deducido del tamaño del ROI): la pupila mide entre
    ~10 % y ~90 % del iris. De la mejora del módulo 1 se usan el gris sin
    filtrar (oscuridad y contraste) y el umbral F2."""
    if mask is None or mask.sum() < 10:
        return False, "sin pupila", {}
    gris_crop, t_multi = mejora["gris"], mejora["umbral"]
    area = mask.sum()
    r_eq = np.sqrt(area / np.pi)
    # forma: elipse ajustada (relación de ejes) y solidez (área / casco convexo)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return False, "sin pupila", {}
    (_, _), (a1, a2), _ = cv2.fitEllipse(c)
    ejes = min(a1, a2) / max(a1, a2)
    solidez = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
    circ = ejes * solidez
    oscuridad = gris_crop[mask].mean()
    med = dict(r_eq=r_eq, circularidad=circ, oscuridad=oscuridad)
    rel = r_eq / r_ref
    if not 0.10 < rel < 0.90:
        return False, f"tamaño fuera de rango ({rel:.2f})", med
    if circ < 0.65:
        return False, f"poco circular ({circ:.2f})", med
    if oscuridad > t_multi + 15:
        return False, "no es oscura (¿parpadeo?)", med
    # la pupila tiene que ser bastante más oscura que lo que la rodea (el iris);
    # una sombra o el párpado cerrado no cumplen esto
    m8 = mask.astype(np.uint8)
    ext = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, int(0.4 * r_eq)) + 1,) * 2))
    intr = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(0.1 * r_eq)) + 1,) * 2))
    anillo = (ext > 0) & (intr == 0)
    contraste = float(gris_crop[anillo].mean() - oscuridad) if anillo.any() else 0.0
    med["contraste"] = contraste
    if contraste < 20:
        return False, f"poco contraste con el iris ({contraste:.0f})", med
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return False, "pupila en el borde del ROI", med
    return True, "", med


# ----------------------------------------------------------------------------
# Referencias manuales (frames anotados)
# ----------------------------------------------------------------------------
def cargar_refs():
    p = ARCHIVO_REFERENCIAS
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def ref_en_recorte(pts, info, shape):
    q = a_recorte(pts, info).astype(np.float32)
    if len(q) < 15:
        (cx, cy), ax, ang = cv2.fitEllipse(q)
        m = np.zeros(shape[:2], np.uint8)
        cv2.ellipse(m, ((cx, cy), ax, ang), 1, -1)
        return m.astype(bool)
    return polygon2mask(shape[:2], q[:, ::-1])


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def preparar(path, rect=None, nuevo_roi=False, factor=None, frame_referencia=None):
    """ROI, escala (radio de iris de referencia) y resolución de procesamiento
    del video: lo que necesitan el mejoramiento (módulo 1) y la segmentación.
    frame_referencia: frame sobre el que se marca el ROI y desde el que se mide
    la escala; si no se da, el guardado para este video o, si no hay, el primero.
    Devuelve un dict con roi, r_iris, factor y frame_referencia."""
    nombre = Path(path).name
    if frame_referencia is None:
        frame_referencia = frame_referencia_guardado(nombre) or 0
    roi = inicializar(path, rect, nuevo_roi, frame_referencia)
    guardar_frame_referencia(nombre, frame_referencia)
    r_iris = calibrar_escala(path, roi, inicio=frame_referencia)
    factor = factor_auto(roi, r_iris) if factor is None else min(1.0, factor)
    print(f"Resolución de procesamiento: {100 * factor:.0f} % del video original "
          f"(radio de iris ≈ {r_iris * factor:.0f} px)")
    return dict(roi=roi, r_iris=r_iris, factor=factor, frame_referencia=frame_referencia)


def frame_mejorado(path, preparacion, indice=None):
    """Recorte del ROI de un frame y su mejora (módulo 1), igual que al comienzo
    del seguimiento: la pupila se ubica en todo el ROI y el rango de interés se
    calcula a su alrededor. Sirve para mostrar el paso de mejoramiento.
    indice: el frame a mostrar; si no se da, el de referencia de la preparación.
    Devuelve (recorte BGR, mejora, info del recorte)."""
    if indice is None:
        indice = preparacion.get("frame_referencia", 0)
    with abrir_video(path) as cap:
        if indice:
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {indice} del video")
    crop, info = recortar(frame, preparacion["roi"], preparacion["factor"], preparacion["r_iris"])
    mejora = m1.suavizar_roi(crop, info["esc"])
    x, y, _ = estimar_pupila(mejora["suave_ubicacion"])
    return crop, m1.realzar_pupila(mejora, info["esc"], (x, y), info["r_ref"]), info


def procesar_video(path, extraer=0, max_frames=None, rect=None, nuevo_roi=False, factor=None,
                   preparacion=None):
    """Seguimiento de la pupila en todo el video. preparacion: la de preparar()
    (si no se da, se calcula acá: ROI, escala y resolución)."""
    global ESC
    OUT_VIDEO.mkdir(exist_ok=True)
    stem = Path(path).stem
    prep = preparacion or preparar(path, rect, nuevo_roi, factor)
    roi, r_iris, factor = prep["roi"], prep["r_iris"], prep["factor"]
    contornos = {}           # contorno de la pupila de cada frame válido (coords. originales)
    tam_vis = (int(round(ALTO_OVERLAY * roi.w / roi.h)) // 2 * 2, ALTO_OVERLAY)
    refs = cargar_refs()
    filas, metricas = [], []
    tmp_out_dir = Path(tempfile.mkdtemp())
    try:
        with abrir_video(path) as cap:
            fps = fps_de(cap)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames:
                total = min(total, max_frames)
            print(f"Video: {total} frames a {fps:.2f} fps ({total / fps:.1f} s). Procesando...", flush=True)
            a_extraer = set(np.linspace(0, total - 1, extraer).astype(int)) if extraer else set()
            if a_extraer:
                (OUT_VIDEO / "frames").mkdir(exist_ok=True)
            escritor = abrir_escritor(tmp_out_dir / "overlay.mp4", fps, tam_vis)
            contorno_prev = None     # contorno de la pupila del frame anterior (coords. del recorte)
            centro = None            # posición de la pupila en el recorte
            ultimo_valido = None     # última posición válida de la pupila
            r0 = None                # radio estimado de la pupila (re-inicialización)
            t_inicio = ultimo_print = time.time()
            try:
                for i, frame in leer_frames(cap, max_frames):
                    if i in a_extraer:
                        nombre = f"{stem}_frame_{i:04d}.png"
                        cv2.imencode(".png", frame)[1].tofile(str(OUT_VIDEO / "frames" / nombre))

                    crop, info = recortar(frame, roi, factor, r_iris)    # SIEMPRE el mismo ROI
                    esc, r_ref = info["esc"], info["r_ref"]
                    # Módulo 1, primera etapa: escala de grises y suavizado
                    mejora = m1.suavizar_roi(crop, esc)
                    if contorno_prev is None or centro is None:
                        # inicio o después de un parpadeo: centro de la zona oscura más
                        # "gruesa" del ROI (estimar_pupila), cerca de la última posición válida
                        ESC = esc
                        x, y, r0 = estimar_pupila(mejora["suave_ubicacion"], cerca_de=ultimo_valido, radio=r_ref)
                        centro = (x, y)
                    # Módulo 1, segunda etapa: rango de interés alrededor de la pupila
                    mejora = m1.realzar_pupila(mejora, esc, centro, r_ref)
                    # Módulo 2: segmentación de la imagen mejorada
                    mask = segmentar_frame(mejora, centro, esc, contorno_prev,
                                           r0 if contorno_prev is None else None)
                    valido, motivo, med = calidad(mask, mejora, r_ref)

                    fila = dict(frame=i, t=i / fps, valido=int(valido), motivo=motivo,
                                roi_x0=roi.x0, roi_y0=roi.y0, roi_w=roi.w, roi_h=roi.h,
                                circularidad=med.get("circularidad", np.nan),
                                cx=np.nan, cy=np.nan, d_pupila_px=np.nan, d_area_px=np.nan)
                    if valido:
                        cy_c, cx_c = ndi.center_of_mass(mask)
                        ox, oy = a_original([(cx_c, cy_c)], info)[0]   # coords. de la imagen original
                        # diámetro HORIZONTAL (ancho máximo, subpíxel sobre la snake): los
                        # párpados tapan la pupila desde arriba/abajo, el ancho no cambia
                        sn = segmentar_frame.snake
                        d_crop = ancho_horizontal(sn) if sn is not None else float(mask.sum(1).max())
                        # medidas en píxeles del video ORIGINAL
                        fila.update(cx=ox, cy=oy, d_pupila_px=d_crop / factor,
                                    d_area_px=2 * med["r_eq"] / factor)
                        centro = ultimo_valido = (cx_c, cy_c)
                        cs = find_contours(mask.astype(float), 0.5)
                        contorno_prev = max(cs, key=len) if cs else None
                        if contorno_prev is not None:     # (fila, col) -> (x, y) originales
                            contornos[i] = a_original(contorno_prev[:, ::-1], info)
                    else:
                        contorno_prev = None

                    nombre = f"{stem}_frame_{i:04d}.png"
                    if nombre in refs:                         # Dice/IoU si el frame está anotado
                        ref = ref_en_recorte(refs[nombre], info, mejora["gris"].shape)
                        m_eval = mask if mask is not None else np.zeros_like(ref)
                        metricas.append(dict(frame=i, dice=dice(m_eval, ref), iou=iou(m_eval, ref)))
                    filas.append(fila)

                    if escritor is not None:                  # overlay
                        vis = cv2.resize(crop, tam_vis, interpolation=cv2.INTER_AREA)
                        if mask is not None:
                            cs2, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), vis.shape[1::-1],
                                                                 interpolation=cv2.INTER_NEAREST),
                                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            cv2.drawContours(vis, cs2, -1, (0, 255, 0) if valido else (0, 0, 255), 2)
                        txt = f"t={i / fps:5.2f}s  " + (f"d={fila['d_pupila_px']:.1f} px" if valido else motivo[:24])
                        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        escritor.write(vis)
                    ahora = time.time()
                    if ahora - ultimo_print > 2 or i == 0 or i == total - 1:
                        # progreso cada ~2 s (con tiempo restante estimado), así se ve
                        # que el programa avanza aunque cada frame tarde
                        hechos = i + 1
                        resta = (ahora - t_inicio) / hechos * (total - hechos)
                        print(f"  frame {hechos}/{total} ({100 * hechos / total:.0f} %)  "
                              f"faltan ~{resta:.0f} s  " + ("ok" if valido else motivo), flush=True)
                        ultimo_print = ahora
            except KeyboardInterrupt:
                # Ctrl+C: se guarda lo procesado hasta ahora en vez de perderlo
                print(f"\n  [interrumpido] se guardan los {len(filas)} frames procesados")
            if escritor is not None:
                escritor.release()
        # columnas relativas al iris de referencia (las usa el módulo 3)
        for f in filas:
            f["pupila_iris"] = f["d_pupila_px"] / (2 * r_iris)
            f["d_area_iris"] = f["d_area_px"] / (2 * r_iris)
        # PRIMERO se guardan los resultados (serie, gráfico, métricas) y recién
        # después se mueve el overlay: si algo falla con el video de salida, la
        # medición no se pierde
        guardar_resultados(filas, metricas, stem, fps)
        borde = sum(r["motivo"] == "pupila en el borde del ROI" for r in filas)
        if borde > 0.05 * max(len(filas), 1):
            print(f"  [aviso] en {borde} frames la pupila tocó el borde del ROI (movimiento del "
                  "ojo o de la cabeza): conviene marcar un ROI con más margen")
        try:
            sal = tmp_out_dir / "overlay.mp4"
            if sal.exists() and sal.stat().st_size > 0:
                shutil.move(str(sal), str(OUT_VIDEO / f"{stem}_overlay.mp4"))
            else:
                print("  [aviso] no se generó el video overlay (los resultados sí se guardaron)")
        except OSError as e:
            print(f"  [aviso] no se pudo guardar el video overlay: {e}")
    finally:
        shutil.rmtree(tmp_out_dir, ignore_errors=True)
    return filas, roi, fps, factor, r_iris, contornos


def limpiar_resultados():
    """Borra resultados/resultados_modulo2 de corridas anteriores (y la vuelve a crear vacía)."""
    if CARPETA_RESULTADOS.exists():
        shutil.rmtree(CARPETA_RESULTADOS, ignore_errors=True)
        if CARPETA_RESULTADOS.exists():   # Windows: algún archivo abierto (p.ej. el overlay)
            print("  [aviso] no se pudo borrar todo resultados_modulo2/ (¿hay un archivo abierto?)")
    OUT_VIDEO.mkdir(parents=True, exist_ok=True)
    OUT_IMAGEN.mkdir(parents=True, exist_ok=True)
    print("Resultados anteriores del módulo 2 borrados")


def abrir_escritor(destino, fps, tam):
    """VideoWriter del overlay. Con fps "raros" (p.ej. 119.84104...) el
    codificador MPEG-4 rechaza la base de tiempo y el escritor NO se abre (sin
    avisar): se redondean los fps y se verifica. Si igual falla, se sigue sin
    overlay en lugar de perder todo el procesamiento. Se escribe en una
    carpeta temporal porque VideoWriter tampoco acepta rutas no ASCII en Windows."""
    for f in (round(fps, 2), round(fps), 30.0):
        w = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), f, tam)
        if w.isOpened():
            if f != fps:
                print(f"  overlay a {f} fps (el video original tiene {fps:.5f} fps)")
            return w
        w.release()
    print("  [aviso] no se pudo crear el video overlay; se continúa sin él")
    return None


# ----------------------------------------------------------------------------
# Vínculo con la sección 2.1 y 2.2: análisis sobre el/los frame(s) de trabajo
# ----------------------------------------------------------------------------
def elegir_frames(filas, pedidos, fps=30.0):
    """pedidos: lista con números de frame y/o 'auto', 'dilatada', 'contraida'.
      auto      -> el frame más "limpio": válido, sin parpadeos en ±T_ESTABLE s y
                   con la pupila más circular
      dilatada  -> el de mayor diámetro pupilar (entre los estables)
      contraida -> el de menor diámetro pupilar (entre los estables)"""
    ok = np.array([r["valido"] == 1 for r in filas])
    k = a_frames(T_ESTABLE, fps)
    estables = [i for i in range(len(filas)) if ok[max(0, i - k):i + k + 1].all()] or list(np.where(ok)[0])
    if not estables:
        raise RuntimeError("No hay frames válidos para analizar")
    d = {i: filas[i]["d_pupila_px"] for i in estables}
    elegidos = {}
    for p in pedidos:
        p = str(p).lower()
        if p == "auto":
            elegidos[max(estables, key=lambda i: filas[i]["circularidad"])] = "trabajo"
        elif p.startswith("dilat"):
            elegidos[max(d, key=d.get)] = "dilatada"
        elif p.startswith("contra"):
            elegidos[min(d, key=d.get)] = "contraída"
        else:
            elegidos[int(p)] = "manual"
    return elegidos


def analizar_frames(path, filas, roi, pedidos, fps, factor=None, r_iris=None, contornos=None):
    stem = Path(path).stem
    elegidos = elegir_frames(filas, pedidos, fps)
    (OUT_VIDEO / "frames").mkdir(parents=True, exist_ok=True)
    filas_img = []
    with abrir_video(path) as cap:
        for i, tipo in sorted(elegidos.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                print(f"  no se pudo leer el frame {i}"); continue
            archivo = OUT_VIDEO / "frames" / f"{stem}_frame_{i:04d}.png"
            cv2.imencode(".png", frame)[1].tofile(str(archivo))
            print(f"\n=== Frame {i} ({tipo}, t={i / fps:.2f}s): módulos 2.1 y 2.2 ===")
            # el MISMO ROI, la MISMA escala y, como punto de partida, la pupila
            # que encontró el video en ese frame
            filas_img += procesar(archivo, roi_fijo=roi, factor=factor, r_iris=r_iris,
                                     pupila_video=(contornos or {}).get(i))
    destino = OUT_IMAGEN / f"metricas_{stem}.csv"
    guardar_metricas(filas_img, destino, mostrar=False)
    print("\nResumen 2.1 (Dice; nan = frame sin referencia anotada):")
    for f in filas_img:
        if f[1] == "2.1":
            print(f"  {f[0]:28s} {f[2]:30s} Dice={f[3]:.3f}")
    print(f"Resultados de imagen en {OUT_IMAGEN.relative_to(RAIZ)}")
    return elegidos


def marcar_atipicos(filas, ventana, tol=0.15):
    """Consistencia temporal: la pupila no cambia más de ~15 % en T_ATIPICOS
    segundos. Los valores que se apartan de la mediana móvil de los
    frames válidos (típicamente parpadeos parciales) se invalidan."""
    d = np.array([r["d_pupila_px"] for r in filas], float)
    ok = np.where(~np.isnan(d))[0]
    for k, i in enumerate(ok):
        vec = d[ok[max(0, k - ventana // 2):k + ventana // 2 + 1]]
        med = np.median(vec)
        if abs(d[i] - med) > tol * med:
            filas[i].update(valido=0, motivo=f"atípico ({d[i]:.1f} vs {med:.1f} px)",
                            d_pupila_px=np.nan, pupila_iris=np.nan)


def guardar_resultados(filas, metricas, stem, fps=30.0):
    if not filas:
        print("  [aviso] no se procesó ningún frame"); return
    marcar_atipicos(filas, a_frames(T_ATIPICOS, fps, impar=True, minimo=3))
    with open(OUT_VIDEO / f"{stem}_serie.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)

    t = np.array([r["t"] for r in filas])
    d = np.array([r["d_pupila_px"] for r in filas], float)
    ok = ~np.isnan(d)
    # interpolación de los frames inválidos (parpadeos) + mediana móvil
    d_int = np.interp(t, t[ok], d[ok]) if ok.sum() > 1 else d
    n_med = a_frames(T_MEDIANA, fps, impar=True, minimo=3)
    d_suave = medfilt(d_int, n_med)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[ok], d[ok], ".", ms=3, color="gray", label="medido")
    ax.plot(t, d_suave, "-", color="C0", label=f"interpolado + mediana ({n_med} frames = {n_med / fps:.2f} s)")
    for ti in t[~ok]:
        ax.axvspan(ti - 0.5 / fps, ti + 0.5 / fps, color="r", alpha=0.15, lw=0)
    ax.set_xlabel("tiempo [s]"); ax.set_ylabel("diámetro horizontal de la pupila [px]")
    ax.set_title(f"{stem}: {ok.sum()}/{len(d)} frames válidos (rojo = inválidos)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_VIDEO / f"{stem}_serie.png", dpi=130); plt.close(fig)

    print(f"\nFrames válidos: {ok.sum()}/{len(d)}")
    if ok.any():
        print(f"Diámetro de la pupila: min={np.nanmin(d):.1f} px  max={np.nanmax(d):.1f} px")
    if metricas:
        print("\nFrames anotados:")
        for m in metricas:
            print(f"  frame {m['frame']:4d}  Dice={m['dice']:.3f}  IoU={m['iou']:.3f}")
        print(f"  promedio  Dice={np.mean([m['dice'] for m in metricas]):.3f}  "
              f"IoU={np.mean([m['iou'] for m in metricas]):.3f}")
    print(f"Resultados en {OUT_VIDEO.relative_to(RAIZ)}")


# ============================================================================
# REFERENCIAS ANOTADAS A MANO
# ============================================================================
def anotar(path):
    bgr = leer_bgr(path)
    # mismo ROI manual que usa el análisis (un frame de video usa el ROI del video)
    crop, info = recortar(bgr, obtener_roi(bgr, Path(path).name))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 9 * crop.shape[0] / crop.shape[1]))
    ax.imshow(rgb, interpolation="nearest")        # el ROI completo, en resolución original
    ax.set_title(f"{Path(path).name}: marcá el borde de la pupila y apretá Enter")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) == 0:
        return []
    # se guardan en coordenadas de la imagen ORIGINAL (independientes del recorte)
    return [[round(float(px), 1), round(float(py), 1)] for px, py in a_original(pts, info)]


def anotar_imagenes(paths):
    """Anota el contorno de la pupila en cada imagen y lo guarda en referencias.json."""
    CARPETA_ANOTACIONES.mkdir(parents=True, exist_ok=True)
    datos = json.loads(ARCHIVO_REFERENCIAS.read_text(encoding="utf-8")) if ARCHIVO_REFERENCIAS.exists() else {}
    for p in paths:
        pts = anotar(p)
        if len(pts) >= 5:
            datos[Path(p).name] = pts
            print(f"{p}: {len(pts)} puntos guardados")
        else:
            print(f"{p}: se necesitan al menos 5 puntos, no se guardó")
    datos.pop("_nota", None)
    ARCHIVO_REFERENCIAS.write_text(json.dumps(datos, indent=2), encoding="utf-8")


# ============================================================================
# EJECUCIÓN
# ============================================================================
def buscar_video(nombre=None):
    """Ruta del video: `nombre` (en videos/ o una ruta), NOMBRE_VIDEO o, si no se
    indica ninguno, el video modificado más recientemente en videos/."""
    nombre = nombre or NOMBRE_VIDEO
    if nombre:
        ruta = _ruta(nombre, CARPETA_VIDEOS)
        if not ruta.exists():
            sys.exit(f"No se encontró el video {nombre} (se buscó en {CARPETA_VIDEOS})")
        return ruta
    videos = sorted((p for p in CARPETA_VIDEOS.iterdir()
                     if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}),
                    key=lambda p: p.stat().st_mtime) if CARPETA_VIDEOS.exists() else []
    if not videos:
        sys.exit(f"No hay ningún video en {CARPETA_VIDEOS}")
    if len(videos) > 1:
        print("Hay varios videos en videos/; se usa el modificado más recientemente:")
        for p in videos:
            print(f"    {p.name}")
    return videos[-1]


def ejecutar_modulo2(ruta_video=None, rect=None, usar_roi_guardado=False, factor=None,
                     frames=("auto",), extraer=8, max_frames=None, borrar=None, preparacion=None):
    """Módulo 2 completo sobre un video: ROI manual, seguimiento de la pupila en
    todo el video y análisis 2.1/2.2 sobre el/los frame(s) de trabajo.
    preparacion: ROI, escala y resolución ya calculados con preparar() (el
    pipeline los calcula antes, en el paso del módulo 1); si no, se calculan acá.
    Devuelve (filas, r_iris): la serie frame a frame (lista de diccionarios,
    también guardada en CSV) y el radio de iris de referencia en px del video
    original (escala usada por el módulo 3)."""
    ruta = buscar_video(ruta_video)
    print(f"Video: {ruta.name}")
    if (BORRAR_RESULTADOS if borrar is None else borrar):
        limpiar_resultados()
    OUT_VIDEO.mkdir(parents=True, exist_ok=True)
    OUT_IMAGEN.mkdir(parents=True, exist_ok=True)
    nuevo_roi = rect is None and PEDIR_ROI_SIEMPRE and not usar_roi_guardado
    filas, roi, fps, factor, r_iris, contornos = procesar_video(ruta, extraer, max_frames,
                                                               rect, nuevo_roi, factor, preparacion)
    analizar_frames(ruta, filas, roi, list(frames), fps, factor, r_iris, contornos)
    return filas, r_iris


def main():
    ap = argparse.ArgumentParser(description="Módulo 2: segmentación y morfología de la pupila")
    ap.add_argument("video", nargs="?", default=None,
                    help="video (nombre dentro de videos/ o ruta); por defecto el más reciente de videos/")
    ap.add_argument("--roi", nargs=4, type=float, metavar=("X", "Y", "ANCHO", "ALTO"),
                    help="ROI sin ventana: esquina superior izquierda, ancho y alto (px de la imagen original)")
    ap.add_argument("--usar-roi-guardado", action="store_true",
                    help="reutilizar el ROI guardado en anotaciones_manuales/roi.json")
    ap.add_argument("--conservar-resultados", action="store_true",
                    help="no borrar resultados/resultados_modulo2 antes de empezar")
    ap.add_argument("--factor", type=float, default=None,
                    help="resolución de procesamiento (1 = original; por defecto el iris queda de 64 px de radio)")
    ap.add_argument("--frame", nargs="+", default=["auto"],
                    help="frame(s) para 2.1/2.2: número(s), 'auto', 'dilatada' y/o 'contraida'")
    ap.add_argument("--extraer", type=int, default=8, help="frames equiespaciados a guardar para anotar")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--imagen", nargs="+", metavar="IMG",
                    help="en vez de un video: análisis 2.1/2.2 sobre imagen(es) suelta(s)")
    ap.add_argument("--anotar", nargs="+", metavar="IMG",
                    help="anotar a mano el contorno de la pupila (referencia para Dice/IoU)")
    a = ap.parse_args()

    if a.anotar:
        anotar_imagenes(a.anotar)
        return
    if a.imagen:
        if BORRAR_RESULTADOS and not a.conservar_resultados:
            shutil.rmtree(OUT_IMAGEN, ignore_errors=True)
        OUT_IMAGEN.mkdir(parents=True, exist_ok=True)
        filas = []
        for p in a.imagen:
            try:
                filas += procesar(_ruta(p), rect=a.roi,
                                  nuevo_roi=PEDIR_ROI_SIEMPRE and not a.usar_roi_guardado)
            except RuntimeError as e:          # selección del ROI cancelada
                print(f"  {p}: se omite ({e})")
        guardar_metricas(filas, OUT_IMAGEN / "metricas.csv")
        return
    ejecutar_modulo2(a.video, a.roi, a.usar_roi_guardado, a.factor, a.frame, a.extraer,
                     a.max_frames, borrar=BORRAR_RESULTADOS and not a.conservar_resultados)


if __name__ == "__main__":
    main()