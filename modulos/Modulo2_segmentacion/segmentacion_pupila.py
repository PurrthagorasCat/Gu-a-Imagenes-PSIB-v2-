"""
TP Procesamiento de Imágenes — Módulo 2
Segmentación y morfología matemática de la pupila.

Uso:
    python segmentacion_pupila.py contraida.png dilatada.png
Genera figuras en ./resultados/ y una tabla de métricas (Dice / IoU) por consola y en CSV.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")   # sin ventanas: las figuras se guardan en resultados/
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian, sobel
from skimage.segmentation import active_contour, watershed
from skimage.morphology import disk, skeletonize
from scipy.ndimage import binary_erosion as _be, binary_dilation as _bd


def binary_erosion(m, se):  return _be(m, se)
def binary_dilation(m, se): return _bd(m, se)
def binary_opening(m, se):  return binary_dilation(binary_erosion(m, se), se)
def binary_closing(m, se):  return binary_erosion(binary_dilation(m, se), se)
from skimage.draw import polygon2mask
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_ojo import recortar_ojo, a_recorte  # noqa: E402

# Rutas relativas a la carpeta del script (no a la carpeta desde donde se ejecuta)
BASE = Path(__file__).resolve().parent
OUT = BASE / "resultados"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def leer_bgr(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE / path
    # cv2.imread falla en Windows con rutas no ASCII (p.ej. "Señales"):
    # se lee el archivo como bytes con numpy y se decodifica con OpenCV
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR) if path.exists() else None
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr


# ----------------------------------------------------------------------------
# ROI del ojo (paso previo) -> ver roi_ojo.py
# ----------------------------------------------------------------------------
def figura_roi(bgr, info, crop, destino):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    grosor = max(1, int(round(max(bgr.shape[:2]) / 400)))
    for c in info["candidatos"][1:]:
        cv2.circle(rgb, (int(c.x), int(c.y)), int(c.r), (255, 255, 0), grosor)
    c = info["cand"]
    cv2.circle(rgb, (int(c.x), int(c.y)), int(c.r), (0, 255, 0), 2 * grosor)
    cv2.rectangle(rgb, (info["x0"], info["y0"]), (info["x0"] + info["lado"], info["y0"] + info["lado"]), (255, 0, 0), 2 * grosor)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].set_title(f"verde: ojo elegido ({c.origen}, confianza={c.puntaje:.2f}) | amarillo: descartados | rojo: recorte", fontsize=7)
    ax[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax[1].set_title("Recorte normalizado (iris de radio fijo)")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(destino, dpi=120); plt.close(fig)


def cargar(path, cand=None):
    """Lee la imagen completa, recorta el ojo y devuelve rgb/gris/lab del recorte.
    Si se pasa `cand` (ROI ya delimitado, p.ej. por procesar_video.py) se usa
    ese recorte en lugar de volver a detectar el ojo."""
    bgr_full = leer_bgr(path)
    print(f"{Path(path).name}:")
    bgr, info = recortar_ojo(bgr_full, cand=cand, manual_si_falla=True)
    figura_roi(bgr_full, info, bgr, OUT / f"{Path(path).stem}_00_roi_ojo.png")
    cv2.imencode(".png", bgr)[1].tofile(str(OUT / f"{Path(path).stem}_ojo.png"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return rgb, gris, lab, info


def localizar_pupila(gris, sigma=4):
    """Semilla = mínimo de la imagen muy suavizada (la pupila es la zona oscura
    más extensa; el suavizado evita que gane una pestaña o un pixel aislado)."""
    suave = cv2.GaussianBlur(gris, (0, 0), sigma)
    _, _, (x, y), _ = cv2.minMaxLoc(suave)
    return x, y


def roi(shape, semilla, frac=0.16):
    """Ventana cuadrada alrededor de la semilla (dentro del iris; el recorte
    del ojo ya está a escala fija, así que la fracción es constante)."""
    h, w = shape[:2]
    m = int(frac * min(h, w))
    x, y = semilla
    y0, y1 = max(0, y - m), min(h, y + m)
    x0, x1 = max(0, x - m), min(w, x + m)
    return (slice(y0, y1), slice(x0, x1)), (x - x0, y - y0)


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
def seg_otsu(gris, sl, s_roi):
    g = cv2.GaussianBlur(gris, (5, 5), 0)
    # (a) global sobre toda la imagen
    t_glob = threshold_otsu(g)
    m_glob = g < t_glob
    # (b) dentro del ROI (el histograma queda bimodal pupila / iris)
    r = g[sl]
    t_roi = threshold_otsu(r)
    m = componente_con_semilla(r < t_roi, s_roi)
    # (c) Otsu multinivel (3 clases: pupila / iris oscuro+párpado / iris claro+piel)
    t_multi = threshold_multiotsu(r, classes=3)[0]
    m_multi = componente_con_semilla(r < t_multi, s_roi)
    return m_glob, t_glob, m, t_roi, m_multi, t_multi


def seg_kmeans(lab, sl, s_roi, K, w_croma=0.5):
    """Espacio de características: L (intensidad) + a,b (croma), ponderando el croma."""
    r = cv2.GaussianBlur(lab, (5, 5), 0)[sl].astype(np.float32)
    feats = r.reshape(-1, 3).copy()
    feats[:, 1:] *= w_croma
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(feats)
    etiquetas = km.labels_.reshape(r.shape[:2])
    c_pupila = np.argmin(km.cluster_centers_[:, 0])      # cluster con menor L
    m = componente_con_semilla(etiquetas == c_pupila, s_roi)
    return m, etiquetas


def seg_snake(gris, sl, centro, r0, t_clip, n=200):
    """Contorno activo inicializado como círculo que rodea a la pupila.
    Antes se recorta la intensidad en t_clip (umbral inferior de Otsu multinivel):
    todo lo más claro que la pupila queda "plano", así el único borde fuerte es
    la frontera pupila/iris y la snake no es atraída por el párpado o el limbo.
    alpha: elasticidad (tiende a encoger), beta: rigidez (suavidad),
    w_edge: atracción a bordes, gamma: paso temporal."""
    img = np.minimum(gris[sl].astype(float), t_clip) / 255.0
    img = gaussian(img, sigma=1.0, preserve_range=True)
    t = np.linspace(0, 2 * np.pi, n)
    cx, cy = centro
    init = np.c_[cy + r0 * np.sin(t), cx + r0 * np.cos(t)]  # (fila, col)
    snake = active_contour(img, init, alpha=0.1, beta=1.0, gamma=0.01,
                           w_line=0, w_edge=1, max_num_iter=2500)
    m = polygon2mask(img.shape, snake)
    return m, init, snake


def seg_watershed(gris, sl, s_roi, m_semilla):
    """Watershed controlado por marcadores sobre el gradiente.
    Marcadores: 1 = núcleo de pupila, 2 = fondo (iris/esclerótica),
    3 = reflejos corneales (muy brillantes)."""
    g = gaussian(gris[sl] / 255.0, sigma=1.0, preserve_range=True)
    grad = sobel(g)
    # el núcleo se toma de Otsu + apertura (corta los "puentes" hacia las
    # pestañas/pliegue del párpado) -> marcador confiable dentro de la pupila
    nucleo = ndi.binary_fill_holes(m_semilla)
    nucleo = componente_con_semilla(binary_opening(nucleo, disk(3)), s_roi)
    marc = np.zeros(g.shape, np.int32)
    marc[~binary_dilation(nucleo, disk(4))] = 2
    brillo = g > np.percentile(g, 99)                     # destellos
    marc[brillo] = 3
    marc[binary_erosion(nucleo, disk(2)) & ~brillo] = 1
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


def cargar_referencia(nombre, shape, info_roi, archivo="referencias.json"):
    """Las referencias se guardan en coordenadas de la IMAGEN ORIGINAL (así no
    dependen de cómo se recorte) y se llevan al recorte con su transformación."""
    p = BASE / archivo
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
    # barrido del radio del elemento estructurante para la combinación propuesta
    for r in range(1, 5):
        for k, x in variantes_morfologicas(m, r, semilla).items():
            if k != "original" or r == 1:
                filas.append((nombre, f"2.2 [{metodo}] r={r}", k, *sc(x)))
    r_opt = 2
    if ref_r is not None:
        r_opt = max(range(1, 5), key=lambda r: dice(variantes_morfologicas(m, r, semilla)["apertura+CC+cierre+relleno"], ref_r))
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
    fig.suptitle(f"{nombre}: morfología sobre '{metodo}' (EE = disco r={r_opt})")
    fig.tight_layout(); fig.subplots_adjust(hspace=0.4)
    tag = metodo.split(" (")[0].replace(" ", "_").replace("=", "")
    fig.savefig(OUT / f"{stem}_22_morfologia_{tag}.png", dpi=130); plt.close(fig)
    return filas


# ----------------------------------------------------------------------------
# Pipeline por imagen
# ----------------------------------------------------------------------------
def overlay(rgb, mask, color=(255, 0, 0), ref=None):
    o = rgb.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cnts, -1, color, 1)
    if ref is not None:
        cnts, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cnts, -1, (0, 255, 0), 1)
    return o


def procesar(path, cand=None):
    nombre = Path(path).name
    stem = Path(path).stem
    rgb, gris, lab, info_roi = cargar(path, cand)
    semilla = localizar_pupila(gris)
    sl, s_roi = roi(gris.shape, semilla)
    ref = cargar_referencia(nombre, gris.shape, info_roi)
    ref_r = ref[sl] if ref is not None else None
    rgb_r = rgb[sl]

    # ---- 2.1 -------------------------------------------------------------
    m_glob, t_glob, m_otsu, t_roi, m_multi, t_multi = seg_otsu(gris, sl, s_roi)
    # estimación gruesa de centro/radio (para inicializar la snake)
    nuc = componente_con_semilla(binary_opening(ndi.binary_fill_holes(m_multi), disk(2)), s_roi)
    cy0, cx0 = ndi.center_of_mass(nuc)
    r_est = np.sqrt(nuc.sum() / np.pi)
    km = {K: seg_kmeans(lab, sl, s_roi, K) for K in range(2, 6)}
    m_snake, init, snake = seg_snake(gris, sl, (cx0, cy0), 1.5 * r_est, t_multi)
    m_ws, et_ws, grad, marc = seg_watershed(gris, sl, s_roi, m_multi)

    metodos = {f"Otsu global (T={t_glob:.0f})": m_glob[sl],
               f"Otsu ROI (T={t_roi:.0f})": m_otsu,
               f"Otsu multinivel ROI (T={t_multi:.0f})": m_multi}
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
    ax[0].imshow(rgb_r); ax[0].set_title("ROI (verde = ref.)")
    if ref_r is not None:
        ax[0].imshow(overlay(rgb_r, np.zeros_like(ref_r), ref=ref_r))
    for a, (k, m), f in zip(ax[1:], metodos.items(), filas):
        a.imshow(overlay(rgb_r, m, ref=ref_r))
        a.set_title(f"{k}\nDice={f[3]:.2f} IoU={f[4]:.2f}", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{nombre}: métodos de segmentación (rojo = método, verde = referencia)")
    fig.tight_layout(); fig.subplots_adjust(hspace=0.4); fig.savefig(OUT / f"{stem}_21_metodos.png", dpi=130); plt.close(fig)

    # Figura de diagnóstico: Otsu global, clusters, snake, watershed
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    ax[0, 0].imshow(m_glob, cmap="gray"); ax[0, 0].set_title("Otsu global (imagen completa)")
    ax[0, 1].hist(gris[sl].ravel(), 64, color="gray"); ax[0, 1].axvline(t_roi, color="r")
    ax[0, 1].set_title("Histograma ROI + umbral Otsu")
    for i, K in enumerate([3, 5]):
        ax[0, 2 + i].imshow(km[K][1], cmap="tab10"); ax[0, 2 + i].set_title(f"K-means K={K} (clusters)")
    ax[1, 0].imshow(np.minimum(gris[sl], t_multi), cmap="gray")
    ax[1, 0].plot(init[:, 1], init[:, 0], "--b", lw=1, label="inicial")
    ax[1, 0].plot(snake[:, 1], snake[:, 0], "-r", lw=1.5, label="final")
    ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("Snake (imagen recortada)")
    ax[1, 1].imshow(grad, cmap="magma"); ax[1, 1].set_title("Gradiente (Sobel)")
    ax[1, 2].imshow(marc, cmap="tab10", vmin=0, vmax=9); ax[1, 2].set_title("Marcadores 1=pupila 2=fondo 3=reflejo")
    ax[1, 3].imshow(et_ws, cmap="tab10", vmin=0, vmax=9); ax[1, 3].set_title("Watershed (cuencas)")
    for a in ax.ravel():
        if a is not ax[0, 1]:
            a.axis("off")
    fig.tight_layout(); fig.subplots_adjust(hspace=0.4); fig.savefig(OUT / f"{stem}_21_diagnostico.png", dpi=130); plt.close(fig)

    # ---- 2.2 -------------------------------------------------------------
    # Se aplica a (a) el mejor método por píxel (Otsu/K-means/Watershed), donde la
    # morfología tiene más trabajo (puentes a pestañas, huecos por reflejos), y
    # (b) la Snake, que ya sale regularizada por sus fuerzas internas.
    candidatos = {k: v for k, v in metodos.items() if "global" not in k}
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


if __name__ == "__main__":
    imgs = sys.argv[1:] or ["contraida.png", "dilatada.png"]
    filas = []
    for p in imgs:
        try:
            filas += procesar(p)
        except RuntimeError as e:          # no se encontró ojo y no hubo selección manual
            print(f"  {p}: se omite ({e})")
    guardar_metricas(filas, OUT / "metricas.csv")
