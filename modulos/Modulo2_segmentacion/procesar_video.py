"""
TP Procesamiento de Imágenes — Módulo 2: extensión temporal (video).

Uso:
    python procesar_video.py video.mp4                  # procesa todo el video
    python procesar_video.py video.mp4 --extraer 8      # además guarda 8 frames para anotar
    python procesar_video.py video.mp4 --max-frames 60  # prueba rápida
    python procesar_video.py video.mp4 --frame dilatada contraida   # frames para 2.1/2.2
    python procesar_video.py video.mp4 --frame 120                  # un frame elegido a mano
    (sin argumentos: usa el video que esté en la carpeta y el frame "auto")

Salidas:
  resultados_video/  -> análisis temporal (serie, overlay, frames extraídos)
  resultados/        -> análisis de imagen (2.1 y 2.2) sobre el/los frame(s) de
                        trabajo, con el mismo ROI que delimitó el video

Pipeline por frame (método elegido en 2.1/2.2: Snake + morfología):
  1. Inicialización (una vez): se detecta el ojo en varios frames del comienzo y
     se FIJA la escala con la mediana del radio del iris. Así el diámetro de la
     pupila no "salta" por errores de estimación del iris entre frames.
  2. Seguimiento: el recorte de cada frame se centra en la pupila del frame
     anterior (no se vuelve a correr el detector, que es lento).
  3. Segmentación: Snake inicializada con el contorno del frame anterior
     (coherencia temporal) + apertura/CC/cierre/relleno.
  4. Control de calidad: tamaño, circularidad y oscuridad de la pupila. Si falla
     (parpadeo, desenfoque) el frame se marca inválido y se reinicia la Snake;
     si falla varios frames seguidos se vuelve a detectar el ojo.
  5. Salidas en resultados_video/: serie temporal (CSV + gráfico), video con el
     contorno superpuesto y Dice/IoU en los frames anotados (si los hay).
"""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import medfilt
from skimage.draw import polygon2mask
from skimage.filters import gaussian, threshold_multiotsu
from skimage.measure import find_contours
from skimage.segmentation import active_contour

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from roi_ojo import Candidato, detectar_ojo, recortar_ojo, a_recorte, R_IRIS_NORM, LADO_RECORTE  # noqa: E402
import segmentacion_pupila as SP  # noqa: E402
from segmentacion_pupila import (componente_con_semilla, binary_opening, binary_closing,  # noqa: E402
                                 dice, iou, disk)

OUT = BASE / "resultados_video"
FRAMES_INIT = 15          # frames del comienzo usados para fijar la escala
FALLAS_REDETECTAR = 5     # frames inválidos seguidos antes de re-detectar el ojo


# ----------------------------------------------------------------------------
# Lectura del video (con rutas no ASCII en Windows)
# ----------------------------------------------------------------------------
def abrir_video(path):
    """cv2.VideoCapture no abre rutas con caracteres no ASCII en Windows
    (p.ej. "Señales"). En ese caso se copia el video a una carpeta temporal."""
    path = Path(path)
    if not path.is_absolute():
        path = BASE / path
    if not path.exists():
        raise FileNotFoundError(path)
    tmp = None
    if not str(path).isascii():
        tmp = Path(tempfile.mkdtemp()) / ("video" + path.suffix)
        shutil.copy(path, tmp)
        path = tmp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video {path}")
    return cap, tmp


def leer_frames(cap, max_frames=None):
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            return
        yield i, f
        i += 1


# ----------------------------------------------------------------------------
# Inicialización: escala fija
# ----------------------------------------------------------------------------
def inicializar(path, n=FRAMES_INIT):
    cap, tmp = abrir_video(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cands = []
    for i in np.linspace(0, min(total, 3 * n) - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            c, _ = detectar_ojo(f, verbose=False)
            if c is not None:
                cands.append(c)
    cap.release()
    if tmp:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    if not cands:
        raise RuntimeError("No se encontró el ojo en el comienzo del video")
    r_fijo = float(np.median([c.r for c in cands]))
    primero = cands[0]
    print(f"Escala fijada: radio de iris = {r_fijo:.1f} px "
          f"(rango en {len(cands)} frames: {min(c.r for c in cands):.0f}–{max(c.r for c in cands):.0f})")
    return Candidato(primero.x, primero.y, r_fijo, "inicial", primero.puntaje), r_fijo


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


def segmentar_frame(crop, init=None):
    """Snake + morfología sobre el recorte normalizado.
    init: contorno inicial (fila, col) en coords. del recorte, o None."""
    gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    c0 = LADO_RECORTE / 2
    m = int(0.16 * LADO_RECORTE)
    sl = (slice(int(c0 - m), int(c0 + m)), slice(int(c0 - m), int(c0 + m)))
    g = cv2.GaussianBlur(gris, (5, 5), 0)[sl]
    # umbral inferior de Otsu multinivel: pupila vs resto (se adapta a la luz)
    t_multi = threshold_multiotsu(g, classes=3)[0]
    semilla = (m, m)
    oscuro = componente_con_semilla(binary_opening(ndi.binary_fill_holes(g < t_multi), disk(2)), semilla)
    img = gaussian(np.minimum(gris[sl].astype(float), t_multi) / 255.0, sigma=1.0, preserve_range=True)
    segmentar_frame.snake = None
    if init is None:
        if oscuro.sum() < 10:
            return None, t_multi
        cy, cx = ndi.center_of_mass(oscuro)
        r = np.sqrt(oscuro.sum() / np.pi)
        init = circulo(cx, cy, 1.5 * r)
    else:
        init = remuestrear(init) - np.array([sl[0].start, sl[1].start]) * 1.0
        # el contorno anterior se agranda un poco para que la snake pueda
        # "cerrarse" sobre el borde también si la pupila se dilató
        cen = init.mean(0)
        init = cen + 1.25 * (init - cen)
    snake = active_contour(img, init, alpha=0.1, beta=1.0, gamma=0.01,
                           w_line=0, w_edge=1, max_num_iter=1500)
    mask = polygon2mask(img.shape, snake)
    se = disk(2)
    mask = componente_con_semilla(binary_opening(mask, se), semilla) if mask.any() else mask
    mask = ndi.binary_fill_holes(binary_closing(mask, se))
    full = np.zeros(gris.shape, bool)
    full[sl] = mask
    segmentar_frame.snake = snake + np.array([sl[0].start, sl[1].start])   # subpíxel
    return full, t_multi


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


def recentrar(crop, info, cand):
    """Busca la pupila (zona oscura grande) dentro del círculo del iris del
    recorte y devuelve el candidato centrado en ella."""
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (0, 0), R_IRIS_NORM / 5).astype(float)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    g[np.hypot(xx - LADO_RECORTE / 2, yy - LADO_RECORTE / 2) > 1.2 * R_IRIS_NORM] = 1e9
    py, px = np.unravel_index(np.argmin(g), g.shape)
    ox = (px - info["M"][0, 2]) / info["escala"]
    oy = (py - info["M"][1, 2]) / info["escala"]
    return Candidato(float(ox), float(oy), cand.r, "recentrado", cand.puntaje)


def calidad(mask, gris_crop, t_multi):
    """Devuelve (válido, motivo, medidas). Rangos fisiológicos con el iris
    normalizado a radio R_IRIS_NORM: la pupila mide entre ~15 % y ~80 % del iris."""
    if mask is None or mask.sum() < 10:
        return False, "sin pupila", {}
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
    rel = r_eq / R_IRIS_NORM
    if not 0.12 < rel < 0.85:
        return False, f"tamaño fuera de rango ({rel:.2f})", med
    if circ < 0.65:
        return False, f"poco circular ({circ:.2f})", med
    if oscuridad > t_multi + 15:
        return False, "no es oscura (¿parpadeo?)", med
    return True, "", med


# ----------------------------------------------------------------------------
# Referencias manuales (frames anotados)
# ----------------------------------------------------------------------------
def cargar_refs():
    p = BASE / "referencias.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def ref_en_recorte(pts, info):
    q = a_recorte(pts, info).astype(np.float32)
    if len(q) < 15:
        (cx, cy), ax, ang = cv2.fitEllipse(q)
        m = np.zeros((LADO_RECORTE, LADO_RECORTE), np.uint8)
        cv2.ellipse(m, ((cx, cy), ax, ang), 1, -1)
        return m.astype(bool)
    return polygon2mask((LADO_RECORTE, LADO_RECORTE), q[:, ::-1])


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def procesar_video(path, extraer=0, max_frames=None):
    OUT.mkdir(exist_ok=True)
    stem = Path(path).stem
    cand, r_fijo = inicializar(path)
    refs = cargar_refs()

    cap, tmp = abrir_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    a_extraer = set(np.linspace(0, total - 1, extraer).astype(int)) if extraer else set()
    if a_extraer:
        (OUT / "frames").mkdir(exist_ok=True)

    # VideoWriter tampoco acepta rutas no ASCII en Windows: se escribe en una
    # carpeta temporal y al final se mueve a resultados_video/
    tmp_out = Path(tempfile.mkdtemp()) / "overlay.mp4"
    escritor = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (2 * LADO_RECORTE, 2 * LADO_RECORTE))
    filas, metricas = [], []
    contorno_prev = None            # contorno anterior (coords. del recorte)
    fallas = 0
    for i, frame in leer_frames(cap, max_frames):
        if i in a_extraer:
            nombre = f"{stem}_frame_{i:04d}.png"
            cv2.imencode(".png", frame)[1].tofile(str(OUT / "frames" / nombre))

        if fallas >= FALLAS_REDETECTAR:            # ¿se perdió el ojo? -> re-detectar
            nuevo, _ = detectar_ojo(frame, verbose=False)
            # se acepta sólo si es coherente con la posición anterior (durante un
            # parpadeo el detector puede marcar otra cosa, p.ej. una fosa nasal)
            if nuevo is not None and np.hypot(nuevo.x - cand.x, nuevo.y - cand.y) < 3 * r_fijo:
                cand = Candidato(nuevo.x, nuevo.y, r_fijo, "re-detectado", nuevo.puntaje)
            fallas = 0
            contorno_prev = None

        crop, info = recortar_ojo(frame, cand=cand, verbose=False)
        if contorno_prev is None:
            # sin contorno previo (inicio, tras parpadeo o re-detección): re-centrar
            # en la mancha oscura más grande dentro del iris antes de segmentar
            cand = recentrar(crop, info, cand)
            crop, info = recortar_ojo(frame, cand=cand, verbose=False)
        gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask, t_multi = segmentar_frame(crop, contorno_prev)
        valido, motivo, med = calidad(mask, gris, t_multi)

        fila = dict(frame=i, t=i / fps, valido=int(valido), motivo=motivo,
                    roi_x=cand.x, roi_y=cand.y, circularidad=med.get("circularidad", np.nan),
                    cx=np.nan, cy=np.nan, d_pupila_px=np.nan, pupila_iris=np.nan,
                    d_area_iris=np.nan)
        if valido:
            fallas = 0
            cy_c, cx_c = ndi.center_of_mass(mask)
            # centro de la pupila en coords. de la imagen original
            ox = (cx_c - info["M"][0, 2]) / info["escala"]
            oy = (cy_c - info["M"][1, 2]) / info["escala"]
            # diámetro HORIZONTAL (ancho máximo por fila): los párpados tapan la
            # pupila desde arriba/abajo, así que el ancho es más robusto que el
            # diámetro equivalente al área cuando el ojo está entrecerrado
            sn = segmentar_frame.snake
            d_crop = ancho_horizontal(sn) if sn is not None else float(mask.sum(1).max())
            fila.update(cx=ox, cy=oy, d_pupila_px=d_crop / info["escala"],
                        pupila_iris=d_crop / (2 * R_IRIS_NORM),
                        d_area_iris=2 * med["r_eq"] / (2 * R_IRIS_NORM))
            # seguimiento suave del centro (evita temblores)
            cand_ant = cand
            cand = Candidato(0.7 * ox + 0.3 * cand.x, 0.7 * oy + 0.3 * cand.y, r_fijo, "seguimiento")
            cs = find_contours(mask.astype(float), 0.5)
            contorno_prev = max(cs, key=len) if cs else None
            # trasladar el contorno al recorte del próximo frame (centrado en el nuevo cand)
            if contorno_prev is not None:
                esc = info["escala"]
                contorno_prev = contorno_prev + esc * np.array([cand_ant.y - cand.y, cand_ant.x - cand.x])
        else:
            fallas += 1
            contorno_prev = None

        # Dice/IoU si este frame está anotado
        nombre = f"{stem}_frame_{i:04d}.png"
        if nombre in refs:
            ref = ref_en_recorte(refs[nombre], info)
            m_eval = mask if mask is not None else np.zeros_like(ref)
            metricas.append(dict(frame=i, dice=dice(m_eval, ref), iou=iou(m_eval, ref)))

        filas.append(fila)
        # overlay
        vis = cv2.resize(crop, (2 * LADO_RECORTE, 2 * LADO_RECORTE), interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            cs, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), vis.shape[1::-1],
                                                interpolation=cv2.INTER_NEAREST),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (0, 255, 0) if valido else (0, 0, 255), 2)
        txt = f"t={i / fps:5.2f}s  " + (f"d/iris={fila['pupila_iris']:.2f}" if valido else motivo[:24])
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        escritor.write(vis)
        if i % 30 == 0:
            print(f"  frame {i}/{total}  " + ("ok" if valido else motivo))

    cap.release()
    escritor.release()
    shutil.move(str(tmp_out), str(OUT / f"{stem}_overlay.mp4"))
    shutil.rmtree(tmp_out.parent, ignore_errors=True)
    if tmp:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    guardar_resultados(filas, metricas, stem)
    return filas, r_fijo


# ----------------------------------------------------------------------------
# Vínculo con segmentacion_pupila.py: análisis 2.1 / 2.2 sobre frames de trabajo
# ----------------------------------------------------------------------------
def elegir_frames(filas, pedidos):
    """pedidos: lista con números de frame y/o 'auto', 'dilatada', 'contraida'.
      auto      -> el frame más "limpio": válido, sin parpadeos en ±3 frames y
                   con la pupila más circular
      dilatada  -> el de mayor diámetro pupilar (entre los estables)
      contraida -> el de menor diámetro pupilar (entre los estables)"""
    ok = np.array([r["valido"] == 1 for r in filas])
    estables = [i for i in range(len(filas)) if ok[max(0, i - 3):i + 4].all()] or list(np.where(ok)[0])
    if not estables:
        raise RuntimeError("No hay frames válidos para analizar")
    d = {i: filas[i]["pupila_iris"] for i in estables}
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


def analizar_frames(path, filas, r_fijo, pedidos):
    stem = Path(path).stem
    elegidos = elegir_frames(filas, pedidos)
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    cap, tmp = abrir_video(path)
    filas_img = []
    for i, tipo in sorted(elegidos.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            print(f"  no se pudo leer el frame {i}"); continue
        archivo = OUT / "frames" / f"{stem}_frame_{i:04d}.png"
        cv2.imencode(".png", frame)[1].tofile(str(archivo))
        print(f"\n=== Frame {i} ({tipo}, t={i / (cap.get(cv2.CAP_PROP_FPS) or 30):.2f}s): "
              "módulos 2.1 y 2.2 ===")
        # mismo ROI que usó el video en ese frame (sin volver a detectar el ojo)
        cand = Candidato(filas[i]["roi_x"], filas[i]["roi_y"], r_fijo, f"video, frame {i}", 1.0)
        filas_img += SP.procesar(archivo, cand=cand)
    cap.release()
    if tmp:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    destino = SP.OUT / f"metricas_{stem}.csv"
    SP.guardar_metricas(filas_img, destino, mostrar=False)
    print("\nResumen 2.1 (Dice; nan = frame sin referencia anotada):")
    for f in filas_img:
        if f[1] == "2.1":
            print(f"  {f[0]:28s} {f[2]:30s} Dice={f[3]:.3f}")
    print(f"Resultados de imagen en {SP.OUT}")
    return elegidos


def marcar_atipicos(filas, ventana=9, tol=0.15):
    """Consistencia temporal: la pupila no cambia más de ~15 % entre frames
    vecinos (a 30 fps). Los valores que se apartan de la mediana móvil de los
    frames válidos (típicamente parpadeos parciales) se invalidan."""
    d = np.array([r["pupila_iris"] for r in filas], float)
    ok = np.where(~np.isnan(d))[0]
    for k, i in enumerate(ok):
        vec = d[ok[max(0, k - ventana // 2):k + ventana // 2 + 1]]
        med = np.median(vec)
        if abs(d[i] - med) > tol * med:
            filas[i].update(valido=0, motivo=f"atípico ({d[i]:.2f} vs {med:.2f})",
                            pupila_iris=np.nan, d_pupila_px=np.nan)


def guardar_resultados(filas, metricas, stem):
    marcar_atipicos(filas)
    with open(OUT / f"{stem}_serie.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)

    t = np.array([r["t"] for r in filas])
    d = np.array([r["pupila_iris"] for r in filas], float)
    ok = ~np.isnan(d)
    # interpolación de los frames inválidos (parpadeos) + mediana móvil
    d_int = np.interp(t, t[ok], d[ok]) if ok.sum() > 1 else d
    d_suave = medfilt(d_int, 7)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[ok], d[ok], ".", ms=3, color="gray", label="medido")
    ax.plot(t, d_suave, "-", color="C0", label="interpolado + mediana (7 frames)")
    for ti in t[~ok]:
        ax.axvspan(ti - 0.5 / 30, ti + 0.5 / 30, color="r", alpha=0.15, lw=0)
    ax.set_xlabel("tiempo [s]"); ax.set_ylabel("diámetro pupila / diámetro iris")
    ax.set_title(f"{stem}: {ok.sum()}/{len(d)} frames válidos (rojo = inválidos)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"{stem}_serie.png", dpi=130); plt.close(fig)

    print(f"\nFrames válidos: {ok.sum()}/{len(d)}")
    if ok.any():
        print(f"Pupila/iris: min={np.nanmin(d):.2f}  max={np.nanmax(d):.2f}")
    if metricas:
        print("\nFrames anotados:")
        for m in metricas:
            print(f"  frame {m['frame']:4d}  Dice={m['dice']:.3f}  IoU={m['iou']:.3f}")
        print(f"  promedio  Dice={np.mean([m['dice'] for m in metricas]):.3f}  "
              f"IoU={np.mean([m['iou'] for m in metricas]):.3f}")
    print(f"Resultados en {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="ruta del video; si se omite se usa el video que esté en la carpeta del script")
    ap.add_argument("--extraer", type=int, default=0, help="guardar N frames equiespaciados para anotar")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame", nargs="+", default=["auto"],
                    help="frame(s) para el análisis de imagen 2.1/2.2: número(s), 'auto', "
                         "'dilatada' y/o 'contraida' (por defecto: auto)")
    ap.add_argument("--extraer-por-defecto", type=int, default=8, help=argparse.SUPPRESS)
    a = ap.parse_args()
    video = a.video
    if video is None:
        # ejecutado sin argumentos (p.ej. con el botón "Run" de VS Code):
        # se busca el video en la carpeta del script
        videos = sorted(p for p in BASE.iterdir()
                        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"})
        if not videos:
            sys.exit(f"No hay ningún video en {BASE}. Copiá el video a esa carpeta o pasá su ruta:\n"
                     f"    python procesar_video.py mi_video.mp4")
        if len(videos) > 1:
            print("Hay varios videos en la carpeta; se usa el modificado más recientemente:")
            for p in videos:
                print(f"    {p.name}")
            videos.sort(key=lambda p: p.stat().st_mtime)
        video = videos[-1].name
        print(f"Video: {video}")
        if a.extraer == 0 and not (OUT / "frames").exists():
            a.extraer = a.extraer_por_defecto     # primera corrida: guardar frames para anotar
    filas, r_fijo = procesar_video(video, a.extraer, a.max_frames)
    analizar_frames(video, filas, r_fijo, a.frame)
