"""
Anotación manual del contorno real de la pupila (ground truth).

Uso:
    python anotar_referencia.py contraida.png dilatada.png ...
Se anota sobre el RECORTE del ojo (el mismo que usa segmentacion_pupila.py),
así las coordenadas coinciden. Por cada imagen: clic izquierdo = agregar punto sobre el borde de la pupila,
clic derecho = borrar el último, Enter = terminar. Conviene marcar 15+ puntos
(con menos de 15 se ajusta una elipse en vez de usar el polígono).
Los puntos se guardan en referencias.json en coordenadas (x, y) de la imagen
ORIGINAL, así siguen valiendo aunque cambie el algoritmo de recorte.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentacion_pupila import leer_bgr  # noqa: E402
from roi_ojo import recortar_ojo, a_original  # noqa: E402

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "referencias.json"


def anotar(path, zoom_px=45):
    crop, info = recortar_ojo(leer_bgr(path), manual_si_falla=True)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, _, (x, y), _ = cv2.minMaxLoc(cv2.GaussianBlur(gris, (0, 0), 4))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xlim(x - zoom_px, x + zoom_px); ax.set_ylim(y + zoom_px, y - zoom_px)
    ax.set_title(f"{Path(path).name}: marcá el borde de la pupila y apretá Enter")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) == 0:
        return []
    # se guardan en coordenadas de la imagen ORIGINAL (independientes del recorte)
    return [[round(float(px), 1), round(float(py), 1)] for px, py in a_original(pts, info)]


if __name__ == "__main__":
    datos = json.loads(ARCHIVO.read_text(encoding="utf-8")) if ARCHIVO.exists() else {}
    for p in sys.argv[1:] or ["contraida.png", "dilatada.png"]:
        pts = anotar(p)
        if len(pts) >= 5:
            datos[Path(p).name] = pts
            print(f"{p}: {len(pts)} puntos guardados")
        else:
            print(f"{p}: se necesitan al menos 5 puntos, no se guardó")
    datos.pop("_nota", None)
    ARCHIVO.write_text(json.dumps(datos, indent=2), encoding="utf-8")
