"""
ROI del ojo robusto a distintos tipos de imagen de entrada.

Casos contemplados
  - cara completa (dos ojos)            -> cara -> ojos dentro de la mitad superior
  - media cara / foto de perfil parcial -> cascada de ojos sobre toda la imagen
  - primer plano del ojo (eye close-up) -> el detector Haar falla porque el ojo
                                           "no entra" en la ventana: se rellena
                                           la imagen con un borde antes de detectar
  - imágenes grandes / chicas, cabeza inclinada (hasta ~±30°), sobre o
    sub-expuestas (CLAHE), con ruido
  - imágenes SIN ojo: se rechazan (umbral de confianza) en vez de devolver
    cualquier cosa

Idea: el clasificador de Viola-Jones PROPONE candidatos y una función de
puntaje los VALIDA buscando lo que tiene todo ojo: un iris circular
con una pupila más oscura en el centro. Al final se normaliza la escala usando
el radio del iris (no el tamaño del recuadro del detector), para que los
parámetros de la segmentación posterior sirvan para cualquier imagen.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

LADO_TRABAJO = 800      # se reduce la imagen a este lado máximo para detectar
R_IRIS_NORM = 32        # radio del iris en el recorte final (px)
LADO_RECORTE = 224      # lado del recorte final (px) -> ojo ≈ 160 px de ancho
UMBRAL_CONFIANZA = 0.62   # por debajo -> "no encontré un ojo"

_cascadas = {}


def _cascada(nombre):
    if nombre not in _cascadas:
        _cascadas[nombre] = cv2.CascadeClassifier(cv2.data.haarcascades + nombre)
    return _cascadas[nombre]


@dataclass
class Candidato:
    x: float          # centro estimado del iris (coords. de la imagen de trabajo)
    y: float
    r: float          # radio estimado del iris
    origen: str
    puntaje: float = 0.0
    detalle: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Preprocesamiento
# ----------------------------------------------------------------------------
def _gris_normalizado(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE: compensa sub/sobre-exposición e iluminación no uniforme
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


# ----------------------------------------------------------------------------
# Estrategias que proponen candidatos
# ----------------------------------------------------------------------------
def _haar(gris, cascada, escala_min, vecinos=5):
    """detectMultiScale3 devuelve además la "confianza" de cada detección
    (levelWeights): cuánto superó la última etapa de la cascada."""
    cajas, _, pesos = _cascada(cascada).detectMultiScale3(
        gris, 1.05, vecinos, minSize=(escala_min, escala_min), outputRejectLevels=True)
    return [(tuple(int(v) for v in c), float(w)) for c, w in zip(cajas, np.ravel(pesos))] if len(cajas) else []


def candidatos_haar(gris):
    """Cascadas de Viola-Jones. La imagen se rellena con un borde (50 % del lado)
    para que un primer plano del ojo, que ocupa toda la imagen, también entre en
    la ventana del detector."""
    H, W = gris.shape
    pad = int(0.5 * max(H, W))
    g = cv2.copyMakeBorder(gris, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    por_cascada = {}
    for c in ["haarcascade_eye.xml", "haarcascade_eye_tree_eyeglasses.xml"]:
        por_cascada[c] = [Candidato(x + w / 2 - pad, y + h / 2 - pad, 0.18 * w, "haar", detalle=dict(peso=p, ancho=w))
                          for (x, y, w, h), p in _haar(g, c, int(0.05 * min(H, W)) + 10)]
        # en un recuadro de ojo de Haar el iris tiene radio ≈ 0.18 del ancho
    out = []
    a, b = por_cascada.values()
    for propios, otros in [(a, b), (b, a)]:
        for c in propios:
            # consenso: ¿la otra cascada también detectó un ojo en el mismo lugar?
            conf = [o for o in otros if np.hypot(o.x - c.x, o.y - c.y) < 0.5 * c.detalle["ancho"]]
            c.detalle["consenso"] = 1.0 if conf else 0.0
            c.detalle["peso"] = max([c.detalle["peso"]] + [o.detalle["peso"] for o in conf])
            out.append(c)
    return [c for c in out if 0 <= c.x < W and 0 <= c.y < H]


# ----------------------------------------------------------------------------
# Validación: ¿hay un iris con una pupila oscura en el centro?
# ----------------------------------------------------------------------------
def _muestrear(img, cx, cy, radios, angulos):
    """Valores de img sobre puntos en coordenadas polares (interpolación bilineal)."""
    rr, aa = np.meshgrid(radios, angulos, indexing="ij")
    mx = (cx + rr * np.cos(aa)).astype(np.float32)
    my = (cy + rr * np.sin(aa)).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def refinar_y_puntuar(gris, hsv, c):
    """Busca el iris (círculo) cerca del candidato y lo puntúa con rasgos que
    tiene un ojo y no tienen el pelo, las fosas nasales o el fondo:
      1. pupila más oscura que el iris
      2. borde del iris: transición oscuro->claro a lo largo del círculo
      3. esclerótica: zonas claras y poco saturadas a los costados del iris
      4. pupila centrada en el iris
      5. el círculo fue confirmado por Hough
    (además, en _validar: piel alrededor del ojo, sólo en imágenes color)
    Puntaje final en [0, 1]."""
    H, W = gris.shape
    R = 2.5 * c.r
    x0, y0 = int(max(0, c.x - R)), int(max(0, c.y - R))
    x1, y1 = int(min(W, c.x + R)), int(min(H, c.y + R))
    sub = gris[y0:y1, x0:x1]
    if min(sub.shape) < 12:
        return None
    # (a) semilla de pupila: punto más oscuro (muy suavizado) cerca del centro
    sb = cv2.GaussianBlur(sub, (0, 0), max(1.0, c.r / 4)).astype(np.float32)
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    dist = np.hypot(xx + x0 - c.x, yy + y0 - c.y)
    sb[dist > 0.8 * c.r] = 1e6
    py, px = np.unravel_index(np.argmin(sb), sb.shape)
    sx, sy = float(px + x0), float(py + y0)
    # (b) iris = círculo de Hough cuyo centro coincide con la pupila
    circ = cv2.HoughCircles(cv2.medianBlur(sub, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=1,
                            param1=80, param2=15,
                            minRadius=int(0.6 * c.r), maxRadius=int(1.5 * c.r) + 1)
    cerca = [] if circ is None else [q for q in circ[0] if np.hypot(q[0] + x0 - sx, q[1] + y0 - sy) < 0.35 * q[2]]
    if cerca:
        q = min(cerca, key=lambda q: np.hypot(q[0] + x0 - sx, q[1] + y0 - sy))
        cx, cy, r, hough_ok = float(q[0] + x0), float(q[1] + y0), float(q[2]), 1.0
    else:
        cx, cy, r, hough_ok = sx, sy, c.r, 0.0
    if r < 4:
        return None

    s = cv2.GaussianBlur(gris, (0, 0), max(1.0, r / 10)).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    # 1. contraste pupila / iris
    pupila = _muestrear(s, cx, cy, np.linspace(0, 0.3 * r, 4), ang).mean()
    iris = _muestrear(s, cx, cy, np.linspace(0.5 * r, 0.85 * r, 4), ang).mean()
    c1 = np.clip((iris - pupila) / 30.0, 0, 1)
    # 2. borde del iris (más claro afuera que adentro en buena parte del círculo;
    #    arriba/abajo lo pueden tapar los párpados, por eso basta ~40 %)
    dentro = _muestrear(s, cx, cy, [0.8 * r], ang)[0]
    fuera = _muestrear(s, cx, cy, [1.25 * r], ang)[0]
    c2 = np.clip(np.mean(fuera - dentro > 10) / 0.4, 0, 1)
    # 3. esclerótica a izquierda y derecha: clara (V alto) y poco saturada
    lados = np.r_[np.linspace(-0.5, 0.5, 9), np.pi + np.linspace(-0.5, 0.5, 9)]
    rad = np.linspace(1.2 * r, 1.7 * r, 4)
    V = _muestrear(hsv[..., 2].astype(np.float32), cx, cy, rad, lados)
    S = _muestrear(hsv[..., 1].astype(np.float32), cx, cy, rad, lados)
    iz, de = V[:, :9].mean(), V[:, 9:].mean()
    sat = min(S[:, :9].mean(), S[:, 9:].mean()) if iz > de else S[:, 9:].mean()
    v_iris = _muestrear(hsv[..., 2].astype(np.float32), cx, cy, np.linspace(0.5 * r, 0.85 * r, 3), ang).mean()
    c3 = np.clip((max(iz, de) - v_iris) / 40.0, 0, 1) * np.clip((160 - sat) / 80.0, 0, 1)
    # 4. pupila centrada: mínimo dentro de 0.6 r
    rr = max(2, int(0.6 * r))
    win = s[int(max(0, cy - rr)):int(cy + rr) + 1, int(max(0, cx - rr)):int(cx + rr) + 1]
    oy, ox = np.unravel_index(np.argmin(win), win.shape)
    desc = np.hypot(ox + max(0, cy - rr) * 0 + int(max(0, cx - rr)) - cx, oy + int(max(0, cy - rr)) - cy) / r
    c4 = np.clip(1 - desc / 0.6, 0, 1)
    puntaje = 0.30 * c1 + 0.25 * c2 + 0.25 * c3 + 0.10 * c4 + 0.10 * hough_ok
    return Candidato(cx, cy, float(r), c.origen, float(puntaje),
                     dict(pupila=round(float(c1), 2), borde=round(float(c2), 2), esclerotica=round(float(c3), 2),
                          centrado=round(float(c4), 2), hough=hough_ok))


def refinar_pupila_iris(gris, c):
    """Ajuste fino del candidato elegido:
      - centro = centroide de la pupila (umbral de Otsu dentro del iris)
      - radio del iris = máximo del gradiente radial (oscuro->claro) mirando
        sólo a izquierda y derecha, donde los párpados no tapan el limbo."""
    H, W = gris.shape
    s = cv2.GaussianBlur(gris, (0, 0), max(1.0, c.r / 15)).astype(np.float32)
    R = int(1.0 * c.r)
    x0, y0 = int(max(0, c.x - R)), int(max(0, c.y - R))
    win = s[y0:int(min(H, c.y + R + 1)), x0:int(min(W, c.x + R + 1))]
    yy, xx = np.mgrid[0:win.shape[0], 0:win.shape[1]]
    disco = np.hypot(xx + x0 - c.x, yy + y0 - c.y) <= 0.9 * c.r
    if disco.sum() > 20:
        v = win[disco].astype(np.uint8)
        t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, lab, st, cen = cv2.connectedComponentsWithStats(((win <= t) & disco).astype(np.uint8))
        if n > 1:
            # componente oscura grande y cercana al centro
            d = [np.hypot(cen[i][0] + x0 - c.x, cen[i][1] + y0 - c.y) / c.r - st[i, 4] / disco.sum()
                 for i in range(1, n)]
            i = 1 + int(np.argmin(d))
            px, py = cen[i][0] + x0, cen[i][1] + y0
            rp = np.sqrt(st[i, 4] / np.pi)
            if np.hypot(px - c.x, py - c.y) < 0.6 * c.r:
                c.x, c.y = float(px), float(py)
                c.detalle["r_pupila"] = float(rp)
    rp = c.detalle.get("r_pupila", 0.3 * c.r)
    radios = np.linspace(max(1.2 * rp, 0.5 * c.r), 1.8 * c.r, 40)
    lados = np.r_[np.linspace(-0.45, 0.45, 9), np.pi + np.linspace(-0.45, 0.45, 9)]
    perfil = _muestrear(s, c.x, c.y, radios, lados)          # (radios, ángulos)
    grad = np.diff(perfil, axis=0)
    r_izq = radios[np.argmax(grad[:, 9:].mean(1))]
    r_der = radios[np.argmax(grad[:, :9].mean(1))]
    r_iris = float(np.median([r_izq, r_der, c.r]))
    c.r = r_iris
    return c


def _sin_duplicados(cands, tol=0.5):
    cands = sorted(cands, key=lambda c: -c.puntaje)
    out = []
    for c in cands:
        if all(np.hypot(c.x - o.x, c.y - o.y) > tol * max(c.r, o.r) for o in out):
            out.append(c)
    return out


# ----------------------------------------------------------------------------
# API principal
# ----------------------------------------------------------------------------
def _rotar(img, ang):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE), cv2.invertAffineTransform(M)


def es_color(bgr):
    """Las cámaras de pupilometría suelen ser infrarrojas (imagen gris)."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()) > 15


def fraccion_piel(bgr, c):
    """Fracción de piel (rango clásico en YCrCb, válido para distintos tonos)
    en un anillo alrededor del iris: un ojo siempre está rodeado de párpados,
    mejilla o frente; el público de fondo, una pelota o una ventana no."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rad = np.linspace(2.0 * c.r, 3.2 * c.r, 4)
    Cr = _muestrear(ycc[..., 1], c.x, c.y, rad, ang)
    Cb = _muestrear(ycc[..., 2], c.x, c.y, rad, ang)
    return float(np.mean((Cr >= 133) & (Cr <= 180) & (Cb >= 77) & (Cb <= 135)))


def _validar(trabajo, propuestas):
    gris = _gris_normalizado(trabajo)
    hsv = cv2.cvtColor(trabajo, cv2.COLOR_BGR2HSV)
    color = es_color(trabajo)
    out = []
    for c in propuestas:
        v = refinar_y_puntuar(gris, hsv, c)
        if v is not None:
            v.detalle.update(c.detalle)
            v.detalle["piel"] = round(fraccion_piel(trabajo, v), 2) if color else None
            out.append(v)
    return out


def detectar_ojo(bgr, manual_si_falla=False, verbose=True):
    """Estrategia jerárquica:
      1. Haar (clasificador entrenado, el más confiable para decir "esto es un
         ojo") sobre la imagen; si no aparece nada válido, sobre la imagen
         rotada ±20° (cabeza inclinada).
      2. Si nada supera el umbral de confianza -> NO se inventa un ojo:
         se avisa y, si se pide, se ofrece selección manual del iris.
    Cada candidato se valida con rasgos de ojo (pupila, borde de iris,
    esclerótica) y se combina con la confianza de la cascada."""
    H, W = bgr.shape[:2]
    f = min(1.0, LADO_TRABAJO / max(H, W))
    trabajo = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else bgr
    gris = _gris_normalizado(trabajo)

    validados = []
    for ang in [0, 20, -20]:
        img, Minv = (trabajo, None) if ang == 0 else _rotar(trabajo, ang)
        v = _validar(img, candidatos_haar(_gris_normalizado(img)))
        if Minv is not None:
            for c in v:
                c.x, c.y = (Minv @ np.array([c.x, c.y, 1.0])).tolist()
                c.origen = f"haar rot{ang:+d}"
        for c in v:
            # 50 % rasgos de ojo + 50 % confianza del clasificador: peso ABSOLUTO
            # de la cascada (ojos reales > 2.5, falsos positivos < 1.5) y consenso
            # entre las dos cascadas
            c.detalle["conf_haar"] = float(0.6 * np.clip((c.detalle["peso"] - 1.0) / 2.5, 0, 1)
                                           + 0.4 * c.detalle["consenso"])
            c.puntaje = 0.5 * c.puntaje + 0.5 * c.detalle["conf_haar"]
            if c.detalle["piel"] is not None:      # sólo en imágenes color
                c.puntaje *= 0.5 + 0.5 * c.detalle["piel"]
        validados += [c for c in v if c.detalle["pupila"] > 0.3 and c.r * 1 / f >= 5]  # iris ≥ 5 px
        if any(c.puntaje >= UMBRAL_CONFIANZA for c in validados):
            break

    validados = _sin_duplicados(validados)
    if validados:
        refinar_pupila_iris(gris, validados[0])
    for v in validados:          # volver a coordenadas de la imagen original
        v.x, v.y, v.r = v.x / f, v.y / f, v.r / f

    mejor = validados[0] if validados else None
    if mejor is None or mejor.puntaje < UMBRAL_CONFIANZA:
        if verbose:
            p = 0 if mejor is None else mejor.puntaje
            print(f"  [aviso] no se encontró un ojo con confianza suficiente (puntaje={p:.2f})")
        mejor = seleccion_manual(bgr) if manual_si_falla else None
    if verbose and mejor is not None and mejor.r < 15:
        print(f"  [aviso] iris de sólo {mejor.r:.0f} px de radio: la imagen tiene poca "
              "resolución en el ojo y la segmentación va a ser menos precisa")
    if verbose and mejor is not None:
        print(f"  ojo: centro=({mejor.x:.0f},{mejor.y:.0f}) r_iris={mejor.r:.1f}px "
              f"origen={mejor.origen} puntaje={mejor.puntaje:.2f}")
    return mejor, validados


def seleccion_manual(bgr):
    """Respaldo interactivo: el usuario encierra el iris con un rectángulo."""
    print("  Seleccioná el IRIS con el mouse y apretá Enter (Esc = cancelar)")
    try:
        H, W = bgr.shape[:2]
        f = min(1.0, 900 / max(H, W))           # que la ventana entre en pantalla
        vista = cv2.resize(bgr, None, fx=f, fy=f) if f < 1 else bgr
        x, y, w, h = [v / f for v in cv2.selectROI("Seleccionar iris", vista, showCrosshair=True)]
        cv2.destroyWindow("Seleccionar iris")
    except cv2.error:
        print("  (no hay entorno gráfico para la selección manual)")
        return None
    if w == 0 or h == 0:
        return None
    return Candidato(x + w / 2, y + h / 2, (w + h) / 4, "manual", 1.0)


def recortar_ojo(bgr, cand=None, manual_si_falla=False, verbose=True):
    """Recorte cuadrado centrado en la pupila con el iris a escala fija
    (radio R_IRIS_NORM). Si se pasa `cand` (p.ej. del frame anterior en un
    video) no se vuelve a detectar."""
    validados = []
    if cand is None:
        cand, validados = detectar_ojo(bgr, manual_si_falla, verbose)
    if cand is None:
        raise RuntimeError("No se pudo ubicar el ojo en la imagen")
    esc = R_IRIS_NORM / cand.r
    # transformación afín: escala + traslación para que el centro quede en el medio
    M = np.float32([[esc, 0, LADO_RECORTE / 2 - esc * cand.x],
                    [0, esc, LADO_RECORTE / 2 - esc * cand.y]])
    crop = cv2.warpAffine(bgr, M, (LADO_RECORTE, LADO_RECORTE),
                          flags=cv2.INTER_CUBIC if esc > 1 else cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)
    lado = LADO_RECORTE / esc
    info = dict(cand=cand, candidatos=validados, escala=esc, M=M,
                x0=int(cand.x - lado / 2), y0=int(cand.y - lado / 2), lado=int(lado))
    return crop, info


def a_recorte(pts, info):
    """Puntos (x, y) de la imagen original -> coordenadas del recorte."""
    p = np.asarray(pts, np.float64)
    return p @ info["M"][:, :2].T + info["M"][:, 2]


def a_original(pts, info):
    """Puntos (x, y) del recorte -> coordenadas de la imagen original."""
    p = np.asarray(pts, np.float64)
    return (p - info["M"][:, 2]) / info["escala"]
