"""
Módulo 3 — Extracción de características, textura y dinámica temporal.

3.1 a 3.3 trabajan sobre un frame y parten de la máscara binaria de la pupila.
3.4 a 3.6 trabajan sobre el video completo y parten de la salida del módulo 2:
    filas, r_iris_px = ejecutar_modulo2(ruta_video)        # módulo 2
    curva = curva_desde_modulo2(filas, ruta_video, r_iris_px)
    curva = limpiar_curva(curva)
    t_estimulo, tipo = detectar_estimulo(curva)
    parametros = parametros_fisiologicos(curva, t_estimulo, tipo,
                                         mm_por_px=mm_por_px_desde_iris(r_iris_px))
    guardar_resultados(curva, parametros, carpeta)
"""

import os                        # Rutas de los archivos de salida
import cv2                       # OpenCV: componentes conexas, contornos y momentos
import numpy as np               # Cálculo numérico (arrays, histogramas, estadística)
import pandas as pd              # Tablas: curva temporal y exportación a CSV
import matplotlib.pyplot as plt  # Gráficos
from matplotlib.patches import Ellipse          # Elipse equivalente sobre la pupila
from scipy.ndimage import binary_dilation       # Ensanchar en el tiempo los tramos con artefactos
from scipy.signal import savgol_filter          # Suavizado de Savitzky-Golay de la curva
from skimage.feature import graycomatrix, graycoprops  # Matriz de co-ocurrencia (GLCM) y sus propiedades
from skimage.measure import perimeter_crofton          # Perímetro de una región binaria

from . import modulo0_importacion as m0


# ------------------------------------------------------------
# 3.1 Descriptores morfométricos
# ------------------------------------------------------------

def componente_principal(mascara):
    """
    Se queda con la región más grande de la máscara y le rellena los huecos, así el
    reflejo corneal queda como parte de la pupila.

    Entrada: mascara (array 2D, la pupila es > 0)
    Salida: array uint8 con 1 en la pupila, o None si la máscara está vacía
    """
    mascara = (mascara > 0).astype(np.uint8)
    # Etiqueta cada región conectada (vecindad 8) y calcula su área; la etiqueta 0 es el fondo
    cantidad, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if cantidad <= 1:
        return None
    mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])   # Se saltea el fondo
    componente = (etiquetas == mayor).astype(np.uint8)

    # Se rellena el contorno exterior: todo lo que queda adentro pasa a ser pupila
    contornos, _ = cv2.findContours(componente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(componente, contornos, -1, 1, thickness=cv2.FILLED)
    return componente


def descriptores_morfometricos(mascara):
    """
    Calcula los descriptores de forma de la pupila con los momentos de la región.

    Entrada: mascara binaria de la pupila
    Salida: diccionario con area, perimetro, diametro (círculo de igual área), centroide_x,
            centroide_y, eje_mayor, eje_menor, orientacion_grados, circularidad (4πA/P², 1 = círculo)
            y excentricidad (0 = círculo). Todo en píxeles; NaN si la máscara está vacía
    """
    claves = ["area", "perimetro", "diametro", "centroide_x", "centroide_y", "eje_mayor",
              "eje_menor", "orientacion_grados", "circularidad", "excentricidad"]
    componente = None if mascara is None else componente_principal(mascara)
    if componente is None:
        return dict.fromkeys(claves, np.nan)

    # Momentos de la región: m00 es el área (cantidad de píxeles), m10/m00 y m01/m00 el centroide,
    # y los momentos centrales mu20, mu02, mu11 describen cómo se reparte el área alrededor de él
    momentos = cv2.moments(componente, binaryImage=True)
    area = momentos["m00"]
    cx, cy = momentos["m10"] / area, momentos["m01"] / area

    # Perímetro por la fórmula de Crofton: cuenta cuántas veces cortan al borde rectas en
    # 4 direcciones. Sumar los pasos del contorno píxel a píxel lo sobreestima por el
    # "escalonado" del borde (un círculo daría circularidad ≈ 0,9 en vez de 1)
    perimetro = perimeter_crofton(componente, directions=4)

    # Matriz de covarianza de las coordenadas de los píxeles. Sus autovalores son las varianzas
    # a lo largo de los ejes principales; para una elipse llena, eje = 4·√λ
    mu20, mu02, mu11 = momentos["mu20"] / area, momentos["mu02"] / area, momentos["mu11"] / area
    diferencia = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    lambda_1 = (mu20 + mu02) / 2 + diferencia
    lambda_2 = max((mu20 + mu02) / 2 - diferencia, 0.0)   # Evita negativos por redondeo
    # Ángulo del eje mayor respecto del eje x de la imagen (la y crece hacia abajo)
    orientacion = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

    return {
        "area": area,
        "perimetro": perimetro,
        "diametro": np.sqrt(4 * area / np.pi),          # Diámetro del círculo de igual área
        "centroide_x": cx,
        "centroide_y": cy,
        "eje_mayor": 4 * np.sqrt(lambda_1),
        "eje_menor": 4 * np.sqrt(lambda_2),
        "orientacion_grados": orientacion,
        # En regiones de pocos píxeles la discretización la aleja de 1 aunque sean círculos
        "circularidad": 4 * np.pi * area / perimetro ** 2 if perimetro > 0 else np.nan,
        "excentricidad": np.sqrt(1 - lambda_2 / lambda_1) if lambda_1 > 0 else np.nan,
    }


def imprimir_descriptores(descriptores, mm_por_px=None):
    """
    Imprime los descriptores por consola.

    Entrada: descriptores (dict de descriptores_morfometricos), mm_por_px (opcional, para verlos en mm)
    Salida: nada
    """
    d = descriptores
    if mm_por_px is None:
        escala, unidad = 1.0, "px"
    else:
        escala, unidad = mm_por_px, "mm"
    print("─" * 45)
    print(f"Área             : {d['area'] * escala ** 2:.2f} {unidad}²")
    print(f"Perímetro        : {d['perimetro'] * escala:.2f} {unidad}")
    print(f"Diámetro equiv.  : {d['diametro'] * escala:.2f} {unidad}")
    print(f"Centroide (x, y) : ({d['centroide_x']:.1f}, {d['centroide_y']:.1f}) px")
    print(f"Ejes (may / men) : {d['eje_mayor'] * escala:.2f} / {d['eje_menor'] * escala:.2f} {unidad}")
    print(f"Orientación      : {d['orientacion_grados']:.1f}°")
    print(f"Circularidad     : {d['circularidad']:.3f}")
    print(f"Excentricidad    : {d['excentricidad']:.3f}")
    print("─" * 45)


# ------------------------------------------------------------
# 3.2 Textura de primer orden (histograma)
# ------------------------------------------------------------

def textura_primer_orden(canal, mascara=None):
    """
    Estadísticos de primer orden calculados con el histograma normalizado de un canal.

    Entrada: canal (imagen 2D de 8 bits), mascara (opcional, para usar solo esos píxeles)
    Salida: diccionario con media, desvio, asimetria, curtosis, suavidad, energia y entropia
    """
    valores = canal[mascara > 0] if mascara is not None else canal.ravel()
    niveles = np.arange(256)
    p = np.bincount(valores.astype(np.uint8), minlength=256) / valores.size

    media = np.sum(niveles * p)
    varianza = np.sum((niveles - media) ** 2 * p)
    desvio = np.sqrt(varianza)
    p_no_nulo = p[p > 0]   # log2(0) no está definido; esos niveles no aportan a la entropía
    return {
        "media": media,
        "desvio": desvio,
        # Asimetría: signo de hacia dónde se estira el histograma (+ hacia los claros)
        "asimetria": np.sum((niveles - media) ** 3 * p) / desvio ** 3 if desvio > 0 else 0.0,
        # Curtosis (exceso): > 0 histograma más puntiagudo que una normal, < 0 más aplanado
        "curtosis": np.sum((niveles - media) ** 4 * p) / desvio ** 4 - 3 if desvio > 0 else 0.0,
        # Suavidad R = 1 - 1/(1 + σ²), con σ² normalizada a [0, 1]: 0 en regiones uniformes
        "suavidad": 1 - 1 / (1 + varianza / 255 ** 2),
        # Energía (uniformidad): máxima (1) cuando hay un único nivel
        "energia": np.sum(p ** 2),
        # Entropía en bits: Σ p log2(1/p), máxima (8) cuando todos los niveles son igual de probables
        "entropia": np.sum(p_no_nulo * np.log2(1 / p_no_nulo)),
    }


def textura_por_canal(canales, mascara=None):
    """
    Aplica textura_primer_orden a cada canal.

    Entrada: canales (dict {nombre: imagen 2D}, por ejemplo {"R": roi[..., 0], ...}), mascara (opcional)
    Salida: DataFrame con una fila por canal
    """
    filas = {nombre: textura_primer_orden(canal, mascara) for nombre, canal in canales.items()}
    return pd.DataFrame(filas).T   # .T para que cada canal quede como una fila


# ------------------------------------------------------------
# 3.3 Textura de segundo orden (GLCM)
# ------------------------------------------------------------

def matriz_coocurrencia(gris, niveles=32, distancia=1, angulos=(0, 45, 90, 135), mascara=None):
    """
    Arma la GLCM normalizada para cada ángulo, con la imagen recuantizada a `niveles`
    niveles. Con máscara solo cuenta los pares de píxeles que están los dos adentro.

    Entrada: gris (imagen 2D de 8 bits), niveles, distancia (px), angulos (grados), mascara (opcional)
    Salida: array (niveles, niveles, cantidad de ángulos); cada matriz suma 1
    """
    # Cuantización uniforme a `niveles` niveles (de 0 a niveles - 1)
    cuantizada = (gris.astype(np.uint16) * niveles // 256).astype(np.uint8)
    if mascara is not None:
        # Se corre todo un nivel hacia arriba y se reserva el 0 para "fuera de la máscara";
        # después se descartan la fila y la columna 0, que son los pares que tocan el exterior
        cuantizada = np.where(mascara > 0, cuantizada + 1, 0).astype(np.uint8)

    P = graycomatrix(cuantizada, distances=[distancia], angles=np.radians(angulos),
                     levels=niveles + (mascara is not None), symmetric=True, normed=False)
    P = P[:, :, 0, :].astype(np.float64)   # Se quita el eje de distancias (hay una sola)
    if mascara is not None:
        P = P[1:, 1:, :]
    # Normalización: cada matriz suma 1
    return P / np.maximum(P.sum(axis=(0, 1), keepdims=True), 1)


def textura_glcm(gris, niveles=32, distancia=1, angulos=(0, 45, 90, 135), mascara=None):
    """
    Descriptores de la GLCM para cada ángulo y el promedio entre ángulos.

    Entrada: los mismos que matriz_coocurrencia
    Salida: DataFrame con contraste, homogeneidad, energia, correlacion y entropia;
            una fila por ángulo más la fila "promedio"
    """
    P = matriz_coocurrencia(gris, niveles, distancia, angulos, mascara)
    # graycoprops espera la forma (niveles, niveles, distancias, ángulos)
    P4 = P[:, :, np.newaxis, :]
    tabla = pd.DataFrame({
        # Contraste: Σ (i - j)² P. Grande si los vecinos suelen tener niveles muy distintos
        "contraste": graycoprops(P4, "contrast")[0],
        # Homogeneidad: Σ P / (1 + (i - j)²). Grande si la matriz se concentra en la diagonal
        "homogeneidad": graycoprops(P4, "homogeneity")[0],
        # Energía: √(Σ P²). Grande si hay pocos pares distintos (textura regular)
        "energia": graycoprops(P4, "energy")[0],
        # Correlación: dependencia lineal entre el nivel de un píxel y el de su vecino
        "correlacion": graycoprops(P4, "correlation")[0],
        # Entropía: Σ P log2(1/P). Grande si los pares están repartidos (textura desordenada)
        "entropia": [np.sum(p[p > 0] * np.log2(1 / p[p > 0])) for p in np.moveaxis(P, 2, 0)],
    }, index=[f"{a}°" for a in angulos])
    tabla.loc["promedio"] = tabla.mean()
    return tabla


def textura_glcm_por_canal(canales, niveles=32, distancia=1, angulos=(0, 45, 90, 135), mascara=None):
    """
    Descriptores de la GLCM (promedio entre ángulos) de cada canal, para comparar cuál
    tiene más textura.

    Entrada: canales (dict {nombre: imagen 2D}) y los mismos parámetros que textura_glcm
    Salida: DataFrame con una fila por canal
    """
    filas = {nombre: textura_glcm(canal, niveles, distancia, angulos, mascara).loc["promedio"]
             for nombre, canal in canales.items()}
    return pd.DataFrame(filas).T


def mostrar_glcm(gris, niveles=32, distancia=1, angulos=(0, 45, 90, 135), mascara=None, titulo=None):
    """
    Grafica la GLCM de cada ángulo en escala logarítmica.

    Entrada: los mismos que matriz_coocurrencia, titulo (opcional)
    Salida: nada (muestra la figura)
    """
    P = matriz_coocurrencia(gris, niveles, distancia, angulos, mascara)
    fig, ejes = plt.subplots(1, len(angulos), figsize=(3.6 * len(angulos), 3.8),
                             squeeze=False, layout="constrained")
    for k, (eje, angulo) in enumerate(zip(ejes[0], angulos)):
        # Se suma 1e-6 porque log(0) no está definido
        im = eje.imshow(np.log10(P[:, :, k] + 1e-6), cmap="viridis", vmin=-6, vmax=0)
        eje.set_title(f"{angulo}°, d = {distancia}")
        eje.set_xlabel("Nivel j")
        eje.set_ylabel("Nivel i")
    fig.colorbar(im, ax=ejes[0].tolist(), label="log₁₀ P(i, j)", fraction=0.02)
    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    plt.show()


# ------------------------------------------------------------
# 3.4 Curva temporal a partir de la segmentación frame a frame (módulo 2)
# ------------------------------------------------------------

# Columnas de la curva temporal, en orden fijo
COLUMNAS_CURVA = ["frame", "t_s", "brillo", "diametro", "area", "centroide_x", "centroide_y",
                  "circularidad", "motivo_invalido"]

# Diámetro horizontal del iris visible en adultos (HVID): ~11,7 mm, con poca variación
# entre personas (aprox. 10,7 a 12,6 mm)
DIAMETRO_IRIS_MM = 11.7


def brillo_por_frame(ruta_video, frames=None):
    """
    Brillo medio de cada frame (en grises). Se usa para encontrar el estímulo.

    Entrada: ruta_video, frames (índices a leer; None = todos)
    Salida: diccionario {índice de frame: brillo}
    """
    video, metadatos = m0.abrir_video(ruta_video)
    pedidos = set(range(metadatos["cantidad_frames"])) if frames is None else {int(f) for f in frames}
    brillo = {}
    for indice in range(max(pedidos, default=-1) + 1):
        # Los frames se leen en orden (es mucho más rápido que posicionarse en cada uno).
        # grab() avanza sin convertir la imagen, para los frames que no se piden
        if indice not in pedidos:
            if not video.grab():
                break
            continue
        ok, frame = video.read()
        if not ok:
            break  # El encabezado puede informar más frames de los que hay realmente
        # Sobre una versión achicada: alcanza para el promedio y es mucho más rápido
        chico = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        brillo[indice] = cv2.cvtColor(chico, cv2.COLOR_BGR2GRAY).mean()
    video.release()
    return brillo


def curva_desde_modulo2(filas, ruta_video=None, r_iris_px=None):
    """
    Pasa la tabla frame a frame del módulo 2 al formato de curva de este módulo y le
    agrega el brillo de cada frame. El diámetro es el ancho horizontal de la pupila que mide
    el módulo 2 (los párpados casi no lo afectan). Los frames que descartó quedan en NaN.

    Entrada: filas (lo que devuelve ejecutar_modulo2 del módulo 2, o su CSV _serie.csv),
             ruta_video (opcional, para el brillo), r_iris_px (radio del iris del módulo 2;
             si no se pasa se intenta calcular de la tabla)
    Salida: DataFrame con las columnas de COLUMNAS_CURVA
    """
    serie = pd.DataFrame(filas).reset_index(drop=True)
    if r_iris_px is None:
        r_iris_px = radio_iris_modulo2(serie)

    # Diámetro equivalente al área. La versión actual del módulo 2 lo da en px (d_area_px);
    # la anterior lo daba dividido por el diámetro del iris (d_area_iris)
    if "d_area_px" in serie:
        d_area = serie["d_area_px"].astype(float)
    else:
        d_area = serie["d_area_iris"].astype(float) * 2 * r_iris_px

    curva = pd.DataFrame({
        "frame": serie["frame"].astype(int),
        "t_s": serie["t"].astype(float),
        "brillo": np.nan,
        "diametro": serie["d_pupila_px"].astype(float),
        "area": np.pi * (d_area / 2) ** 2,
        "centroide_x": serie["cx"].astype(float),
        "centroide_y": serie["cy"].astype(float),
        "circularidad": serie["circularidad"].astype(float),
        "motivo_invalido": serie["motivo"].fillna("").astype(str),
    }, columns=COLUMNAS_CURVA)
    # El área solo vale en los frames en los que el diámetro es válido: el filtro de
    # atípicos del módulo 2 anula d_pupila_px pero no el diámetro del área
    curva.loc[curva["diametro"].isna(), "area"] = np.nan

    if ruta_video is not None:
        brillo = brillo_por_frame(ruta_video, curva["frame"])
        curva["brillo"] = curva["frame"].map(brillo)

    # Resumen de los motivos de descarte del módulo 2 (fuentes de artefactos)
    motivos = curva.loc[curva["motivo_invalido"] != "", "motivo_invalido"]
    motivos = motivos.str.replace(r"\s*\(.*\)", "", regex=True)   # Sin el valor entre paréntesis
    print(f"Módulo 2: {len(curva) - len(motivos)} de {len(curva)} frames válidos")
    for motivo, cantidad in motivos.value_counts().items():
        print(f"  {cantidad:5d}  {motivo}")
    return curva


def radio_iris_modulo2(serie):
    """
    Recupera el radio del iris (px) que usó el módulo 2 como escala. Solo funciona con la
    tabla de la versión anterior del módulo 2 (columna pupila_iris); con la actual el radio
    hay que pasarlo, porque lo devuelve ejecutar_modulo2.

    Entrada: serie (DataFrame del módulo 2)
    Salida: radio en px (NaN si no se puede calcular)
    """
    if "pupila_iris" not in serie:
        return np.nan
    validos =serie["d_pupila_px"].notna() & serie["pupila_iris"].notna() & (serie["pupila_iris"] > 0)
    if not validos.any():
        return np.nan
    return float(np.median(serie.loc[validos, "d_pupila_px"] / serie.loc[validos, "pupila_iris"])) / 2


def mm_por_px_desde_iris(r_iris_px, diametro_iris_mm=DIAMETRO_IRIS_MM):
    """
    Escala aproximada suponiendo que el iris mide 11,7 mm (varía cerca de ±5 % entre
    personas). Sirve para pasar los resultados a mm sin calibrar cada video.

    Entrada: r_iris_px, diametro_iris_mm (opcional)
    Salida: mm por píxel (NaN si no hay radio)
    """
    return diametro_iris_mm / (2 * r_iris_px) if r_iris_px and np.isfinite(r_iris_px) else np.nan


# ------------------------------------------------------------
# 3.5 Artefactos: detección, interpolación y suavizado
# ------------------------------------------------------------

def a_muestras(segundos, dt, minimo=1, impar=False):
    """
    Pasa una duración en segundos a cantidad de muestras.

    Entrada: segundos, dt (tiempo entre muestras), minimo, impar (los filtros necesitan ventanas impares)
    Salida: entero
    """
    n = max(minimo, int(round(segundos / dt)))
    return n + 1 if impar and n % 2 == 0 else n


def limpiar_curva(curva, columna="diametro", ventana_s=0.6, k=3.0, desvio_min_relativo=0.01,
                  circularidad_min=0.6, velocidad_centroide_max=5.0, margen_s=0.05,
                  ventana_suavizado_s=0.15, columnas_extra=("area",)):
    """
    Marca los frames con artefactos, los reemplaza interpolando y suaviza la curva.

    Un frame se marca si no hay pupila (NaN), si la circularidad es menor a circularidad_min
    (parpadeo, pestañas), si el centroide se mueve más de velocidad_centroide_max diámetros
    por segundo (sacada o salto de la segmentación) o si es atípico según el filtro de Hampel
    (ventana de ventana_s segundos, k desvíos robustos). Los tramos marcados se agrandan
    margen_s de cada lado, porque al empezar y terminar un parpadeo la pupila ya está tapada.

    Entrada: curva (de curva_desde_modulo2), columna donde se buscan los artefactos,
             columnas_extra que se limpian con los mismos frames, y los umbrales
    Salida: copia de la curva con las columnas artefacto, <columna>_limpio (interpolada)
            y <columna>_suave (Savitzky-Golay)
    """
    curva = curva.copy()
    valores = curva[columna]
    if valores.notna().sum() < 2:
        # Sin al menos dos frames válidos no hay nada que interpolar: se devuelve todo en NaN
        print(f"Atención: no hay frames válidos en la columna '{columna}'; la curva queda vacía")
        curva["artefacto"] = True
        for nombre in [columna] + [c for c in columnas_extra if c in curva and c != columna]:
            curva[f"{nombre}_limpio"] = np.nan
            curva[f"{nombre}_suave"] = np.nan
        return curva
    dt = np.median(np.diff(curva["t_s"]))   # Intervalo entre muestras (tiene en cuenta el paso)

    invalido = valores.isna().to_numpy(copy=True)   # Copia: se va a modificar
    if "circularidad" in curva:
        invalido |= (curva["circularidad"] < circularidad_min).to_numpy()

    # Velocidad del centroide en diámetros por segundo. Se normaliza por el diámetro típico
    # para que el umbral no dependa de la resolución del video ni de la distancia a la cámara.
    # El desplazamiento se mide en 50 ms (lo que dura una sacada) y no entre frames
    # consecutivos: a 120 fps, el temblor normal de la segmentación (1 o 2 px por frame)
    # daría velocidades altas aunque el ojo esté quieto
    movimiento = np.zeros(len(curva), dtype=bool)
    if {"centroide_x", "centroide_y", "diametro"} <= set(curva.columns):
        periodo = a_muestras(0.05, dt)
        desplazamiento = np.hypot(curva["centroide_x"].diff(periodo), curva["centroide_y"].diff(periodo))
        velocidad = desplazamiento / curva["t_s"].diff(periodo) / np.nanmedian(curva["diametro"])
        movimiento = (velocidad > velocidad_centroide_max).to_numpy()   # NaN da False
        invalido |= movimiento

    # Filtro de Hampel sobre los frames que quedaron válidos
    validos = valores.where(~invalido)
    ventana = a_muestras(ventana_s, dt, minimo=3, impar=True)
    mediana = validos.rolling(ventana, center=True, min_periods=1).median()
    mad = (validos - mediana).abs().rolling(ventana, center=True, min_periods=1).median()
    # 1,4826·MAD estima el desvío estándar si los datos son normales, pero sin que los
    # valores atípicos lo inflen (como pasaría con el desvío común). Se le pone un piso
    # de `desvio_min_relativo` de la mediana: en tramos muy estables el MAD es casi 0 y
    # cualquier diferencia mínima se marcaría como atípica
    desvio = np.maximum(1.4826 * mad, desvio_min_relativo * mediana)
    atipico = ((validos - mediana).abs() > k * desvio).to_numpy()
    invalido |= atipico

    # Se ensanchan los tramos marcados (dilatación morfológica en 1D)
    margen = a_muestras(margen_s, dt, minimo=0)
    if margen > 0:
        invalido = binary_dilation(invalido, structure=np.ones(2 * margen + 1, dtype=bool))
    curva["artefacto"] = invalido

    ventana = a_muestras(ventana_suavizado_s, dt, minimo=5, impar=True)
    for nombre in [columna] + [c for c in columnas_extra if c in curva and c != columna]:
        # Interpolación lineal en el tiempo de los frames descartados; en los extremos se
        # repite el primer/último valor válido
        limpio = curva[nombre].where(~invalido)
        limpio.index = curva["t_s"]
        limpio = limpio.interpolate(method="index", limit_direction="both")
        curva[f"{nombre}_limpio"] = limpio.to_numpy()

        # Savitzky-Golay: ajusta un polinomio de grado 2 en una ventana móvil. Suaviza el
        # ruido de la segmentación sin aplanar tanto los mínimos como una media móvil
        if len(curva) >= ventana and curva[f"{nombre}_limpio"].notna().all():
            curva[f"{nombre}_suave"] = savgol_filter(curva[f"{nombre}_limpio"], ventana, polyorder=2)
        else:
            curva[f"{nombre}_suave"] = curva[f"{nombre}_limpio"]

    print(f"Artefactos: {invalido.sum()} de {len(curva)} frames ({invalido.mean():.1%}); "
          f"{atipico.sum()} por el filtro de Hampel, {movimiento.sum()} por movimiento del centroide")
    return curva


# ------------------------------------------------------------
# 3.6 Parámetros fisiológicos
# ------------------------------------------------------------

def velocidad_suavizada(t, d, ventana_s=0.3):
    """
    Derivada del diámetro con un filtro de Savitzky-Golay. Derivando frame a frame el
    ruido de la segmentación tapa la velocidad real; con ventanas de 0,3 s el pico de una
    constricción se subestima menos de un 10 %.

    Entrada: t (s), d (diámetro), ventana_s
    Salida: array con la velocidad (px/s)
    """
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    if len(t) < 2 or not np.isfinite(d).all():
        return np.gradient(d, t) if len(t) > 1 else np.full(len(t), np.nan)
    dt = np.median(np.diff(t))
    ventana = a_muestras(ventana_s, dt, minimo=5, impar=True)
    if len(d) < ventana:
        return np.gradient(d, t)
    return savgol_filter(d, ventana, polyorder=2, deriv=1, delta=dt)


def respuesta_pupilar(t, d, t_estimulo, tipo, desde_s=0.3, hasta_s=2.0, basal_s=1.0):
    """
    Cuánto cambió el diámetro después de un instante: compara el segundo previo con el
    tramo entre desde_s y hasta_s segundos después.

    Entrada: t, d, t_estimulo, tipo ("contraccion" o "dilatacion")
    Salida: cambio relativo; positivo si la pupila respondió en el sentido esperado,
            NaN si faltan datos
    """
    antes = d[(t >= t_estimulo - basal_s) & (t < t_estimulo)]
    despues = d[(t >= t_estimulo + desde_s) & (t <= t_estimulo + hasta_s)]
    if antes.size == 0 or despues.size == 0 or not np.isfinite(antes).any():
        return np.nan
    basal = np.nanmean(antes)
    signo = -1 if tipo == "contraccion" else 1
    return signo * (np.nanmean(despues) - basal) / basal


def detectar_estimulo(curva, columna="brillo", columna_pupila="diametro_suave", ventana_s=1.0,
                      t_min=None, t_max=None, fraccion_candidato=0.3, separacion_s=2.0,
                      respuesta_min=0.10, fraccion_inicio=0.1):
    """
    Busca cuándo se prendió o se apagó la luz y qué respuesta se espera.

    Compara el brillo medio del segundo siguiente con el del anterior para encontrar los
    cambios sostenidos de luz. Para cada candidato toma el inicio de la rampa de brillo y mide
    la respuesta de la pupila, y se queda con el primero que responde al menos respuesta_min.
    Así se ignoran los cambios de luz posteriores, como apagar la linterna al terminar.

    Entrada: curva limpia (con brillo y diametro_suave), t_min y t_max (opcionales, para
             limitar la búsqueda) y los umbrales
    Salida: (t_estimulo en s, tipo); tipo es "contraccion" si la luz subió y "dilatacion"
            si bajó. Si no se puede estimar devuelve (NaN, "desconocido")
    """
    if len(curva) < 3 or curva[columna].notna().sum() < 3:
        print("Atención: no hay suficientes frames para detectar el estímulo")
        return np.nan, "desconocido"
    dt = np.median(np.diff(curva["t_s"]))
    ventana = a_muestras(ventana_s, dt)
    brillo = curva[columna].reset_index(drop=True)
    t = curva["t_s"].to_numpy()
    # Promedio de las `ventana` muestras anteriores a i (sin incluirla) y de las siguientes
    # (incluyéndola). Cerca de los bordes no hay ventana completa y queda NaN
    antes = brillo.rolling(ventana).mean().shift(1).to_numpy()
    despues = brillo[::-1].rolling(ventana).mean()[::-1].to_numpy()
    escalon = despues - antes
    if t_min is not None:
        escalon[t < t_min] = np.nan
    if t_max is not None:
        escalon[t > t_max] = np.nan
    if np.isnan(escalon).all():
        # El video (o el intervalo de búsqueda) dura menos que dos ventanas
        print("Atención: el intervalo de búsqueda es muy corto para detectar el estímulo")
        return np.nan, "desconocido"

    # Candidatos: de mayor a menor escalón, salteando los que están cerca de uno ya elegido
    magnitud = np.nan_to_num(np.abs(escalon), nan=0.0)
    candidatos = []
    for i in np.argsort(magnitud)[::-1]:
        if magnitud[i] < fraccion_candidato * magnitud.max():
            break
        if all(abs(t[i] - t[j]) >= separacion_s for j in candidatos):
            candidatos.append(i)
    candidatos.sort()   # En orden temporal

    # Mediana móvil de 0,1 s para que el ruido no corte antes de tiempo la búsqueda del inicio
    suave = brillo.rolling(a_muestras(0.1, dt), center=True, min_periods=1).median().to_numpy()
    d = curva[columna_pupila].to_numpy() if columna_pupila in curva else np.full(len(t), np.nan)
    elegido = mejor = None
    mejor_respuesta = 0.0
    print("Candidatos a estímulo:")
    for i in candidatos:
        # Inicio de la rampa: se retrocede mientras el brillo siga más allá del 10 % del escalón
        signo = np.sign(escalon[i])
        nivel = antes[i] + fraccion_inicio * escalon[i]
        inicio = i
        while inicio > 0 and i - inicio < ventana and signo * (suave[inicio - 1] - nivel) > 0:
            inicio -= 1

        tipo = "contraccion" if signo > 0 else "dilatacion"
        respuesta = respuesta_pupilar(t, d, t[inicio], tipo)
        print(f"  t = {t[inicio]:6.2f} s  escalón de brillo = {escalon[i]:+6.1f}  "
              f"respuesta ({tipo}) = {respuesta:+.1%}")
        if elegido is None and respuesta >= respuesta_min:
            elegido = (inicio, tipo)
        if respuesta > mejor_respuesta:
            mejor, mejor_respuesta = (inicio, tipo), respuesta

    if elegido is None:
        if mejor is not None:
            elegido = mejor
        else:
            # Ningún candidato provocó respuesta: se usa el mayor escalón
            i = int(np.argmax(magnitud))
            elegido = (i, "contraccion" if escalon[i] > 0 else "dilatacion")
            print("Atención: ningún cambio de brillo provocó una respuesta pupilar clara")
    return t[elegido[0]], elegido[1]


# Columnas de los parámetros fisiológicos, en orden fijo: el CSV tiene siempre las mismas,
# sea un registro de contracción o de dilatación; lo que no corresponde queda en NaN
COLUMNAS_PARAMETROS = [
    "tipo", "t_estimulo_s", "t_inicio_s", "latencia_s", "t_extremo_s", "tiempo_respuesta_s",
    "diametro_basal_px", "diametro_extremo_px", "amplitud_diametro_px", "amplitud_diametro_%",
    "area_basal_px2", "area_extremo_px2", "amplitud_area_px2", "amplitud_area_%",
    "velocidad_constriccion_max_px_s", "velocidad_constriccion_media_px_s",
    "velocidad_redilatacion_max_px_s", "velocidad_redilatacion_media_px_s",
    "recuperacion_%", "mm_por_px",
]


def parametros_fisiologicos(curva, t_estimulo, tipo="contraccion", columna="diametro_suave",
                            columna_area="area_suave", ventana_basal_s=1.0, ventana_respuesta_s=5.0,
                            ventana_redilatacion_s=5.0, ventana_velocidad_s=0.3, fraccion_inicio=0.1,
                            mm_por_px=None):
    """
    Calcula los parámetros del reflejo fotomotor a partir de la curva limpia.

    - basal: promedio del segundo previo al estímulo
    - extremo: mínimo (contracción) o máximo (dilatación) en los ventana_respuesta_s
      segundos siguientes; amplitud = |extremo - basal|, en diámetro y en área
    - inicio de la respuesta: desde el pico de velocidad se retrocede hasta que la pupila
      vuelve a estar a menos del 10 % de la amplitud; latencia = inicio - estímulo
    - velocidades máxima y media de constricción y de redilatación (en una contracción la
      redilatación se mide en los ventana_redilatacion_s segundos después del mínimo)

    Entrada: curva limpia, t_estimulo y tipo (de detectar_estimulo), mm_por_px (opcional,
             se guarda para poder convertir) y las ventanas
    Salida: diccionario con las claves de COLUMNAS_PARAMETROS, en px. Lo que no corresponde
            al tipo de registro o no se pudo calcular queda en NaN
    """
    parametros = dict.fromkeys(COLUMNAS_PARAMETROS, np.nan)
    parametros.update(tipo=tipo, t_estimulo_s=t_estimulo,
                      mm_por_px=np.nan if mm_por_px is None else mm_por_px)
    if tipo not in ("contraccion", "dilatacion") or not np.isfinite(t_estimulo) or columna not in curva:
        print("Atención: sin estímulo o sin curva no se pueden calcular los parámetros fisiológicos")
        return parametros

    t = curva["t_s"].to_numpy()
    d = curva[columna].to_numpy()
    area = curva[columna_area].to_numpy() if columna_area in curva else np.pi * d ** 2 / 4
    # Signo de la respuesta: multiplicando por él, la respuesta siempre crece
    signo = -1 if tipo == "contraccion" else 1
    velocidad = velocidad_suavizada(t, d, ventana_velocidad_s)   # px/s

    antes = (t >= t_estimulo - ventana_basal_s) & (t < t_estimulo) & np.isfinite(d)
    despues = np.flatnonzero((t >= t_estimulo) & (t <= t_estimulo + ventana_respuesta_s) & np.isfinite(d))
    if not antes.any() or despues.size < 3:
        print("Atención: no hay suficientes datos antes o después del estímulo para calcular los parámetros")
        return parametros

    basal, area_basal = d[antes].mean(), area[antes].mean()
    i_extremo = despues[np.argmax(signo * d[despues])]

    # Instante de velocidad máxima (en el sentido de la respuesta) entre el estímulo y el extremo
    tramo = despues[despues <= i_extremo]
    i_vmax = tramo[np.argmax(signo * velocidad[tramo])]
    v_max = signo * velocidad[i_vmax]
    amplitud = signo * (d[i_extremo] - basal)
    # Inicio: desde el pico de velocidad se retrocede mientras la pupila siga apartada del
    # basal más de `fraccion_inicio` de la amplitud. Se mide sobre el diámetro y no sobre
    # la velocidad suavizada, que por la ventana del filtro empieza a subir antes de tiempo
    i_inicio = i_vmax
    while i_inicio > tramo[0] and signo * (d[i_inicio - 1] - basal) > fraccion_inicio * amplitud:
        i_inicio -= 1

    amplitud_area = signo * (area[i_extremo] - area_basal)
    if amplitud <= 0:
        print(f"Atención: no se encontró una {tipo} después del estímulo (amplitud = {amplitud:.2f} px). "
              f"Revisar el instante del estímulo o la segmentación")
    duracion = t[i_extremo] - t[i_inicio]
    v_media = amplitud / duracion if duracion > 0 else np.nan

    parametros.update({
        "t_inicio_s": t[i_inicio],
        "latencia_s": t[i_inicio] - t_estimulo,
        "t_extremo_s": t[i_extremo],
        "tiempo_respuesta_s": duracion,
        "diametro_basal_px": basal,
        "diametro_extremo_px": d[i_extremo],
        "amplitud_diametro_px": amplitud,
        "amplitud_diametro_%": 100 * amplitud / basal,
        "area_basal_px2": area_basal,
        "area_extremo_px2": area[i_extremo],
        "amplitud_area_px2": amplitud_area,
        "amplitud_area_%": 100 * amplitud_area / area_basal,
    })

    if tipo == "contraccion":
        parametros["velocidad_constriccion_max_px_s"] = v_max
        parametros["velocidad_constriccion_media_px_s"] = v_media
        # Redilatación: desde el mínimo hasta `ventana_redilatacion_s` segundos después
        posterior = np.flatnonzero((t >= t[i_extremo]) & (t <= t[i_extremo] + ventana_redilatacion_s))
        if posterior.size > 1:
            t_fin = t[posterior[-1]]
            final = d[posterior][t[posterior] >= t_fin - 0.5].mean()   # Diámetro en el último 0,5 s
            parametros["velocidad_redilatacion_max_px_s"] = max(velocidad[posterior].max(), 0.0)
            parametros["velocidad_redilatacion_media_px_s"] = (final - d[i_extremo]) / (t_fin - t[i_extremo])
            if amplitud > 0:
                parametros["recuperacion_%"] = 100 * (final - d[i_extremo]) / amplitud
    else:
        parametros["velocidad_redilatacion_max_px_s"] = v_max
        parametros["velocidad_redilatacion_media_px_s"] = v_media
    return parametros


def imprimir_parametros(parametros):
    """
    Imprime los parámetros por consola.

    Entrada: parametros (dict de parametros_fisiologicos)
    Salida: nada
    """
    print("─" * 45)
    for clave, valor in parametros.items():
        texto = valor if isinstance(valor, str) else f"{valor:.3f}"
        print(f"{clave:<34}: {texto}")
    print("─" * 45)


def variabilidad_entre_registros(parametros_por_registro,
                                 columnas=("latencia_s", "amplitud_area_%", "amplitud_diametro_%",
                                           "velocidad_constriccion_max_px_s", "velocidad_redilatacion_max_px_s")):
    """
    Variabilidad de los parámetros entre registros, separando contracción y dilatación.
    Los valores en px solo se pueden comparar si los videos se filmaron a la misma distancia.

    Entrada: parametros_por_registro (dict {nombre del registro: parámetros}), columnas a comparar
    Salida: (tabla con los valores de cada registro, tabla con media, desvío y CV % por tipo)
    """
    columnas = list(columnas)
    tabla = pd.DataFrame(parametros_por_registro).T[["tipo"] + columnas]
    resumen = []
    for tipo, grupo in tabla.groupby("tipo"):
        valores = grupo[columnas].astype(float)
        media, desvio = valores.mean(), valores.std(ddof=1)   # ddof=1: desvío muestral
        resumen += [media.rename(f"{tipo}: media"),
                    desvio.rename(f"{tipo}: desvío"),
                    (100 * desvio / media.abs()).rename(f"{tipo}: CV %")]
    return tabla, pd.DataFrame(resumen)


def guardar_resultados(curva, parametros, carpeta="resultados/"):
    """
    Guarda la curva y los parámetros en CSV.

    Entrada: curva, parametros, carpeta
    Salida: nada (escribe curva_temporal.csv y parametros_fisiologicos.csv)
    """
    os.makedirs(carpeta, exist_ok=True)
    ruta_curva = os.path.join(carpeta, "curva_temporal.csv")
    ruta_parametros = os.path.join(carpeta, "parametros_fisiologicos.csv")
    curva.to_csv(ruta_curva, index=False)
    # `columns` fija el orden de las columnas y agrega en NaN las que falten
    pd.DataFrame([parametros], columns=COLUMNAS_PARAMETROS).to_csv(ruta_parametros, index=False)
    print(f"Guardados: {ruta_curva} y {ruta_parametros}")


# ------------------------------------------------------------
# Visualización
# ------------------------------------------------------------

def mostrar_descriptores(imagen, mascara, descriptores=None, titulo=None):
    """
    Muestra la imagen con el contorno de la pupila, el centroide y la elipse
    equivalente, y al lado la máscara.

    Entrada: imagen, mascara, descriptores (opcional; si no se pasan se calculan), titulo
    Salida: nada (muestra la figura)
    """
    if descriptores is None:
        descriptores = descriptores_morfometricos(mascara)
    componente = componente_principal(mascara)
    d = descriptores

    fig, (eje_img, eje_mask) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    eje_img.imshow(imagen, cmap=None if imagen.ndim == 3 else "gray")
    if componente is not None:
        # contour dibuja la curva de nivel 0,5 de la máscara: el borde de la región
        eje_img.contour(componente, levels=[0.5], colors="lime", linewidths=1.5)
        eje_img.add_patch(Ellipse((d["centroide_x"], d["centroide_y"]), d["eje_mayor"], d["eje_menor"],
                                  angle=d["orientacion_grados"], fill=False, edgecolor="red",
                                  linestyle="--", linewidth=1.5))
        eje_img.plot(d["centroide_x"], d["centroide_y"], "r+", markersize=12, markeredgewidth=2)
    eje_img.set_title(f"Contorno (verde), elipse equivalente (rojo)\n"
                      f"Circularidad = {d['circularidad']:.3f}, excentricidad = {d['excentricidad']:.3f}")
    eje_img.axis("off")

    eje_mask.imshow(mascara > 0, cmap="gray")
    eje_mask.set_title(f"Máscara\nÁrea = {d['area']:.0f} px, diámetro equivalente = {d['diametro']:.1f} px")
    eje_mask.axis("off")

    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    plt.show()


def mostrar_curva(curva, columna="diametro", parametros=None, unidad="px", titulo=None):
    """
    Grafica la curva cruda y la limpia, la velocidad y el brillo. Si se pasan los
    parámetros marca el estímulo, el inicio de la respuesta y el extremo.

    Entrada: curva limpia, columna, parametros (opcional), unidad, titulo
    Salida: nada (muestra la figura)
    """
    t = curva["t_s"]
    suave = f"{columna}_suave"
    filas = 3 if "brillo" in curva else 2
    fig, ejes = plt.subplots(filas, 1, figsize=(13, 3 * filas), sharex=True, layout="constrained")

    # Diámetro
    eje = ejes[0]
    eje.plot(t, curva[columna], color="0.7", linewidth=1, label="Crudo")
    if "artefacto" in curva:
        art = curva["artefacto"]
        eje.plot(t[art], curva[columna][art], "x", color="tab:red", markersize=4, label="Artefacto")
    if suave in curva:
        eje.plot(t, curva[suave], color="tab:blue", linewidth=2, label="Limpio y suavizado")
    eje.set_ylabel(f"Diámetro ({unidad})")
    eje.legend(loc="best")
    eje.grid(alpha=0.3)

    # Velocidad (la misma derivada suavizada que usan los parámetros)
    if suave in curva and len(curva) > 1:
        ejes[1].plot(t, velocidad_suavizada(t, curva[suave]), color="tab:purple")
        ejes[1].axhline(0, color="0.5", linewidth=0.8)
    ejes[1].set_ylabel(f"Velocidad ({unidad}/s)")
    ejes[1].grid(alpha=0.3)

    # Brillo medio del frame (el estímulo)
    if filas == 3:
        ejes[2].plot(t, curva["brillo"], color="tab:orange")
        ejes[2].set_ylabel("Brillo medio")
        ejes[2].grid(alpha=0.3)
    ejes[-1].set_xlabel("Tiempo (s)")

    if parametros is not None:
        marcas = [("t_estimulo_s", "tab:orange", "--"), ("t_inicio_s", "tab:green", ":"),
                  ("t_extremo_s", "tab:red", ":")]
        for clave, color, estilo in marcas:
            if np.isfinite(parametros[clave]):   # Si no se pudo calcular, no se dibuja
                for eje in ejes:
                    eje.axvline(parametros[clave], color=color, linestyle=estilo, linewidth=1.2)
        ejes[0].set_title(f"Estímulo (naranja), inicio de la respuesta (verde) y extremo (rojo). "
                          f"Latencia = {parametros['latencia_s'] * 1000:.0f} ms, "
                          f"amplitud = {parametros['amplitud_area_%']:.1f} % del área")

    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    plt.show()
