"""
Módulo 4 — Compresión de la información de imagen.

Compara Huffman, RLE (horizontal y vertical) y JPEG (Q = 90, 50, 10 y 5) sobre el frame,
la ROI o la máscara segmentada. Las imágenes van en RGB o en grises (uint8); las máscaras
binarias se pasan a 0/255.
"""

import heapq                     # Cola de prioridad para armar el árbol de Huffman
import cv2                       # OpenCV: codificación JPEG, gradientes y transformaciones
import numpy as np               # Cálculo numérico
import pandas as pd              # Tablas de resultados

from . import metricas

CALIDADES_JPEG = (90, 50, 10, 5)


def a_uint8(imagen):
    """
    Pasa las máscaras booleanas a 0/255; el resto lo deja igual.

    Entrada: imagen
    Salida: imagen uint8
    """
    return imagen.astype(np.uint8) * 255 if imagen.dtype == bool else imagen


def a_gris(imagen):
    """
    Pasa la imagen a escala de grises en float, para restar y calcular gradientes.

    Entrada: imagen RGB o en grises
    Salida: array 2D float
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) if imagen.ndim == 3 else imagen
    return gris.astype(np.float64)


# ------------------------------------------------------------
# 4.1 Huffman
# ------------------------------------------------------------

def longitudes_huffman(frecuencias):
    """
    Longitud del código de Huffman de cada símbolo. Arma el árbol uniendo siempre los
    dos nodos menos frecuentes; cada unión le suma un bit a los símbolos de esos nodos.

    Entrada: frecuencias (array con la cantidad de veces que aparece cada valor 0-255)
    Salida: diccionario {símbolo: longitud en bits}
    """
    simbolos = [s for s in range(len(frecuencias)) if frecuencias[s] > 0]
    if len(simbolos) == 1:
        return {simbolos[0]: 1}   # Con un único símbolo igual hace falta 1 bit por dato
    longitudes = dict.fromkeys(simbolos, 0)
    # Cada nodo es (frecuencia, desempate, símbolos que contiene). El desempate es un
    # número distinto para cada nodo, para que heapq nunca tenga que comparar las listas
    cola = [(frecuencias[s], s, [s]) for s in simbolos]
    heapq.heapify(cola)
    siguiente = len(frecuencias)
    while len(cola) > 1:
        f1, _, s1 = heapq.heappop(cola)
        f2, _, s2 = heapq.heappop(cola)
        for s in s1 + s2:
            longitudes[s] += 1
        heapq.heappush(cola, (f1 + f2, siguiente, s1 + s2))
        siguiente += 1
    return longitudes


def codigos_canonicos(longitudes):
    """
    Arma los códigos de Huffman canónico a partir de las longitudes. Con este formato
    alcanza con guardar las 256 longitudes para poder decodificar, sin el árbol.

    Entrada: longitudes (dict de longitudes_huffman)
    Salida: diccionario {símbolo: código como texto de '0' y '1'}
    """
    codigos = {}
    codigo, longitud_anterior = 0, 0
    for simbolo, longitud in sorted(longitudes.items(), key=lambda par: (par[1], par[0])):
        codigo <<= longitud - longitud_anterior   # Al pasar a códigos más largos se agregan ceros
        codigos[simbolo] = format(codigo, f"0{longitud}b")
        codigo += 1
        longitud_anterior = longitud
    return codigos


def comprimir_huffman(imagen):
    """
    Codifica con Huffman todos los bytes de la imagen.

    Entrada: imagen
    Salida: diccionario con codigos, bytes (datos + tabla de 256 longitudes),
            bits_por_simbolo y entropia (mínimo teórico en bits por símbolo)
    """
    datos = a_uint8(imagen).ravel()
    frecuencias = np.bincount(datos, minlength=256)
    longitudes = longitudes_huffman(frecuencias)
    bits = sum(int(frecuencias[s]) * l for s, l in longitudes.items())
    p = frecuencias[frecuencias > 0] / datos.size
    return {
        "codigos": codigos_canonicos(longitudes),
        "bytes": int(np.ceil(bits / 8)) + 256,   # 256: una longitud de código por símbolo
        "bits_por_simbolo": bits / datos.size,
        "entropia": float(np.sum(p * np.log2(1 / p))),
    }


def codificar_huffman(imagen, codigos):
    """
    Reemplaza cada byte de la imagen por su código.

    Entrada: imagen, codigos (de comprimir_huffman)
    Salida: texto con los bits
    """
    return "".join(codigos[v] for v in a_uint8(imagen).ravel().tolist())


def decodificar_huffman(bits, codigos, forma):
    """
    Reconstruye la imagen leyendo los bits hasta formar un código conocido. Es lenta
    (va bit por bit); sirve para comprobar que no se pierde información.

    Entrada: bits (de codificar_huffman), codigos, forma de la imagen
    Salida: imagen uint8
    """
    simbolo_de = {c: s for s, c in codigos.items()}
    salida, actual = [], ""
    for bit in bits:
        actual += bit
        if actual in simbolo_de:
            salida.append(simbolo_de[actual])
            actual = ""
    return np.array(salida, dtype=np.uint8).reshape(forma)


# ------------------------------------------------------------
# 4.2 RLE (Run-Length Encoding)
# ------------------------------------------------------------

def _orden(direccion):
    """
    Orden de recorrido de numpy según la dirección del RLE.

    Entrada: direccion ("horizontal" o "vertical")
    Salida: "C" (por filas) o "F" (por columnas)
    """
    if direccion not in ("horizontal", "vertical"):
        raise ValueError("La dirección tiene que ser 'horizontal' o 'vertical'")
    return "C" if direccion == "horizontal" else "F"


def rle_codificar(imagen, direccion="horizontal", max_corrida=255):
    """
    Codifica la imagen como pares (valor, longitud de la corrida), recorriéndola por
    filas o por columnas, un canal después del otro. La longitud ocupa 1 byte, así que
    las corridas de más de max_corrida se parten en varios pares.

    Entrada: imagen, direccion ("horizontal" o "vertical"), max_corrida
    Salida: (valores, longitudes); cada par ocupa 2 bytes
    """
    imagen = a_uint8(imagen)
    orden = _orden(direccion)
    planos = [imagen] if imagen.ndim == 2 else [imagen[..., c] for c in range(imagen.shape[2])]
    secuencia = np.concatenate([p.ravel(order=orden) for p in planos])

    # Una corrida empieza en la posición 0 y en cada lugar donde el valor cambia
    inicios = np.r_[0, np.flatnonzero(secuencia[1:] != secuencia[:-1]) + 1]
    largos = np.diff(np.r_[inicios, secuencia.size])
    valores = secuencia[inicios]

    # Partición de las corridas largas: una de 600 píxeles queda como 255 + 255 + 90
    partes = -(-largos // max_corrida)              # División entera redondeando hacia arriba
    valores = np.repeat(valores, partes)
    longitudes = np.full(partes.sum(), max_corrida)
    ultimas = np.cumsum(partes) - 1                 # Posición del último pedazo de cada corrida
    longitudes[ultimas] = largos - max_corrida * (partes - 1)
    return valores, longitudes.astype(np.uint8)


def rle_decodificar(valores, longitudes, forma, direccion="horizontal"):
    """
    Reconstruye la imagen repitiendo cada valor tantas veces como dice su longitud.

    Entrada: valores, longitudes (de rle_codificar), forma de la imagen, direccion
    Salida: imagen uint8
    """
    orden = _orden(direccion)
    secuencia = np.repeat(valores, longitudes)
    if len(forma) == 2:
        return secuencia.reshape(forma, order=orden)
    alto, ancho, canales = forma
    planos = secuencia.reshape(canales, alto * ancho)
    return np.stack([p.reshape((alto, ancho), order=orden) for p in planos], axis=-1)


# ------------------------------------------------------------
# 4.3 JPEG
# ------------------------------------------------------------

def comprimir_jpeg(imagen, calidad):
    """
    Comprime con JPEG y vuelve a decodificar, para medir la pérdida.

    Entrada: imagen, calidad (1 a 100)
    Salida: (imagen reconstruida, tamaño en bytes, con el encabezado incluido)
    """
    imagen = a_uint8(imagen)
    color = imagen.ndim == 3
    # OpenCV codifica en BGR; una imagen de un canal se guarda como JPEG en grises
    entrada = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR) if color else imagen
    ok, buffer = cv2.imencode(".jpg", entrada, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
    if not ok:
        raise ValueError(f"No se pudo codificar en JPEG con Q = {calidad}")
    reconstruida = cv2.imdecode(buffer, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if color:
        reconstruida = cv2.cvtColor(reconstruida, cv2.COLOR_BGR2RGB)
    return reconstruida, buffer.size


# ------------------------------------------------------------
# 4.4 Comparación: ratio de compresión, PSNR y SSIM
# ------------------------------------------------------------

def ssim_color(original, procesada):
    """
    SSIM para imágenes en color: metricas.ssim trabaja en grises, así que en RGB se
    calcula canal por canal y se promedia.

    Entrada: original, procesada
    Salida: SSIM medio
    """
    if original.ndim == 2:
        return metricas.ssim(original, procesada)
    return float(np.mean([metricas.ssim(original[..., c], procesada[..., c])
                          for c in range(original.shape[2])]))


def comparar_metodos(imagen, calidades=CALIDADES_JPEG, verificar_huffman=False):
    """
    Aplica todos los métodos a la imagen y mide cuánto comprime cada uno y cuánto se
    pierde. RLE se decodifica siempre para verificar que no pierde información; Huffman
    solo si verificar_huffman es True, porque es lento.

    Entrada: imagen, calidades de JPEG, verificar_huffman
    Salida: (tabla con bytes, ratio, bits por píxel, PSNR, SSIM y si la reconstrucción es
            exacta, una fila por método; dict {método: imagen reconstruida})
    """
    imagen = a_uint8(imagen)
    original = imagen.nbytes
    pixeles = imagen.shape[0] * imagen.shape[1]
    filas, reconstruidas = [], {}

    def agregar(metodo, tipo, tamano, reconstruida):
        if reconstruida is None:
            # Huffman sin verificar: es sin pérdida por construcción
            exacta, valor_psnr, valor_ssim = np.nan, np.inf, 1.0
        else:
            reconstruidas[metodo] = reconstruida
            exacta = bool(np.array_equal(reconstruida, imagen))
            valor_psnr = metricas.psnr(imagen, reconstruida)
            valor_ssim = ssim_color(imagen, reconstruida)
        filas.append({"metodo": metodo, "tipo": tipo, "bytes": tamano,
                      "ratio": original / tamano, "bits_por_pixel": 8 * tamano / pixeles,
                      "psnr_db": valor_psnr, "ssim": valor_ssim, "exacta": exacta})

    agregar("Original", "sin compresión", original, imagen)

    huffman = comprimir_huffman(imagen)
    reconstruida = None
    if verificar_huffman:
        bits = codificar_huffman(imagen, huffman["codigos"])
        reconstruida = decodificar_huffman(bits, huffman["codigos"], imagen.shape)
    agregar("Huffman", "sin pérdida", huffman["bytes"], reconstruida)

    for direccion in ("horizontal", "vertical"):
        valores, longitudes = rle_codificar(imagen, direccion)
        agregar(f"RLE {direccion}", "sin pérdida", 2 * len(valores),
                rle_decodificar(valores, longitudes, imagen.shape, direccion))

    for calidad in calidades:
        reconstruida, tamano = comprimir_jpeg(imagen, calidad)
        agregar(f"JPEG Q={calidad}", "con pérdida", tamano, reconstruida)

    return pd.DataFrame(filas).set_index("metodo"), reconstruidas


# ------------------------------------------------------------
# 4.5 Artefactos de JPEG: bloques, bordes y regiones uniformes
# ------------------------------------------------------------

def indice_bloques(imagen, mascara=None, lado=8):
    """
    Mide los artefactos de bloque de JPEG: cuánto más saltan los valores entre píxeles
    vecinos que están en distintos bloques de 8x8 que entre vecinos del mismo bloque.
    Da cerca de 0 sin artefactos y crece cuando aparecen los bloques.

    Entrada: imagen, mascara (opcional, para usar solo los pares que caen adentro), lado del bloque
    Salida: diferencia en niveles de gris
    """
    gris = a_gris(imagen)
    alto, ancho = gris.shape
    dentro = np.ones((alto, ancho), bool) if mascara is None else mascara > 0

    # Diferencias con el vecino de la derecha (columna j contra j+1) y con el de abajo
    dif_h = np.abs(np.diff(gris, axis=1))
    dif_v = np.abs(np.diff(gris, axis=0))
    validos_h = dentro[:, 1:] & dentro[:, :-1]
    validos_v = dentro[1:, :] & dentro[:-1, :]
    # El par (j, j+1) cruza un borde de bloque cuando j+1 es múltiplo de `lado`
    cruza_h = np.broadcast_to((np.arange(1, ancho) % lado == 0)[np.newaxis, :], dif_h.shape)
    cruza_v = np.broadcast_to((np.arange(1, alto) % lado == 0)[:, np.newaxis], dif_v.shape)

    en_borde = np.r_[dif_h[validos_h & cruza_h], dif_v[validos_v & cruza_v]]
    en_bloque = np.r_[dif_h[validos_h & ~cruza_h], dif_v[validos_v & ~cruza_v]]
    if en_borde.size == 0 or en_bloque.size == 0:
        return np.nan
    return en_borde.mean() - en_bloque.mean()


def regiones_borde_y_uniforme(imagen, percentil_borde=90, percentil_uniforme=30):
    """
    Separa los bordes nítidos (gradiente alto) de las zonas uniformes (gradiente bajo).

    Entrada: imagen, percentil_borde, percentil_uniforme
    Salida: (máscara de bordes, máscara de zonas uniformes)
    """
    gris = cv2.GaussianBlur(a_gris(imagen), (5, 5), 0)   # Que el ruido no cuente como borde
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1)
    magnitud = np.hypot(gx, gy)
    return (magnitud >= np.percentile(magnitud, percentil_borde),
            magnitud <= np.percentile(magnitud, percentil_uniforme))


def analizar_artefactos(imagen, reconstruidas):
    """
    Compara los artefactos de cada versión en los bordes y en las zonas uniformes. Son
    valores absolutos: en una zona uniforme un salto chico ya se ve, en un borde se disimula.

    Entrada: imagen original, reconstruidas (de comparar_metodos)
    Salida: tabla con el índice de bloques (total, bordes y uniforme) y el error medio en
            bordes y en zonas uniformes, una fila por versión
    """
    imagen = a_uint8(imagen)
    bordes, uniforme = regiones_borde_y_uniforme(imagen)
    filas = {}
    for metodo, reconstruida in reconstruidas.items():
        error = np.abs(a_gris(imagen) - a_gris(reconstruida))
        filas[metodo] = {
            "indice_bloques": indice_bloques(reconstruida),
            "indice_bloques_bordes": indice_bloques(reconstruida, bordes),
            "indice_bloques_uniforme": indice_bloques(reconstruida, uniforme),
            "error_bordes": error[bordes].mean(),
            "error_uniforme": error[uniforme].mean(),
        }
    return pd.DataFrame(filas).T
