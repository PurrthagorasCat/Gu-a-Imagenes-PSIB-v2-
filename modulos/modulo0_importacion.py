"""
Módulo 0 — Importación de datos.

Apertura de los videos (.MOV), lectura de metadatos y frames, y visualización del registro.
"""

import cv2                       # OpenCV: lectura de video, procesamiento de imágenes y ventanas
import numpy as np               # Cálculo numérico (arrays, linspace, ceil)
import matplotlib.pyplot as plt  # Gráficos para mostrar frames sueltos
from matplotlib.patches import Rectangle, ConnectionPatch  # Recuadro de la ROI y flecha entre gráficos


# ------------------------------------------------------------
# Importación
# ------------------------------------------------------------

def abrir_video(ruta):
    """
    Abre el video y devuelve el objeto de lectura y un diccionario con sus metadatos.

    Las dimensiones se toman del primer frame y no del encabezado del archivo,
    ya que los videos de celular guardan la orientación aparte y podría ser que
    el encabezado informe el ancho y alto invertidos.
    """
    # Se corrobora que el video se haya abierto correctamente
    # ================================================================================== #
    video = cv2.VideoCapture(str(ruta))
    if not video.isOpened():
        # El archivo no existe, la ruta es incorrecta o OpenCV no soporta el formato
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta}")

    # Se intenta leer el primer frame: read() devuelve (éxito, imagen)
    ok, primer_frame = video.read()
    if not ok: # Si el archivo se abrió pero está vacío/corrupto o falta el códec para decodificarlo
        raise ValueError(f"El video se abrió pero no se pudo leer ningún frame: {ruta}")
    # Al leer, el lector avanzó un frame; se lo vuelve al inicio para no "perder" el frame 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # ================================================================================== #

    # Lectura de metadatos
    # ================================================================================== #
    alto, ancho, canales = primer_frame.shape
    fps = video.get(cv2.CAP_PROP_FPS)                                # Frames por segundo
    cantidad_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))       # Total de frames (estimado por el encabezado)
    codigo = int(video.get(cv2.CAP_PROP_FOURCC))                     # Códec codificado como un entero de 32 bits

    metadatos = {
        "ruta": str(ruta),
        "ancho": ancho,
        "alto": alto,
        "canales": canales,
        "fps": fps,
        "cantidad_frames": cantidad_frames,
        "duracion_s": cantidad_frames / fps if fps > 0 else 0.0,
        # El FOURCC empaqueta 4 caracteres (1 byte cada uno) en un entero: se extrae cada
        # byte desplazando 8 bits por vez y se lo convierte a letra (ej. "avc1", "hvc1")
        "codec": "".join(chr((codigo >> 8 * i) & 0xFF) for i in range(4)),
    }
    return video, metadatos


def imprimir_metadatos(metadatos):
    """Muestra por consola los metadatos del video."""
    print("─" * 45)  # Línea separadora
    print(f"Video            : {metadatos['ruta']}")
    print(f"Resolución       : {metadatos['ancho']} x {metadatos['alto']} px")
    print(f"Canales          : {metadatos['canales']}")
    print(f"Frecuencia       : {metadatos['fps']:.2f} fps")   # :.2f → 2 decimales
    print(f"Cantidad frames  : {metadatos['cantidad_frames']}")
    print(f"Duración         : {metadatos['duracion_s']:.2f} s")
    print(f"Códec            : {metadatos['codec']}")
    print("─" * 45)


def leer_frame(video, indice):
    """
    Devuelve el frame número `indice` en RGB.

    OpenCV lee los frames en orden BGR; se convierten a RGB para que
    matplotlib y el resto del procesamiento muestren los colores reales.
    """
    # Se posiciona el lector directamente en el frame pedido (acceso aleatorio)
    video.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, frame = video.read()
    if not ok:
        # Índice fuera de rango o frame imposible de decodificar
        raise IndexError(f"No se pudo leer el frame {indice}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# Región de interés (ROI)
# ------------------------------------------------------------

def seleccionar_roi(frame, alto_max=900, ancho_max=1400):
    """
    Abre una ventana para seleccionar la ROI a mano sobre el frame (en RGB).

    Se arrastra un rectángulo con el mouse y se confirma con Enter o Espacio;
    la tecla c cancela. Si el frame no entra en pantalla se muestra achicado,
    pero las coordenadas devueltas corresponden a la imagen original.

    Devuelve (x, y, ancho, alto) en píxeles.
    """
    alto, ancho = frame.shape[:2]
    # Factor de escala para que la imagen entre en pantalla
    escala = min(1.0, alto_max / alto, ancho_max / ancho)
    # INTER_AREA es la interpolación recomendada para reducir imágenes (evita el aliasing)
    vista = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    vista = cv2.cvtColor(vista, cv2.COLOR_RGB2BGR)  # La ventana de OpenCV espera BGR

    # selectROI bloquea el programa hasta que se confirma o cancela, y devuelve el rectángulo elegido
    titulo = "Seleccionar ROI - Enter confirma, c cancela"
    x, y, w, h = cv2.selectROI(titulo, vista, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Esta línae es necesario para que la ventana se cierre al usarla desde Jupyter

    if w == 0 or h == 0:
        # Se canceló o no se arrastró ningún rectángulo
        raise ValueError("No se seleccionó ninguna región")
    # Las coordenadas están en la vista achicada, así que se dividen por la escala
    # para llevarlas al tamaño original del frame
    return tuple(int(round(v / escala)) for v in (x, y, w, h))


def recortar_roi(frame, roi):
    """Devuelve la porción del frame delimitada por roi = (x, y, ancho, alto)."""
    x, y, w, h = roi
    return frame[y:y + h, x:x + w].copy()


def mostrar_roi(frame, roi, indice_frame=None):
    """
    Muestra el frame completo con la ROI recuadrada y, al lado, la ROI ampliada,
    unidas por una flecha.
    """
    x, y, w, h = roi
    recorte = recortar_roi(frame, roi)

    fig, (eje_frame, eje_roi) = plt.subplots(1, 2, figsize=(13, 6))

    # Frame completo con la ROI marcada con un rectángulo rojo sin relleno
    eje_frame.imshow(frame)
    eje_frame.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
    titulo = "Frame completo" if indice_frame is None else f"Frame {indice_frame}"
    eje_frame.set_title(f"{titulo}  ({frame.shape[1]} x {frame.shape[0]} px)")
    eje_frame.axis("off")

    # ROI ampliada
    eje_roi.imshow(recorte)
    eje_roi.set_title(f"ROI  ({w} x {h} px, origen en x = {x}, y = {y})")
    eje_roi.axis("off")

    # Flecha desde el borde derecho de la ROI (en píxeles del frame) hasta el borde
    # izquierdo del segundo gráfico (en fracción de ejes: 0 = izquierda, 0.5 = mitad de alto)
    flecha = ConnectionPatch(
        xyA=(x + w, y + h / 2), coordsA="data", axesA=eje_frame,
        xyB=(0, 0.5), coordsB="axes fraction", axesB=eje_roi,
        arrowstyle="-|>", mutation_scale=20, color="red", linewidth=2,
    )
    fig.add_artist(flecha)  # Se agrega a la figura (y no a un eje) porque une dos gráficos

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Visualización
# ------------------------------------------------------------

def reproducir_video(ruta, alto_ventana=720):
    """
    Reproduce el video en una ventana, con el número de frame y el tiempo superpuestos.

    Controles:  Espacio = pausa / continuar   |   q o Esc = cerrar
    """
    video, metadatos = abrir_video(ruta)
    fps = metadatos["fps"]
    # Tiempo de espera entre frames (en ms) para reproducir a velocidad real; mínimo 1 ms
    espera_ms = max(1, int(1000 / fps))
    # Factor de escala para que la ventana no supere `alto_ventana` píxeles de alto
    escala = min(1.0, alto_ventana / metadatos["alto"])

    indice = 0        # Número del frame actual
    pausado = False   # Estado de pausa
    while True:
        # Solo se avanza al siguiente frame si no está en pausa
        if not pausado:
            ok, frame = video.read()
            if not ok:
                break  # Se llegó al final del video
            # Se redimensiona el frame con el mismo factor en x e y (mantiene proporción)
            frame = cv2.resize(frame, None, fx=escala, fy=escala)
            # Texto superpuesto con el número de frame y el tiempo transcurrido
            texto = f"Frame {indice}/{metadatos['cantidad_frames']}   t = {indice / fps:.2f} s"
            cv2.putText(frame, texto, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Video", frame)
            indice += 1

        # waitKey espera `espera_ms` y devuelve la tecla presionada (o -1 si no hubo).
        tecla = cv2.waitKey(espera_ms) & 0xFF
        if tecla in (ord("q"), 27):  # 27 = código ASCII de la tecla Esc
            break
        if tecla == ord(" "):
            pausado = not pausado    # Alterna entre pausa y reproducción

    # Se liberan el archivo de video y las ventanas abiertas por OpenCV
    video.release()
    cv2.destroyAllWindows()


def mostrar_frames_muestra(video, metadatos, cantidad=6):
    """Muestra `cantidad` frames equiespaciados a lo largo del video, con su tiempo."""
    indices = np.linspace(0, metadatos["cantidad_frames"] - 1, cantidad, dtype=int)
    columnas = 3
    filas = int(np.ceil(cantidad / columnas))

    # Grilla de subgráficos
    fig, ejes = plt.subplots(filas, columnas, figsize=(4 * columnas, 3.5 * filas))
    for eje, indice in zip(ejes.flat, indices):
        try:
            eje.imshow(leer_frame(video, indice))
            # Tiempo del frame = índice / fps
            eje.set_title(f"Frame {indice}  —  t = {indice / metadatos['fps']:.2f} s")
        except IndexError:
            eje.set_title(f"Frame {indice}: no disponible")
        eje.axis("off")
    for eje in ejes.flat[len(indices):]:
        eje.axis("off")

    fig.suptitle(f"Frames de muestra — {metadatos['ruta']}")
    fig.tight_layout()  # Ajusta espacios para que los títulos no se superpongan
    plt.show()
