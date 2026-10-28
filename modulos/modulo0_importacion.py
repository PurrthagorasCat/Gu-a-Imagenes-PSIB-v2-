"""
Módulo 0 — Importación de datos.

Apertura de los videos (.MOV), lectura de metadatos y frames, y visualización del registro.
"""

import os                        # Rutas alternativas para abrir el video en Windows
import shutil                    # Copia temporal del video (último recurso en Windows)
import sys                       # Detectar Windows
import tempfile                  # Carpeta para esa copia temporal
from contextlib import contextmanager  # Para usar abrir_video con "with"
from pathlib import Path
import cv2                       # OpenCV: lectura de video, procesamiento de imágenes y ventanas
import numpy as np               # Cálculo numérico (arrays, linspace, ceil)
import matplotlib.pyplot as plt  # Gráficos para mostrar frames sueltos
from matplotlib.patches import Rectangle, ConnectionPatch  # Recuadro de la ROI y flecha entre gráficos

RAIZ = Path(__file__).resolve().parent.parent   # Carpeta del proyecto
CARPETA_VIDEOS = RAIZ / "videos"


# ------------------------------------------------------------
# Importación
# ------------------------------------------------------------

@contextmanager
def abrir_video(ruta):
    """
    Abre el video para usarlo con "with" y lo libera al salir, aunque haya un error
    o se corte el proceso con Ctrl+C:

        with abrir_video(ruta) as video:
            metadatos = leer_metadatos(video, ruta)

    La ruta puede ser absoluta, relativa a la carpeta actual o el nombre de un video de videos/.

    En Windows, cv2.VideoCapture no abre rutas con caracteres no ASCII (p. ej. "Señales").
    Se prueba en este orden, de más barato a más caro:
      1. la ruta corta 8.3 de Windows (suele ser ASCII);
      2. si solo la carpeta tiene acentos y el nombre del archivo no, se abre con el nombre
         desde la carpeta del video (no copia nada);
      3. como último recurso, una copia temporal, que se borra siempre al salir.
    En Linux y macOS no hace falta nada de esto.
    """
    ruta = Path(ruta)
    if not ruta.exists() and (CARPETA_VIDEOS / ruta).exists():
        ruta = CARPETA_VIDEOS / ruta
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el video: {ruta}")
    carpeta_temporal = None
    video = None
    try:
        if sys.platform == "win32" and not str(ruta).isascii():
            corta = _ruta_corta_windows(ruta)
            if corta and corta.isascii():
                video = cv2.VideoCapture(corta)
            elif ruta.name.isascii():
                actual = os.getcwd()
                try:
                    os.chdir(ruta.parent)
                    video = cv2.VideoCapture(ruta.name)
                finally:
                    os.chdir(actual)
            else:
                carpeta_temporal = Path(tempfile.mkdtemp())
                copia = carpeta_temporal / ("video" + ruta.suffix)
                print(f"  (copiando el video a {carpeta_temporal} por el nombre con acentos)")
                shutil.copy(ruta, copia)
                video = cv2.VideoCapture(str(copia))
        else:
            video = cv2.VideoCapture(str(ruta))
        if not video.isOpened():
            # La ruta existe, pero OpenCV no soporta el formato o el archivo está dañado
            raise IOError(f"No se pudo abrir el video: {ruta}")
        yield video
    finally:
        if video is not None:
            video.release()
        if carpeta_temporal is not None:
            shutil.rmtree(carpeta_temporal, ignore_errors=True)


def _ruta_corta_windows(ruta):
    """Nombre corto 8.3 de Windows (p. ej. C:\\Users\\Facu\\DESKTOP\\PROCES~1\\...), o None."""
    try:
        import ctypes
        buffer = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(ruta), buffer, 1024)
        return buffer.value if n else None
    except Exception:
        return None


def fps_de(video):
    """Frames por segundo del video. Algunos archivos no los informan: en ese caso se supone 30."""
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps if fps and np.isfinite(fps) and fps > 0 else 30.0


def leer_metadatos(video, ruta=""):
    """
    Devuelve un diccionario con los metadatos del video ya abierto.

    Las dimensiones se toman del primer frame y no del encabezado del archivo,
    ya que los videos de celular guardan la orientación aparte y podría ser que
    el encabezado informe el ancho y alto invertidos.
    """
    # Se intenta leer el primer frame: read() devuelve (éxito, imagen)
    posicion = video.get(cv2.CAP_PROP_POS_FRAMES)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, primer_frame = video.read()
    if not ok:  # Si el archivo se abrió pero está vacío/corrupto o falta el códec para decodificarlo
        raise ValueError(f"El video se abrió pero no se pudo leer ningún frame: {ruta}")
    # Al leer, el lector avanzó un frame; se lo vuelve a donde estaba para no "perder" frames
    video.set(cv2.CAP_PROP_POS_FRAMES, posicion)

    alto, ancho, canales = primer_frame.shape
    fps = fps_de(video)
    cantidad_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))       # Total de frames (estimado por el encabezado)
    codigo = int(video.get(cv2.CAP_PROP_FOURCC))                     # Códec codificado como un entero de 32 bits
    return {
        "ruta": str(ruta),
        "ancho": ancho,
        "alto": alto,
        "canales": canales,
        "fps": fps,
        "cantidad_frames": cantidad_frames,
        "duracion_s": cantidad_frames / fps,
        # El FOURCC empaqueta 4 caracteres (1 byte cada uno) en un entero: se extrae cada
        # byte desplazando 8 bits por vez y se lo convierte a letra (ej. "avc1", "hvc1")
        "codec": "".join(chr((codigo >> 8 * i) & 0xFF) for i in range(4)),
    }


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

def seleccionar_roi(frame, bgr=False, alto_max=900, ancho_max=1400,
                    titulo="ROI: encerrar el iris con margen y apretar Enter (c = cancelar)"):
    """
    Abre una ventana para seleccionar la ROI a mano sobre el frame (en RGB, o en BGR
    si bgr=True, como lo lee OpenCV).

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
    if not bgr:
        vista = cv2.cvtColor(vista, cv2.COLOR_RGB2BGR)  # La ventana de OpenCV espera BGR

    print("  >>> Se abrió una ventana para marcar el ROI (si no la ves, puede haber quedado "
          "detrás de otras ventanas). Encerrá el iris con margen y apretá Enter.", flush=True)
    try:
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(titulo, cv2.WND_PROP_TOPMOST, 1)    # Adelante de todo
        # selectROI bloquea el programa hasta que se confirma o cancela, y devuelve el rectángulo elegido
        x, y, w, h = cv2.selectROI(titulo, vista, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(titulo)
        cv2.waitKey(1)  # Necesario para que la ventana se cierre al usarla desde Jupyter
    except cv2.error:
        # Sin pantalla no se puede marcar: se marca una vez en una computadora con pantalla
        # (pipeline o aplicación) y queda guardado en roi.json para las corridas siguientes
        raise RuntimeError("No hay entorno gráfico para abrir la ventana del ROI. Marcalo una vez en una "
                           "computadora con pantalla (pipeline_completo.py o la aplicación): queda guardado "
                           "en anotaciones_manuales/roi.json. Con modulo2_segmentacion.py también se puede "
                           "pasar con --roi X Y ANCHO ALTO")

    if w == 0 or h == 0:
        # Se canceló o no se arrastró ningún rectángulo
        raise RuntimeError("Selección del ROI cancelada")
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
    with abrir_video(ruta) as video:
        _reproducir(video, leer_metadatos(video, ruta), alto_ventana)


def _reproducir(video, metadatos, alto_ventana):
    """Bucle de reproducción de reproducir_video, con el video ya abierto."""
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

    # Se cierran las ventanas abiertas por OpenCV (el video lo libera abrir_video)
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