"""
Módulo 0 — Importación de datos.

Apertura de los videos (.MOV), lectura de metadatos y frames, y visualización del registro.
"""

import cv2                       # OpenCV: lectura de video, procesamiento de imágenes y ventanas
import numpy as np               # Cálculo numérico (arrays, linspace, ceil)
import matplotlib.pyplot as plt  # Gráficos para mostrar frames sueltos


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
