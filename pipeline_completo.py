"""
Pipeline completo: reflejo fotomotor a partir de un video del ojo.

Entrada : el video indicado en RUTA_VIDEO (o el que se pase al ejecutar)
Salida  : resultados/curva_temporal.csv           (diámetro y área pupilar frame a frame)
          resultados/parametros_fisiologicos.csv  (latencia, amplitud, velocidades)
          y una copia de ambos en resultados/por_video/<nombre del video>/
          resultados/resultados_modulo1/<video>_mejoramiento.png  (pasos del mejoramiento)

Flujo: importación (módulo 0) -> ROI y mejoramiento (módulo 1) -> segmentación (módulo 2)
       -> curva y parámetros (módulo 3). En el video, cada frame se mejora con el módulo 1
       y la imagen mejorada es la que segmenta el módulo 2.

Uso:
    python pipeline_completo.py                      # usa RUTA_VIDEO
    python pipeline_completo.py registro_A2.MOV      # otro video de la carpeta videos/

Video nuevo: lo único que se pide es el ROI. Se abre una ventana con un frame del video;
se arrastra un rectángulo que encierre el iris con un poco de margen y se aprieta Enter
(c cancela). Queda guardado en anotaciones_manuales/roi.json y el resto corre solo hasta
el final. Si en ese frame el ojo está cerrado, cancelar e indicar otro en FRAME_REFERENCIA.
"""

import sys
import time
import traceback
from pathlib import Path

from modulos import interfaz as ui
from modulos import modulo0_importacion as m0
from modulos import modulo1_mejoramiento as m1
from modulos import modulo2_segmentacion as m2
from modulos import modulo3_caracteristicas as m3

# ============================================================
# CONFIGURACIÓN (lo único que hace falta tocar)
# ============================================================

RUTA_VIDEO = "videos/registro_A1.MOV"     # relativa a la carpeta del proyecto, o absoluta
CARPETA_RESULTADOS = "resultados"

# True: muestra el video y los gráficos (hay que cerrar cada ventana para seguir).
# False: corre de punta a punta sin ventanas.
MOSTRAR_PASOS = False

# True: reutiliza el ROI guardado en anotaciones_manuales/roi.json para este video; si el
#       video es nuevo (no tiene ROI guardado), abre la ventana para marcarlo una vez.
# False: pide marcar el ROI con el mouse en cada corrida.
USAR_ROI_GUARDADO = True

# Frame sobre el que se marca el ROI y desde el que se mide la escala (por si el video
# empieza con un parpadeo). None: el elegido al marcar el ROI en la aplicación, o el primero.
FRAME_REFERENCIA = None

# Escala en mm por píxel, si se calibró (por ejemplo, con la regla filmada junto al ojo).
# None: los resultados quedan en píxeles y en porcentajes.
MM_POR_PX = None

# ============================================================

RAIZ = Path(__file__).resolve().parent
EXTENSIONES_VIDEO = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def resolver_video(ruta):
    """Ruta absoluta del video: tal cual, relativa a la carpeta del proyecto o dentro de videos/."""
    ruta = Path(ruta)
    for candidata in (ruta, RAIZ / ruta, RAIZ / "videos" / ruta):
        if candidata.is_file():
            return candidata.resolve()
    disponibles = sorted(p.name for p in (RAIZ / "videos").glob("*") if p.suffix.lower() in EXTENSIONES_VIDEO)
    mensaje = f"No se encontró el video '{ruta}'."
    if disponibles:
        mensaje += " Videos disponibles en videos/: " + ", ".join(disponibles)
    raise FileNotFoundError(mensaje)


def ejecutar_pipeline(ruta_video=RUTA_VIDEO, frame_referencia=None):
    inicio = time.perf_counter()
    ruta = resolver_video(ruta_video)
    carpeta = RAIZ / CARPETA_RESULTADOS
    carpeta_video = carpeta / "por_video" / ruta.stem

    ui.encabezado("PUPILOMETRÍA · Reflejo fotomotor",
                  "Procesamiento de Señales e Imágenes Biomédicas · ITBA",
                  {"Video": ruta.name,
                   "Resultados": f"{CARPETA_RESULTADOS}/",
                   "Ventanas": "sí" if MOSTRAR_PASOS else "no",
                   "ROI": "el guardado; si no hay, se marca a mano" if USAR_ROI_GUARDADO else "marcar a mano",
                   "Inicio": ui.fecha_hora()})
    total = 6

    # 1. Importación del video ----------------------------------------------------------
    with ui.paso(1, total, "Importación del video (Módulo 0)"):
        video, metadatos = m0.abrir_video(ruta)
        ui.tabla({"Resolución": f"{metadatos['ancho']} x {metadatos['alto']} px",
                  "Frecuencia": ui.numero(metadatos["fps"], 2, "fps"),
                  "Frames": metadatos["cantidad_frames"],
                  "Duración": ui.numero(metadatos["duracion_s"], 2, "s"),
                  "Códec": metadatos["codec"]})
        if MOSTRAR_PASOS:
            ui.info("Se abre el video (Espacio: pausa | q: cerrar) y después los frames de muestra.")
            m0.reproducir_video(ruta)
            m0.mostrar_frames_muestra(video, metadatos)
        video.release()

    # 2. ROI y mejoramiento -------------------------------------------------------------
    figura_mejora = carpeta / "resultados_modulo1" / f"{ruta.stem}_mejoramiento.png"
    with ui.paso(2, total, "ROI y mejoramiento (Módulo 1)"):
        nuevo_roi = m2.PEDIR_ROI_SIEMPRE and not USAR_ROI_GUARDADO
        if nuevo_roi or m2.roi_guardado(ruta.name) is None:
            # Única intervención: marcar el ROI (p. ej., la cátedra con un video nuevo)
            ui.info("Hay que marcar el ROI: encerrar el iris con margen y apretar Enter (c cancela).")
            ui.info("Si en ese frame el ojo está cerrado, cancelar e indicar otro en FRAME_REFERENCIA.")
        preparacion = m2.preparar(ruta, nuevo_roi=nuevo_roi,
                                  frame_referencia=FRAME_REFERENCIA if frame_referencia is None else frame_referencia)
        k = preparacion["frame_referencia"]
        recorte, mejora, _ = m2.frame_mejorado(ruta, preparacion)   # frame de referencia, para mostrar
        f2 = "sin contraste en este frame" if mejora["umbral"] is None else mejora["umbral"]
        ui.tabla({"Frame de referencia": f"{k} (t = {ui.numero(k / metadatos['fps'], 2, 's')})",
                  "ROI procesado": f"{recorte.shape[1]} x {recorte.shape[0]} px, en escala de grises",
                  "Suavizado": f"gaussiano de {mejora['ventana']} x {mejora['ventana']} px",
                  "Rango de interés": f"[0, {f2}] (F2 por Otsu multinivel, se recalcula en cada frame)"})
        if MOSTRAR_PASOS:
            ui.info("Se abre la figura del mejoramiento (cerrarla para continuar).")
        m1.mostrar_mejoramiento(recorte, mejora, titulo=f"{ruta.name}: mejoramiento del ROI (frame {k})",
                                destino=figura_mejora, mostrar=MOSTRAR_PASOS)

    # 3. Segmentación frame a frame -------------------------------------------------------
    with ui.paso(3, total, "Segmentación de la pupila frame a frame (Módulo 2)"):
        ui.info("Cada frame: mejoramiento (Módulo 1) y segmentación de la imagen mejorada (Módulo 2)")
        filas, r_iris = m2.ejecutar_modulo2(ruta, preparacion=preparacion)

    # 4. Curva temporal y artefactos ------------------------------------------------------
    with ui.paso(4, total, "Curva temporal y artefactos (Módulo 3)"):
        curva = m3.curva_desde_modulo2(filas, ruta, r_iris)
        curva = m3.limpiar_curva(curva)

    # 5. Estímulo y parámetros fisiológicos -----------------------------------------------
    with ui.paso(5, total, "Estímulo y parámetros fisiológicos (Módulo 3)"):
        t_estimulo, tipo = m3.detectar_estimulo(curva)
        parametros = m3.parametros_fisiologicos(curva, t_estimulo, tipo, mm_por_px=MM_POR_PX)
        if MOSTRAR_PASOS:
            ui.info("Se abre el gráfico de la curva temporal (cerrarlo para continuar).")
            m3.mostrar_curva(curva, parametros=parametros, titulo=ruta.name)

    # 6. Exportación ----------------------------------------------------------------------
    with ui.paso(6, total, "Exportación de resultados"):
        m3.guardar_resultados(curva, parametros, carpeta)          # nombres fijos (última corrida)
        m3.guardar_resultados(curva, parametros, carpeta_video)    # copia por video

    ui.resumen_parametros(parametros)
    ui.archivos([carpeta / "curva_temporal.csv", carpeta / "parametros_fisiologicos.csv",
                 carpeta_video, figura_mejora, m2.CARPETA_RESULTADOS], raiz=RAIZ)
    ui.cierre(time.perf_counter() - inicio)
    return curva, parametros


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else RUTA_VIDEO
    try:
        ejecutar_pipeline(ruta)
    except KeyboardInterrupt:
        print()
        ui.aviso("Ejecución interrumpida por el usuario")
        sys.exit(130)
    except FileNotFoundError as e:   # Error esperable: alcanza con el mensaje
        print()
        ui.error(str(e))
        sys.exit(1)
    except Exception as e:
        print()
        ui.error(str(e) or type(e).__name__)
        print("\n  Detalle técnico:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()