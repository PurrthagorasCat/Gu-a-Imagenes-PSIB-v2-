"""
Pipeline completo

Entrada : video indicado en RUTA_VIDEO
Salida  : resultados/curva_temporal.csv          (área/diámetro pupilar frame a frame)
          resultados/parametros_fisiologicos.csv (latencia, amplitud, velocidades)
"""

import os
import sys
import matplotlib
import matplotlib.pyplot as plt

from modulos import modulo0_importacion as m0
from modulos import modulo3_caracteristicas as m3

# El módulo 2 no es un paquete: se agrega su carpeta al path para importarlo.
# Al importarse desactiva las ventanas de matplotlib (backend "Agg"), así que se
# guarda el backend actual para restaurarlo después
_backend = matplotlib.get_backend()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modulos", "Modulo2_segmentacion"))
import procesar_video as m2  # noqa: E402
plt.switch_backend(_backend)

RUTA_VIDEO = "videos/registro_A1.MOV"
CARPETA_RESULTADOS = "resultados/"

# True: muestra el resultado de cada paso.  False: corre sin ventanas (entrega final)
MOSTRAR_PASOS = True


def ejecutar_pipeline():
    # El módulo 2 interpreta las rutas relativas desde su propia carpeta: se pasa la absoluta
    ruta = os.path.abspath(RUTA_VIDEO)

    # 1. Importación del video
    video, metadatos = m0.abrir_video(ruta)
    m0.imprimir_metadatos(metadatos)
    if MOSTRAR_PASOS:
        m0.reproducir_video(ruta)
        m0.mostrar_frames_muestra(video, metadatos)
    video.release()

    # 2-3. ROI, segmentación y limpieza morfológica frame a frame -> módulo 2
    filas, r_iris = m2.procesar_video(ruta)

    # 4. Curva temporal y artefactos                -> modulos/modulo3_caracteristicas.py
    curva = m3.curva_desde_modulo2(filas, ruta, r_iris)
    curva = m3.limpiar_curva(curva)

    # 5. Parámetros fisiológicos                    -> modulos/modulo3_caracteristicas.py
    t_estimulo, tipo = m3.detectar_estimulo(curva)
    parametros = m3.parametros_fisiologicos(curva, t_estimulo, tipo,
                                            mm_por_px=m3.mm_por_px_desde_iris(r_iris))
    m3.imprimir_parametros(parametros)
    if MOSTRAR_PASOS:
        m3.mostrar_curva(curva, parametros=parametros)

    # 6. Exportar CSV
    m3.guardar_resultados(curva, parametros, CARPETA_RESULTADOS)


if __name__ == "__main__":
    ejecutar_pipeline()
