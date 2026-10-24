"""
Pipeline completo

Entrada : video indicado en RUTA_VIDEO
Salida  : resultados/curva_temporal.csv          (área/diámetro pupilar frame a frame)
          resultados/parametros_fisiologicos.csv (latencia, amplitud, velocidades)
"""

from modulos import modulo0_importacion as m0
 
RUTA_VIDEO = "videos/registro_A1.MOV"
CARPETA_RESULTADOS = "resultados/"
 
# True: muestra el resultado de cada paso.  False: corre sin ventanas (entrega final)
MOSTRAR_PASOS = True
 
def ejecutar_pipeline():
    # 1. Importación del video
    video, metadatos = m0.abrir_video(RUTA_VIDEO)
    m0.imprimir_metadatos(metadatos)
    if MOSTRAR_PASOS:
        m0.reproducir_video(RUTA_VIDEO)
        m0.mostrar_frames_muestra(video, metadatos)
 
    # 2. Preprocesamiento de cada frame           -> modulos/modulo1_mejoramiento.py
    # 3. Segmentación y limpieza morfológica      -> modulos/modulo2_segmentacion.py
    # 4. Área/diámetro por frame y artefactos     -> modulos/modulo3_caracteristicas.py
    # 5. Parámetros fisiológicos                  -> modulos/modulo3_caracteristicas.py
    # 6. Exportar CSV
 
    video.release()
 
 
if __name__ == "__main__":
    ejecutar_pipeline()
 
