"""
Pipeline completo

Entrada : video indicado en RUTA_VIDEO
Salida  : resultados/curva_temporal.csv          (área/diámetro pupilar frame a frame)
          resultados/parametros_fisiologicos.csv (latencia, amplitud, velocidades)
"""

RUTA_VIDEO = "videos/registro_A1.mp4"
CARPETA_RESULTADOS = "resultados/"

def ejecutar_pipeline():
    # 1. Importar video y extraer frames          -> modulos/modulo0_importacion.py
    # 2. Preprocesamiento de cada frame           -> modulos/modulo1_mejoramiento.py
    # 3. Segmentación y limpieza morfológica      -> modulos/modulo2_segmentacion.py
    # 4. Área/diámetro por frame y artefactos     -> modulos/modulo3_caracteristicas.py
    # 5. Parámetros fisiológicos                  -> modulos/modulo3_caracteristicas.py
    # 6. Exportar CSV
    pass


if __name__ == "__main__":
    ejecutar_pipeline()
