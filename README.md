# Pupilometría: reflejo fotomotor por video

Trabajo práctico de Procesamiento de Señales e Imágenes Biomédicas (ITBA).

A partir de un video del ojo, el programa segmenta la pupila en cada frame, arma la curva del diámetro en el tiempo, detecta el estímulo luminoso y calcula los parámetros del reflejo fotomotor: latencia, amplitud y velocidades de constricción y redilatación.

## Instalación

Probado con Python 3.12 y 3.13.

```bash
pip install -r librerias.txt
```

La aplicación gráfica usa `tkinter`, que viene con Python. En macOS con Python de Homebrew hay que instalarlo aparte (`brew install python-tk`).

Los videos no están en el repositorio porque pesan más de lo que acepta GitHub: hay que copiarlos en la carpeta `videos/`.

## Uso

### Pipeline completo (lo que pide la competencia)

El video de entrada se define en una sola variable, `RUTA_VIDEO`, al comienzo de `pipeline_completo.py`. También se puede pasar al ejecutar:

```bash
python pipeline_completo.py                      # usa RUTA_VIDEO
python pipeline_completo.py registro_A2.MOV      # otro video de videos/
python pipeline_completo.py /ruta/a/otro.mp4     # o cualquier ruta
```

La primera vez que se procesa un video se abre una ventana para marcar la región de interés: se arrastra un rectángulo que encierre el iris con margen y se confirma con Enter. El ROI queda guardado en `anotaciones_manuales/roi.json` y las siguientes corridas lo reutilizan sin preguntar.

Salida, en formato fijo:

| Archivo | Contenido |
|---|---|
| `resultados/curva_temporal.csv` | diámetro y área pupilar frame a frame (medidos, limpios y suavizados), brillo y artefactos |
| `resultados/parametros_fisiologicos.csv` | tipo de respuesta, estímulo, latencia, amplitud, velocidades y recuperación |
| `resultados/por_video/<video>/` | copia de los dos CSV para cada video procesado |

Los resultados quedan en píxeles y en porcentajes. Si se calibró la escala, se puede indicar en `MM_POR_PX`.

### Aplicación gráfica

```bash
python app_pupilometria.py
```

Corre el mismo pipeline desde una ventana: se elige el video, se marca el ROI (con flechas para pasar a otro frame si en el primero el ojo está cerrado), se ve el mejoramiento del módulo 1 y se procesa. Al terminar muestra los parámetros y la curva; al pasar el cursor por el gráfico se ve el frame de ese instante con el contorno detectado.

### Notebooks de análisis

| Notebook | Contenido |
|---|---|
| `analisis_modulo1_1.ipynb` | Módulo 1.1: carga, muestreo, cuantización y espacios de color |
| `analisis_modulo1_2.ipynb` | Módulo 1.2: histograma, transformaciones de intensidad, ruido y filtrado espacial |
| `analisis_modulo1_3.ipynb` | Módulo 1.3: bordes, Fourier, PSD y filtrado en frecuencia |
| `analisis_pipeline_completo.ipynb` | El programa paso a paso (módulos 0 a 4), con las alternativas que descartamos, la validación contra la anotación manual, los descriptores y la textura, y la compresión |

Los notebooks se ejecutan desde la carpeta del proyecto. El 1.1 guarda `frame0.png` y `roi_frame0.png`, que usan el 1.2 y el 1.3.

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `pipeline_completo.py` | Pipeline de punta a punta: video → CSV. Contiene la variable `RUTA_VIDEO` |
| `app_pupilometria.py` | Aplicación gráfica del pipeline |
| `modulos/modulo0_importacion.py` | Apertura del video, metadatos, frames y selección del ROI |
| `modulos/modulo1_mejoramiento.py` | Mejoramiento: escala de grises, suavizado y rango de interés (y las funciones del módulo 1 que usan los notebooks) |
| `modulos/modulo2_segmentacion.py` | Segmentación (Otsu, K-means, snake, watershed), morfología y seguimiento de la pupila en el video |
| `modulos/modulo3_caracteristicas.py` | Descriptores, textura, curva temporal, artefactos, estímulo y parámetros fisiológicos |
| `modulos/modulo4_compresion.py` | Huffman, RLE y JPEG: ratio, PSNR/SSIM y artefactos de bloque |
| `modulos/metricas.py` | Métricas compartidas: PSNR, SSIM, Dice, IoU, MAE y RMSE |
| `modulos/interfaz.py` | Formato de la salida por consola |
| `anotaciones_manuales/` | ROI de cada video (`roi.json`), frame sobre el que se marcó (`frames_referencia.json`) y contornos de la pupila anotados a mano para la validación (`referencias.json`) |
| `videos/` | Videos de entrada (no se suben al repositorio) |
| `resultados/` | CSV del pipeline y figuras de los módulos 1 y 2, incluido el video de control con el contorno en cada frame |
| `imagenes/` | Capturas usadas en el notebook |
| `librerias.txt` | Librerías de Python necesarias |

No usamos detección automática del ojo ni modelos entrenados: el ROI se marca a mano y todo el procesamiento se hace dentro de él.
