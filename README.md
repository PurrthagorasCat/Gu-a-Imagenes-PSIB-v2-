# Pupilometría: reflejo fotomotor por video

Trabajo práctico de Procesamiento de Señales e Imágenes Biomédicas (ITBA).

A partir de un video del ojo, el programa segmenta la pupila en cada frame, arma la curva del diámetro en el tiempo, detecta el estímulo luminoso y calcula los parámetros del reflejo fotomotor: latencia, amplitud y velocidades de constricción y redilatación.

## Cómo usarlo

Desde la carpeta del proyecto, abrir la aplicación:

```bash
python app_pupilometria.py
```

La aplicación indica cada paso. Hay que elegir el video, marcar la región de interés con un rectángulo alrededor del iris y apretar "Procesar video". Cuando termina muestra los parámetros y la curva, y deja los resultados en la carpeta `resultados/`.

### Región de interés

La primera vez que se procesa un video se pide marcar la región de interés: se arrastra un rectángulo que encierre el iris con un poco de margen. Si en ese frame el ojo está cerrado, con las flechas se puede pasar a otro frame óptimo. El ROI queda guardado en `anotaciones_manuales/roi.json`, así que las corridas siguientes de ese video lo usan sin preguntar. De todos modos, se puede cambiar el ROI cuando se quiera con el botón "Cambiar ROI o frame…".

### Resultados

| Archivo | Contenido |
|---|---|
| `resultados/curva_temporal.csv` | diámetro y área pupilar frame a frame (medidos, limpios y suavizados), brillo y artefactos |
| `resultados/parametros_fisiologicos.csv` | tipo de respuesta, estímulo, latencia, amplitud, velocidades y recuperación |
| `resultados/por_video/<video>/` | copia de los dos CSV para cada video procesado |

Los valores quedan en píxeles y en porcentajes. Si se tiene la escala calibrada, se puede cargar esa escala en `MM_POR_PX`, al comienzo de `pipeline_completo.py`.

En la aplicación, al pasar el cursor por la curva se ve el frame de ese instante con el contorno detectado: verde si el frame se aceptó y rojo si se rechazó.

### Desde la consola

`pipeline_completo.py` hace lo mismo que la aplicación, pero sin interfaz:

```bash
python pipeline_completo.py registro_A1.MOV
```

Busca el video en `videos/`, aunque también acepta una ruta completa. Sin argumento, usa el que está en `RUTA_VIDEO`, al comienzo del archivo. Si el video no tiene ROI guardado, abre una ventana para marcarlo y después sigue solo hasta el final. Para volver a marcarlo, poner `USAR_ROI_GUARDADO = False`.

## Notebooks de análisis

| Notebook | Contenido |
|---|---|
| `analisis_modulo1_1.ipynb` | Módulo 1.1: carga, muestreo, cuantización y espacios de color |
| `analisis_modulo1_2.ipynb` | Módulo 1.2: histograma, transformaciones de intensidad, ruido y filtrado espacial |
| `analisis_modulo1_3.ipynb` | Módulo 1.3: bordes, Fourier, PSD y filtrado en frecuencia |
| `analisis_pipeline_completo.ipynb` | El programa paso a paso (módulos 0 a 4), con las alternativas que descartamos, la validación contra la anotación manual, los descriptores, la textura y la compresión |

Se abren desde la carpeta del proyecto. Conviene no volver a correr los notebooks porque muchos análisis usan los datos de la corrida nuestra. Sin embargo, ya probamos de correrlo varias veces, y los datos se mantienen razonablemente iguales, por lo que no habría problema. En caso de querer correr los notebooks, conviene correr primero el 1.1, porque guarda `frame0.png` y `roi_frame0.png`, que después usan el 1.2 y el 1.3.

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `app_pupilometria.py` | Aplicación gráfica |
| `pipeline_completo.py` | El mismo programa sin interfaz, del video a los CSV |
| `modulos/modulo0_importacion.py` | Apertura del video, metadatos, frames y selección del ROI |
| `modulos/modulo1_mejoramiento.py` | Mejoramiento: escala de grises, suavizado y rango de interés, más las funciones del módulo 1 que usan los notebooks |
| `modulos/modulo2_segmentacion.py` | Segmentación (Otsu, K-means, snake, watershed), morfología y seguimiento de la pupila en el video |
| `modulos/modulo3_caracteristicas.py` | Descriptores, textura, curva temporal, artefactos, estímulo y parámetros fisiológicos |
| `modulos/modulo4_compresion.py` | Huffman, RLE y JPEG: ratio, PSNR/SSIM y artefactos de bloque |
| `modulos/metricas.py` | Métricas compartidas: PSNR, SSIM, Dice, IoU, MAE y RMSE |
| `modulos/interfaz.py` | Formato de la salida por consola |
| `anotaciones_manuales/` | ROI de cada video (`roi.json`), frame sobre el que se marcó (`frames_referencia.json`) y contornos de la pupila anotados a mano para la validación (`referencias.json`) |
| `videos/` | Videos de entrada (no están en el repositorio porque pesan demasiado) |
| `resultados/` | CSV del programa y figuras de los módulos 1 y 2, incluido el video de control con el contorno en cada frame |
| `imagenes/` | Capturas usadas en el notebook |
| `librerias.txt` | Librerías de Python necesarias |

El ROI se marca a mano a propósito: no usamos detección automática del ojo ni modelos entrenados, y todo el procesamiento se hace dentro de esa región.