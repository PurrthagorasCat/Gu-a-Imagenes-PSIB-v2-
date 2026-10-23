# Pupilometría: reflejo fotomotor por video

TP de Procesamiento de Señales e Imágenes Biomédicas (ITBA).

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `pipeline_completo.py` | Pipeline de punta a punta: video → CSV. Contiene la variable `RUTA_VIDEO` |
| `videos/` | Registros A (contracción) y B (redilatación) |
| `anotaciones_manuales/` | Contornos pupilares anotados a mano (referencia) |
| `modulos/` | Un archivo por módulo de la consigna, más las métricas compartidas |
| `resultados/` | Curva temporal y parámetros fisiológicos en CSV |
| `librerias.txt` | Librerías de Python necesarias |

## Uso

```bash
pip install -r librerias.txt
# editar RUTA_VIDEO en pipeline_completo.py
python pipeline_completo.py
```
