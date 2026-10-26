"""
Interfaz de consola del pipeline: encabezados, pasos numerados, avisos y resumen final.

Solo da formato a lo que se imprime; no hace ningún procesamiento.
"""

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

ANCHO = 66


# ------------------------------------------------------------
# Colores (solo si la salida es una terminal que los soporta)
# ------------------------------------------------------------

def _soporta_color():
    if os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        os.system("")   # Activa los códigos de color en la consola de Windows
    return True


_COLOR = _soporta_color()
_CODIGOS = {"negrita": "1", "tenue": "2", "verde": "32", "amarillo": "33", "rojo": "31", "cian": "36"}


def _estilo(texto, *estilos):
    if not _COLOR or not estilos:
        return texto
    return "\033[" + ";".join(_CODIGOS[e] for e in estilos) + "m" + texto + "\033[0m"


# ------------------------------------------------------------
# Formato
# ------------------------------------------------------------

def duracion(segundos):
    """12,3 s  |  2 min 05 s"""
    if segundos < 60:
        return f"{segundos:.1f} s".replace(".", ",")
    minutos, seg = divmod(int(round(segundos)), 60)
    return f"{minutos} min {seg:02d} s"


def numero(valor, decimales=1, unidad=""):
    """Número con coma decimal; '—' si no se pudo calcular."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "—"
    texto = f"{valor:.{decimales}f}".replace(".", ",")
    return f"{texto} {unidad}".strip()


# ------------------------------------------------------------
# Bloques de salida
# ------------------------------------------------------------

def encabezado(titulo, subtitulo=None, datos=None):
    """Título del programa con los datos de la corrida."""
    print()
    print(_estilo("═" * ANCHO, "cian"))
    print(_estilo(f"  {titulo}", "negrita"))
    if subtitulo:
        print(_estilo(f"  {subtitulo}", "tenue"))
    print(_estilo("═" * ANCHO, "cian"))
    if datos:
        tabla(datos)
    print()


_en_paso = False   # Dentro de un paso, toda la salida ya lleva sangría (ver paso)


def _sangria():
    return "" if _en_paso else "  "


def tabla(datos):
    """Pares clave: valor alineados."""
    ancho = max(len(k) for k in datos)
    for clave, valor in datos.items():
        print(_sangria() + f"{clave:<{ancho}} : {valor}")


def info(texto):
    print(_sangria() + texto)


def ok(texto):
    print(_estilo(_sangria() + f"✔ {texto}", "verde"))


def aviso(texto):
    print(_estilo(_sangria() + f"⚠ {texto}", "amarillo"))


def error(texto):
    print(_estilo(_sangria() + f"✖ {texto}", "rojo", "negrita"))


class _ConSangria:
    """Envuelve la salida estándar y agrega una sangría al comienzo de cada línea."""

    def __init__(self, destino, prefijo="  "):
        self.destino, self.prefijo, self.inicio_de_linea = destino, prefijo, True

    def write(self, texto):
        for parte in texto.splitlines(keepends=True):
            if self.inicio_de_linea and parte.strip():
                self.destino.write(self.prefijo)
            self.destino.write(parte)
            self.inicio_de_linea = parte.endswith("\n")
        return len(texto)

    def flush(self):
        self.destino.flush()

    def __getattr__(self, nombre):
        return getattr(self.destino, nombre)


@contextmanager
def paso(numero_paso, total, titulo):
    """
    Encabezado de un paso del pipeline. Todo lo que se imprime adentro (también lo que
    imprimen los módulos) sale con sangría, y al terminar se informa cuánto tardó.
    Si el paso falla, lo informa y deja pasar el error.
    """
    global _en_paso
    print(_estilo(f"[{numero_paso}/{total}] {titulo}", "negrita"))
    print(_estilo("─" * ANCHO, "tenue"))
    inicio = time.perf_counter()
    salida_original, _en_paso = sys.stdout, True
    sys.stdout = _ConSangria(salida_original)
    try:
        yield
    except Exception:
        error(f"Falló el paso {numero_paso}: {titulo}")
        raise
    else:
        ok(f"Listo ({duracion(time.perf_counter() - inicio)})")
    finally:
        sys.stdout, _en_paso = salida_original, False
    print()


def resumen_parametros(parametros):
    """Imprime los parámetros fisiológicos principales, con nombres y unidades legibles."""
    print(_estilo("═" * ANCHO, "cian"))
    print(_estilo("  RESULTADOS", "negrita"))
    print(_estilo("═" * ANCHO, "cian"))
    datos = parametros_legibles(parametros)
    if not datos:
        aviso("No se pudo detectar el estímulo: no hay parámetros fisiológicos")
        return
    tabla(datos)
    print()


def parametros_legibles(parametros):
    """{nombre legible: valor con unidad} de los parámetros principales; vacío si no hay estímulo."""
    p = parametros
    contraccion = p["tipo"] == "contraccion"
    if p["tipo"] not in ("contraccion", "dilatacion"):
        return {}

    datos = {
        "Tipo de registro": "contracción (luz encendida)" if contraccion else "dilatación (luz apagada)",
        "Estímulo": numero(p["t_estimulo_s"], 2, "s"),
        "Latencia": numero(1000 * p["latencia_s"], 0, "ms"),
        "Diámetro basal → extremo": f"{numero(p['diametro_basal_px'])} → {numero(p['diametro_extremo_px'])} px",
        "Amplitud (diámetro)": numero(p["amplitud_diametro_%"], 1, "%"),
        "Amplitud (área)": numero(p["amplitud_area_%"], 1, "%"),
        "Tiempo de respuesta": numero(p["tiempo_respuesta_s"], 2, "s"),
    }
    if contraccion:
        datos["Velocidad de constricción"] = (f"máx {numero(p['velocidad_constriccion_max_px_s'])} | "
                                              f"media {numero(p['velocidad_constriccion_media_px_s'])} px/s")
    datos["Velocidad de redilatación"] = (f"máx {numero(p['velocidad_redilatacion_max_px_s'])} | "
                                          f"media {numero(p['velocidad_redilatacion_media_px_s'])} px/s")
    if contraccion:
        datos["Recuperación"] = numero(p["recuperacion_%"], 1, "%")
    if np.isfinite(p.get("mm_por_px", np.nan)):
        datos["Escala"] = numero(p["mm_por_px"], 4, "mm/px")
    return datos


def archivos(rutas, raiz=None):
    """Lista de archivos y carpetas generados (rutas relativas a `raiz` si se da)."""
    print(_estilo("  Archivos generados:", "negrita"))
    for ruta in rutas:
        texto = os.path.relpath(ruta, raiz) if raiz else str(ruta)
        if os.path.isdir(ruta):
            texto += os.sep
        print(f"    {texto}")


def cierre(segundos):
    print()
    print(_estilo(f"  Tiempo total: {duracion(segundos)}", "tenue"))
    print(_estilo("═" * ANCHO, "cian"))
    print()


def fecha_hora():
    return datetime.now().strftime("%d/%m/%Y %H:%M")
