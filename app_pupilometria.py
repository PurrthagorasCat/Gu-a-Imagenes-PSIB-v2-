"""
Aplicación gráfica del pipeline de pupilometría.

    python app_pupilometria.py

1. Elegir el video con el explorador de archivos.
2. Marcar la región de interés (el iris con margen) arrastrando un rectángulo.
3. Ver la ROI mejorada (módulo 1): es la imagen que recibe la segmentación.
4. Procesar: corre el pipeline completo (el mismo de pipeline_completo.py) en segundo plano,
   muestra el avance y, al terminar, los parámetros fisiológicos y la curva temporal.
"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")   # Los módulos dibujan en segundo plano; acá los gráficos se insertan con FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import cv2
import numpy as np
from PIL import Image, ImageTk

import pipeline_completo as pipeline
from modulos import interfaz as ui
from modulos import modulo0_importacion as m0
from modulos import modulo1_mejoramiento as m1
from modulos import modulo2_segmentacion as m2

RAIZ = pipeline.RAIZ
TIPOS_VIDEO = [("Videos", "*.mov *.MOV *.mp4 *.MP4 *.avi *.AVI *.mkv *.m4v"), ("Todos los archivos", "*.*")]
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRIS = "#52514e"
ANSI = re.compile(r"\x1b\[[0-9;]*m")   # Códigos de color de la terminal


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def a_foto(rgb, ancho_max, alto_max, ampliar_hasta=1.0):
    """Imagen RGB (numpy) -> PhotoImage que entra en ancho_max x alto_max, y la escala usada."""
    alto, ancho = rgb.shape[:2]
    escala = min(ancho_max / ancho, alto_max / alto, ampliar_hasta)
    tam = (max(1, int(ancho * escala)), max(1, int(alto * escala)))
    return ImageTk.PhotoImage(Image.fromarray(rgb).resize(tam, Image.LANCZOS)), escala


def abrir_en_sistema(ruta):
    """Abre un archivo o carpeta con la aplicación predeterminada del sistema."""
    if sys.platform == "win32":
        os.startfile(ruta)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(ruta)], check=False)
    else:
        subprocess.run(["xdg-open", str(ruta)], check=False)


class SalidaACola:
    """Reemplaza a la salida estándar: manda el texto a una cola que lee la ventana."""

    def __init__(self, cola):
        self.cola = cola

    def write(self, texto):
        if texto:
            self.cola.put(("texto", ANSI.sub("", texto)))
        return len(texto)

    def flush(self):
        pass

    def isatty(self):
        return False


# ------------------------------------------------------------
# Ventana para marcar el ROI
# ------------------------------------------------------------

class SelectorROI(tk.Toplevel):
    """Muestra el primer frame y deja marcar el ROI arrastrando un rectángulo con el mouse."""

    def __init__(self, padre, frame_rgb, roi=None):
        super().__init__(padre)
        self.title("Marcar la región de interés")
        self.transient(padre)
        self.resizable(False, False)
        self.resultado = None
        self._alto, self._ancho = frame_rgb.shape[:2]

        # El frame se achica para que entre en la pantalla
        self.foto, self.escala = a_foto(frame_rgb, int(self.winfo_screenwidth() * 0.8),
                                        int(self.winfo_screenheight() * 0.72))

        ttk.Label(self, text="Arrastrá un rectángulo que encierre el IRIS con un poco de margen.",
                  style="Dato.TLabel", padding=(12, 10, 12, 4)).pack(anchor="w")
        ttk.Label(self, text="Podés volver a arrastrar para corregirlo. Enter confirma, Esc cancela.",
                  style="Sub.TLabel", padding=(12, 0, 12, 8)).pack(anchor="w")

        self.lienzo = tk.Canvas(self, width=self.foto.width(), height=self.foto.height(),
                                cursor="crosshair", highlightthickness=0)
        self.lienzo.pack(padx=12)
        self.lienzo.create_image(0, 0, anchor="nw", image=self.foto)
        self.rectangulo = None
        self.inicio = None
        if roi:
            x, y, w, h = (v * self.escala for v in roi)
            self._dibujar(x, y, x + w, y + h)

        self.lienzo.bind("<ButtonPress-1>", self._presionar)
        self.lienzo.bind("<B1-Motion>", self._arrastrar)
        self.lienzo.bind("<ButtonRelease-1>", self._arrastrar)

        pie = ttk.Frame(self, padding=12)
        pie.pack(fill="x")
        self.etiqueta = ttk.Label(pie, style="Sub.TLabel")
        self.etiqueta.pack(side="left")
        ttk.Button(pie, text="Cancelar", command=self.destroy).pack(side="right")
        ttk.Button(pie, text="Confirmar", style="Accion.TButton", command=self.confirmar).pack(side="right", padx=8)
        self.bind("<Return>", lambda _: self.confirmar())
        self.bind("<Escape>", lambda _: self.destroy())
        self._actualizar_etiqueta()

        self.grab_set()   # Ventana modal: hay que confirmar o cancelar para volver
        self.focus_set()

    def _limitar(self, x, y):
        return min(max(x, 0), self.foto.width() - 1), min(max(y, 0), self.foto.height() - 1)

    def _dibujar(self, x0, y0, x1, y1):
        if self.rectangulo is None:
            self.rectangulo = self.lienzo.create_rectangle(x0, y0, x1, y1, outline=COLORES[1], width=2)
        else:
            self.lienzo.coords(self.rectangulo, x0, y0, x1, y1)

    def _presionar(self, evento):
        self.inicio = self._limitar(evento.x, evento.y)
        self._dibujar(*self.inicio, *self.inicio)

    def _arrastrar(self, evento):
        if self.inicio is not None:
            self._dibujar(*self.inicio, *self._limitar(evento.x, evento.y))
            self._actualizar_etiqueta()

    def _roi_original(self):
        """Rectángulo actual en píxeles del video, o None si no hay uno válido."""
        if self.rectangulo is None:
            return None
        x0, y0, x1, y1 = self.lienzo.coords(self.rectangulo)
        x, y = min(x0, x1) / self.escala, min(y0, y1) / self.escala
        w, h = abs(x1 - x0) / self.escala, abs(y1 - y0) / self.escala
        if w < 10 or h < 10:
            return None
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    def _actualizar_etiqueta(self):
        roi = self._roi_original()
        self.etiqueta.configure(text="Sin rectángulo" if roi is None else
                                f"ROI: x = {roi[0]}, y = {roi[1]}, {roi[2]} x {roi[3]} px")

    def confirmar(self):
        roi = self._roi_original()
        if roi is None:
            messagebox.showwarning("Falta el ROI", "Arrastrá un rectángulo sobre el ojo antes de confirmar.",
                                   parent=self)
            return
        self.resultado = roi
        self.destroy()


# ------------------------------------------------------------
# Ventana principal
# ------------------------------------------------------------

class Aplicacion(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Pupilometría · Reflejo fotomotor")
        self.geometry("1220x870")
        self.minsize(1050, 800)
        self._estilos()

        self.ruta = self.metadatos = self.frame0 = self.roi = None
        self.cola = queue.Queue()
        self.hilo = None
        self.cola_mejora = queue.Queue()   # Resultados del mejoramiento calculado en segundo plano
        self.mejorando = False
        self._version_mejora = 0           # Para descartar resultados de un ROI anterior
        self.paso_actual = 0
        self.total_pasos = 6
        self._resto_de_linea = ""
        self._inicio = 0.0

        self._armar_ventana()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

        inicial = RAIZ / pipeline.RUTA_VIDEO   # Si el video por defecto existe, se carga de entrada
        if inicial.is_file():
            self.cargar_video(inicial)

    # ---------------- Aspecto ----------------

    def _estilos(self):
        estilo = ttk.Style(self)
        tema = {"win32": "vista", "darwin": "aqua"}.get(sys.platform, "clam")
        if tema in estilo.theme_names():
            estilo.theme_use(tema)
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=10)
        familia = base.actual("family")
        estilo.configure("Titulo.TLabel", font=(familia, 17, "bold"))
        estilo.configure("Sub.TLabel", foreground=GRIS)
        estilo.configure("Dato.TLabel", font=(familia, 10, "bold"))
        estilo.configure("Flecha.TLabel", font=(familia, 16), foreground=GRIS)
        estilo.configure("Accion.TButton", font=(familia, 11, "bold"), padding=(10, 8))
        estilo.configure("TLabelframe.Label", font=(familia, 10, "bold"))
        estilo.configure("Treeview", rowheight=26)
        # Barra de progreso azul sobre fondo claro (en Windows y macOS se usa la nativa)
        estilo.configure("Horizontal.TProgressbar", background=COLORES[0], troughcolor="#e4e3dd",
                         bordercolor="#e4e3dd", lightcolor=COLORES[0], darkcolor=COLORES[0])

    def _armar_ventana(self):
        cabecera = ttk.Frame(self, padding=(18, 14, 18, 6))
        cabecera.pack(fill="x")
        ttk.Label(cabecera, text="Pupilometría · Reflejo fotomotor", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(cabecera, text="Procesamiento de Señales e Imágenes Biomédicas · ITBA",
                  style="Sub.TLabel").pack(anchor="w")

        cuerpo = ttk.Frame(self, padding=(18, 8, 18, 18))
        cuerpo.pack(fill="both", expand=True)
        cuerpo.columnconfigure(1, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        izquierda = ttk.Frame(cuerpo, width=420)
        izquierda.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        self.pestanas = ttk.Notebook(cuerpo)
        self.pestanas.grid(row=0, column=1, sticky="nsew")

        self._armar_video(izquierda)
        self._armar_roi(izquierda)
        self._armar_mejoramiento(izquierda)
        self._armar_procesamiento(izquierda)
        self._armar_registro()
        self._armar_resultados()

    def _armar_video(self, padre):
        marco = ttk.LabelFrame(padre, text=" 1 · Video ", padding=12)
        marco.pack(fill="x")
        self.btn_video = ttk.Button(marco, text="Elegir video…", command=self.elegir_video)
        self.btn_video.pack(anchor="w")
        self.lbl_video = ttk.Label(marco, text="Ningún video seleccionado", style="Dato.TLabel", wraplength=380)
        self.lbl_video.pack(anchor="w", pady=(10, 2))
        self.lbl_meta = ttk.Label(marco, text="", style="Sub.TLabel", justify="left")
        self.lbl_meta.pack(anchor="w")

    def _armar_roi(self, padre):
        marco = ttk.LabelFrame(padre, text=" 2 · Región de interés ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        imagenes = ttk.Frame(marco)
        imagenes.pack(anchor="w")
        self.img_frame = ttk.Label(imagenes)
        self.img_frame.grid(row=0, column=0, sticky="n")
        self.img_roi = ttk.Label(imagenes, style="Sub.TLabel")
        self.img_roi.grid(row=0, column=1, sticky="n", padx=(10, 0))
        fila = ttk.Frame(marco)
        fila.pack(fill="x", pady=(8, 0))
        self.btn_roi = ttk.Button(fila, text="Marcar ROI…", command=self.marcar_roi, state="disabled")
        self.btn_roi.pack(side="left")
        self.lbl_roi = ttk.Label(fila, text="Primero elegí un video", style="Sub.TLabel", wraplength=250)
        self.lbl_roi.pack(side="left", padx=(10, 0))

    def _armar_mejoramiento(self, padre):
        marco = ttk.LabelFrame(padre, text=" 3 · Mejoramiento (Módulo 1) ", padding=12)
        marco.pack(fill="x", pady=10)
        # Mensaje mientras no hay imágenes (sin ROI, calculando o error)
        self.lbl_mejora = ttk.Label(marco, style="Sub.TLabel", wraplength=380, justify="left")
        # Los dos pasos uno al lado del otro: suavizado -> rango de interés
        self.pasos_mejora = ttk.Frame(marco)
        self.img_suave = ttk.Label(self.pasos_mejora)
        self.img_suave.grid(row=0, column=0)
        ttk.Label(self.pasos_mejora, text="→", style="Flecha.TLabel").grid(row=0, column=1, padx=8)
        self.img_rango = ttk.Label(self.pasos_mejora)
        self.img_rango.grid(row=0, column=2)
        self.lbl_suave = ttk.Label(self.pasos_mejora, style="Sub.TLabel", justify="center")
        self.lbl_suave.grid(row=1, column=0, pady=(4, 0))
        self.lbl_rango = ttk.Label(self.pasos_mejora, style="Sub.TLabel", justify="center")
        self.lbl_rango.grid(row=1, column=2, pady=(4, 0))
        self._mensaje_mejora("Cuando haya un ROI, acá se ven los pasos del mejoramiento.")

    def _armar_procesamiento(self, padre):
        marco = ttk.LabelFrame(padre, text=" 4 · Procesamiento ", padding=12)
        marco.pack(fill="x")
        self.btn_procesar = ttk.Button(marco, text="▶   Procesar video", style="Accion.TButton",
                                       command=self.procesar, state="disabled")
        self.btn_procesar.pack(fill="x")
        self.progreso = ttk.Progressbar(marco, maximum=100)
        self.progreso.pack(fill="x", pady=(10, 6))
        self.lbl_estado = ttk.Label(marco, text="Elegí un video para empezar", style="Sub.TLabel", wraplength=380)
        self.lbl_estado.pack(anchor="w")

    def _armar_registro(self):
        pestana = ttk.Frame(self.pestanas, padding=6)
        self.pestanas.add(pestana, text="  Registro  ")
        pestana.rowconfigure(0, weight=1)
        pestana.columnconfigure(0, weight=1)
        self.texto = tk.Text(pestana, wrap="none", font="TkFixedFont", state="disabled", relief="flat",
                             padx=10, pady=8, background="#fbfbfa")
        barra_y = ttk.Scrollbar(pestana, orient="vertical", command=self.texto.yview)
        barra_x = ttk.Scrollbar(pestana, orient="horizontal", command=self.texto.xview)
        self.texto.configure(yscrollcommand=barra_y.set, xscrollcommand=barra_x.set)
        self.texto.grid(row=0, column=0, sticky="nsew")
        barra_y.grid(row=0, column=1, sticky="ns")
        barra_x.grid(row=1, column=0, sticky="ew")
        self.texto.tag_configure("ok", foreground="#1a7f4b")
        self.texto.tag_configure("aviso", foreground="#a86b00")
        self.texto.tag_configure("error", foreground="#c0392b")
        self.texto.tag_configure("titulo", font=(tkfont.nametofont("TkFixedFont").actual("family"), 10, "bold"))

    def _armar_resultados(self):
        pestana = ttk.Frame(self.pestanas, padding=12)
        self.pestana_resultados = pestana
        self.pestanas.add(pestana, text="  Resultados  ", state="disabled")
        pestana.columnconfigure(0, weight=1)
        pestana.rowconfigure(1, weight=1)

        arriba = ttk.Frame(pestana)
        arriba.grid(row=0, column=0, sticky="ew")
        arriba.columnconfigure(0, weight=1)
        self.tabla = ttk.Treeview(arriba, columns=("parametro", "valor"), show="headings", height=10)
        self.tabla.heading("parametro", text="Parámetro", anchor="w")
        self.tabla.heading("valor", text="Valor", anchor="w")
        self.tabla.column("parametro", width=250, anchor="w")
        self.tabla.column("valor", width=330, anchor="w")
        self.tabla.grid(row=0, column=0, sticky="ew")

        botones = ttk.Frame(arriba)
        botones.grid(row=0, column=1, sticky="n", padx=(14, 0))
        ttk.Button(botones, text="Abrir carpeta de resultados",
                   command=self.abrir_resultados).pack(fill="x")
        ttk.Button(botones, text="Ver video de control",
                   command=self.abrir_video_control).pack(fill="x", pady=8)
        ttk.Label(botones, text="El video de control muestra el\ncontorno detectado en cada frame.",
                  style="Sub.TLabel", justify="left").pack(anchor="w")

        grafico = ttk.Frame(pestana)
        grafico.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.figura = Figure(figsize=(7, 4), dpi=100, layout="constrained")
        self.lienzo_figura = FigureCanvasTkAgg(self.figura, master=grafico)
        barra = NavigationToolbar2Tk(self.lienzo_figura, grafico, pack_toolbar=False)
        barra.update()
        barra.pack(side="bottom", fill="x")
        self.lienzo_figura.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- 1. Video ----------------

    def elegir_video(self):
        ruta = filedialog.askopenfilename(title="Elegir video", initialdir=RAIZ / "videos", filetypes=TIPOS_VIDEO)
        if ruta:
            self.cargar_video(Path(ruta))

    def cargar_video(self, ruta):
        try:
            video, metadatos = m0.abrir_video(ruta)
            ok, bgr = video.read()
            video.release()
            if not ok:
                raise ValueError("No se pudo leer el primer frame")
        except Exception as e:
            messagebox.showerror("No se pudo abrir el video", str(e))
            return

        self.ruta, self.metadatos = Path(ruta), metadatos
        self.frame0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.lbl_video.configure(text=self.ruta.name)
        self.lbl_meta.configure(text=f"{metadatos['ancho']} x {metadatos['alto']} px  ·  "
                                     f"{ui.numero(metadatos['fps'], 1, 'fps')}  ·  "
                                     f"{ui.numero(metadatos['duracion_s'], 1, 's')}  ·  "
                                     f"{metadatos['cantidad_frames']} frames")

        guardado = m2.roi_guardado(self.ruta.name)   # ROI de una corrida anterior, si existe
        self.roi = None if guardado is None else (guardado.x0, guardado.y0, guardado.w, guardado.h)
        self.btn_roi.configure(state="normal")
        self.pestanas.tab(self.pestana_resultados, state="disabled")
        self._mostrar_vista_previa()
        self._calcular_mejora()
        self._actualizar_estado()

    # ---------------- 2. ROI ----------------

    def marcar_roi(self):
        selector = SelectorROI(self, self.frame0, self.roi)
        self.wait_window(selector)
        if selector.resultado:
            self.roi = selector.resultado
            m2.guardar_roi(self.ruta.name, m2.ROI(*self.roi))   # El pipeline lo toma de roi.json
            self._mostrar_vista_previa()
            self._calcular_mejora()
            self._actualizar_estado()

    def _mostrar_vista_previa(self):
        vista = self.frame0.copy()
        if self.roi:
            x, y, w, h = self.roi
            grosor = max(2, vista.shape[1] // 200)
            cv2.rectangle(vista, (x, y), (x + w, y + h), (235, 104, 52), grosor)
        self._foto_frame, _ = a_foto(vista, 200, 110)
        self.img_frame.configure(image=self._foto_frame)

        recorte = None
        if self.roi:
            x, y, w, h = self.roi
            recorte = self.frame0[max(0, y):y + h, max(0, x):x + w]
        if recorte is not None and recorte.size:
            self._foto_roi, _ = a_foto(recorte, 180, 110, ampliar_hasta=3.0)
            self.img_roi.configure(image=self._foto_roi, text="")
            self.lbl_roi.configure(text=f"x = {self.roi[0]}, y = {self.roi[1]}, "
                                        f"{self.roi[2]} x {self.roi[3]} px (guardado en roi.json)")
            self.btn_roi.configure(text="Cambiar ROI…")
        else:
            self._foto_roi = None
            self.img_roi.configure(image="", text="Sin ROI")
            self.lbl_roi.configure(text="Marcá el iris con un rectángulo para poder procesar")
            self.btn_roi.configure(text="Marcar ROI…")

    # ---------------- 3. Mejoramiento ----------------

    def _calcular_mejora(self):
        """Mejoramiento del primer frame (Módulo 1) en segundo plano: lo mismo que hace el paso 2 del pipeline."""
        self._version_mejora += 1
        if self.roi is None:
            self.mejorando = False
            self._mensaje_mejora("Cuando haya un ROI, acá se ven los pasos del mejoramiento.")
            return
        version, ruta, roi = self._version_mejora, self.ruta, self.roi
        arrancar = not self.mejorando
        self.mejorando = True
        self._mensaje_mejora("Calculando el mejoramiento del ROI…")

        def trabajo():
            try:
                preparacion = m2.preparar(ruta, rect=roi)          # ROI, escala y resolución
                _, mejora, _ = m2.frame_mejorado(ruta, preparacion)
                self.cola_mejora.put((version, mejora, None))
            except Exception as e:
                self.cola_mejora.put((version, None, str(e) or type(e).__name__))

        threading.Thread(target=trabajo, daemon=True).start()
        if arrancar:
            self.after(100, self._revisar_mejora)

    def _revisar_mejora(self):
        while True:
            try:
                version, mejora, error = self.cola_mejora.get_nowait()
            except queue.Empty:
                break
            if version == self._version_mejora:   # Los de un ROI anterior se descartan
                self._mostrar_mejora(mejora, error)
        if self.mejorando:
            self.after(100, self._revisar_mejora)

    def _mensaje_mejora(self, texto):
        """Muestra un mensaje en lugar de las imágenes del mejoramiento."""
        self._fotos_mejora = None
        self.pasos_mejora.pack_forget()
        self.lbl_mejora.configure(text=texto)
        self.lbl_mejora.pack(anchor="w")

    def _mostrar_mejora(self, mejora, error):
        self.mejorando = False
        if error:
            self._mensaje_mejora(f"No se pudo calcular el mejoramiento: {error}")
        else:
            suave, _ = a_foto(mejora["suave"], 170, 100, ampliar_hasta=3.0)
            rango, _ = a_foto(m1.realzada_para_ver(mejora), 170, 100, ampliar_hasta=3.0)
            self._fotos_mejora = (suave, rango)   # Tkinter necesita guardar una referencia a cada imagen
            self.img_suave.configure(image=suave)
            self.img_rango.configure(image=rango)
            k = mejora["ventana"]
            f2 = "sin contraste" if mejora["umbral"] is None else mejora["umbral"]
            self.lbl_suave.configure(text=f"1. Suavizado gaussiano\n{k} x {k} px (en grises)")
            self.lbl_rango.configure(text=f"2. Rango de interés [0, {f2}]\nentra a la segmentación")
            self.lbl_mejora.pack_forget()
            self.pasos_mejora.pack(anchor="w")
        self._actualizar_estado()

    # ---------------- 4. Procesamiento ----------------

    def _actualizar_estado(self):
        listo = self.ruta is not None and self.roi is not None and not self.mejorando
        self.btn_procesar.configure(state="normal" if listo else "disabled")
        if self.ruta is None:
            self.lbl_estado.configure(text="Elegí un video para empezar")
        elif self.roi is None:
            self.lbl_estado.configure(text="Falta marcar la región de interés")
        elif self.mejorando:
            self.lbl_estado.configure(text="Calculando el mejoramiento…")
        else:
            self.lbl_estado.configure(text="Todo listo para procesar")

    def _bloquear(self, bloqueado):
        estado = "disabled" if bloqueado else "normal"
        for boton in (self.btn_video, self.btn_roi, self.btn_procesar):
            boton.configure(state=estado)

    def procesar(self):
        self._bloquear(True)
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.configure(state="disabled")
        self.progreso.configure(value=0)
        self.pestanas.select(0)
        self.pestanas.tab(self.pestana_resultados, state="disabled")
        self.lbl_estado.configure(text="Iniciando…")
        self.paso_actual = 0
        self._resto_de_linea = ""
        self._inicio = time.perf_counter()

        self.hilo = threading.Thread(target=self._trabajo, daemon=True)
        self.hilo.start()
        self.after(100, self._revisar_cola)

    def _trabajo(self):
        """Corre en segundo plano: el pipeline completo, con su salida redirigida a la ventana."""
        salida, errores = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = SalidaACola(self.cola)
        try:
            pipeline.MOSTRAR_PASOS = False      # Los gráficos se muestran en la ventana
            pipeline.USAR_ROI_GUARDADO = True   # El ROI marcado acá ya está en roi.json
            curva, parametros = pipeline.ejecutar_pipeline(str(self.ruta))
            self.cola.put(("fin", (curva, parametros)))
        except Exception as e:
            traceback.print_exc()
            self.cola.put(("error", str(e) or type(e).__name__))
        finally:
            sys.stdout, sys.stderr = salida, errores

    def _revisar_cola(self):
        """Lee lo que mandó el hilo de procesamiento (texto, fin o error) y actualiza la ventana."""
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "texto":
                    self._agregar_texto(dato)
                elif tipo == "fin":
                    self._terminar(*dato)
                elif tipo == "error":
                    self._fallar(dato)
        except queue.Empty:
            pass
        if (self.hilo is not None and self.hilo.is_alive()) or not self.cola.empty():
            self.after(100, self._revisar_cola)

    def _agregar_texto(self, texto):
        self.texto.configure(state="normal")
        lineas = (self._resto_de_linea + texto).split("\n")
        self._resto_de_linea = lineas.pop()    # La última puede estar incompleta
        for linea in lineas:
            self.texto.insert("end", linea + "\n", self._etiqueta(linea))
            self._interpretar(linea)
        self.texto.see("end")
        self.texto.configure(state="disabled")

    @staticmethod
    def _etiqueta(linea):
        limpia = linea.strip()
        if limpia.startswith("✔"):
            return "ok"
        if limpia.startswith("⚠") or "[aviso]" in limpia or limpia.startswith("Atención"):
            return "aviso"
        if limpia.startswith("✖") or limpia.startswith("Traceback"):
            return "error"
        if re.match(r"\[\d+/\d+\]", limpia) or limpia == "RESULTADOS":
            return "titulo"
        return ()

    def _interpretar(self, linea):
        """Avance y estado a partir de los mensajes del pipeline."""
        paso = re.match(r"\s*\[(\d+)/(\d+)\]\s*(.+)", linea)
        if paso:
            self.paso_actual, self.total_pasos = int(paso.group(1)), int(paso.group(2))
            self.progreso.configure(value={1: 2, 2: 4, 3: 8, 4: 92, 5: 95, 6: 98}.get(self.paso_actual, 0))
            self.lbl_estado.configure(text=f"Paso {paso.group(1)} de {paso.group(2)}: {paso.group(3)}")
            return
        frame = re.search(r"frame (\d+)/(\d+) \(\d+ %\)\s+faltan ~(\d+) s", linea)
        if frame and self.paso_actual == 3:
            hechos, total, faltan = (int(g) for g in frame.groups())
            self.progreso.configure(value=8 + 78 * hechos / max(total, 1))
            self.lbl_estado.configure(text=f"Paso 3 de {self.total_pasos}: mejorando y segmentando "
                                           f"(frame {hechos} de {total}, faltan ~{faltan} s)")
        elif "=== Frame" in linea:
            self.progreso.configure(value=88)
            self.lbl_estado.configure(text=f"Paso 3 de {self.total_pasos}: frame de trabajo (2.1 y 2.2)")
        elif "Se abrió una ventana para marcar el ROI" in linea:
            self.lbl_estado.configure(text="Marcá el ROI en la ventana que se abrió")

    def _terminar(self, curva, parametros):
        self.progreso.configure(value=100)
        self.lbl_estado.configure(text=f"Listo: {self.ruta.name} procesado en "
                                       f"{ui.duracion(time.perf_counter() - self._inicio)}")
        self._bloquear(False)
        self._mostrar_resultados(curva, parametros)
        self.pestanas.tab(self.pestana_resultados, state="normal")
        self.pestanas.select(self.pestana_resultados)
        self.bell()

    def _fallar(self, mensaje):
        self.lbl_estado.configure(text="Error: no se pudo procesar el video")
        self._bloquear(False)
        messagebox.showerror("No se pudo procesar el video",
                             f"{mensaje}\n\nEl detalle está en la pestaña Registro.")

    # ---------------- Resultados ----------------

    def _mostrar_resultados(self, curva, parametros):
        self.tabla.delete(*self.tabla.get_children())
        legibles = ui.parametros_legibles(parametros) or {"Estímulo": "no se pudo detectar"}
        for nombre, valor in legibles.items():
            self.tabla.insert("", "end", values=(nombre, valor))
        self._dibujar_curva(curva, parametros)

    def _dibujar_curva(self, curva, parametros):
        self.figura.clear()
        eje_d, eje_b = self.figura.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        t = curva["t_s"]

        eje_d.plot(t, curva["diametro"], color="0.75", linewidth=1, label="Medido")
        if "artefacto" in curva:
            artefacto = curva["artefacto"].astype(bool)
            eje_d.plot(t[artefacto], curva["diametro"][artefacto], "x", color="#c0392b", markersize=4,
                       label="Artefacto")
        if "diametro_suave" in curva:
            eje_d.plot(t, curva["diametro_suave"], color=COLORES[0], linewidth=2, label="Limpio y suavizado")
        eje_d.set_ylabel("Diámetro (px)", color=GRIS)
        eje_b.plot(t, curva["brillo"], color=COLORES[1], linewidth=1.5)
        eje_b.set_ylabel("Brillo", color=GRIS)
        eje_b.set_xlabel("Tiempo (s)", color=GRIS)

        marcas = [("t_estimulo_s", "Estímulo", GRIS, "--"), ("t_inicio_s", "Inicio de la respuesta", COLORES[2], ":"),
                  ("t_extremo_s", "Extremo", COLORES[1], ":")]
        for clave, nombre, color, estilo in marcas:
            valor = parametros.get(clave, np.nan)
            if np.isfinite(valor):
                for eje in (eje_d, eje_b):
                    eje.axvline(valor, color=color, linestyle=estilo, linewidth=1.3,
                                label=nombre if eje is eje_d else None)
        eje_d.legend(loc="best", fontsize=8, frameon=False)
        eje_d.set_title(f"Curva temporal · {self.ruta.name}", fontweight="bold", fontsize=11)
        for eje in (eje_d, eje_b):
            for lado in ("top", "right"):
                eje.spines[lado].set_visible(False)
            eje.grid(color="#e4e3dd", linewidth=0.8)
            eje.tick_params(colors=GRIS, labelsize=9)
        self.lienzo_figura.draw()

    def abrir_resultados(self):
        carpeta = RAIZ / pipeline.CARPETA_RESULTADOS / "por_video" / self.ruta.stem
        abrir_en_sistema(carpeta if carpeta.exists() else RAIZ / pipeline.CARPETA_RESULTADOS)

    def abrir_video_control(self):
        video = m2.OUT_VIDEO / f"{self.ruta.stem}_overlay.mp4"
        if video.exists():
            abrir_en_sistema(video)
        else:
            messagebox.showinfo("Video de control", "No se generó el video de control para este registro.")

    # ---------------- Cierre ----------------

    def _cerrar(self):
        if self.hilo is not None and self.hilo.is_alive():
            if not messagebox.askyesno("Procesamiento en curso",
                                       "El video se está procesando. ¿Cerrar igual?"):
                return
        self.destroy()


def main():
    if sys.platform == "win32":
        try:   # Texto nítido en pantallas con escala (125 %, 150 %...)
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    Aplicacion().mainloop()


if __name__ == "__main__":
    main()