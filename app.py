"""
Restaurador de fotos — interfaz gráfica.

Flujo completo sin depender de nada externo:
  abrir imagen -> detectar el texto (automático y/o pincel) -> restaurar -> comparar -> guardar

Uso:
    python3 -m restaurador.app
"""

from __future__ import annotations

import os

# Windows: OpenCV y torch traen cada uno su propio runtime de OpenMP; cargarlos
# juntos puede abortar al inicializar la 2da DLL. Esto los deja convivir. Tiene
# que ir ANTES de importar cv2/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import queue
import threading
import traceback

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageOps, ImageTk

from . import motor, deteccion, backends


NOMBRES = {
    "original": "Original",
    "algoritmo": "Algoritmo",
    "ia": "IA",
    "combinado": "Combinado",
}


def leer_imagen(ruta):
    """Carga respetando la orientación EXIF (si no, las fotos de celular salen giradas)."""
    pil = Image.open(ruta)
    pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def guardar_imagen(ruta, img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".jpg", ".jpeg"):
        pil.save(ruta, quality=97, subsampling=0)
    else:
        pil.save(ruta)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Restaurador de fotos")
        self.geometry("1180x780")
        self.minsize(900, 600)

        # --- estado ---
        self.img = None              # BGR full res
        self.mask = None             # uint8 full res (255 = reconstruir)
        self.ruta = None
        self.resultados = {}
        self.undo_stack = []
        self.zoom = 1.0
        self.fit = True
        self._tkimg = None
        self._pintando = False
        self._trabajando = False
        self.cola = queue.Queue()

        self.vista = tk.StringVar(value="original")
        self.modo_pincel = tk.StringVar(value="agregar")
        self.tam_pincel = tk.IntVar(value=24)
        self.preset = tk.StringVar(value="equilibrado")
        self.backend_ia = tk.StringVar(value=motor.BACKEND_IA)
        self.modo_deteccion = tk.StringVar(value="trazos")
        self.sensib_txt = tk.StringVar(value="media")   # baja | media | alta
        self.mostrar_mascara = tk.BooleanVar(value=True)
        self.peso_ia = tk.DoubleVar(value=motor.PESO_IA * 100)

        self._construir_ui()
        self.after(80, self._procesar_cola)

    # ------------------------------------------------------------------ UI
    def _construir_ui(self):
        barra = ttk.Frame(self, padding=(8, 6))
        barra.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(barra, text="Abrir imagen…", command=self.abrir).pack(side=tk.LEFT)
        ttk.Separator(barra, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(barra, text="Detectar texto", command=self.detectar).pack(side=tk.LEFT)
        ttk.OptionMenu(barra, self.modo_deteccion, "trazos", "trazos", "caja").pack(side=tk.LEFT, padx=(4, 0))
        ttk.OptionMenu(barra, self.sensib_txt, "media", "baja", "media", "alta").pack(side=tk.LEFT, padx=(2, 0))

        ttk.Separator(barra, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(barra, text="Pincel:").pack(side=tk.LEFT)
        ttk.Radiobutton(barra, text="Agregar", value="agregar",
                        variable=self.modo_pincel).pack(side=tk.LEFT)
        ttk.Radiobutton(barra, text="Borrar", value="borrar",
                        variable=self.modo_pincel).pack(side=tk.LEFT)
        ttk.Scale(barra, from_=4, to=200, variable=self.tam_pincel,
                  orient=tk.HORIZONTAL, length=110).pack(side=tk.LEFT, padx=4)
        self.lbl_pincel = ttk.Label(barra, text="24 px", width=6)
        self.lbl_pincel.pack(side=tk.LEFT)

        ttk.Button(barra, text="Deshacer", command=self.deshacer).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(barra, text="Limpiar", command=self.limpiar_mascara).pack(side=tk.LEFT, padx=2)

        ttk.Separator(barra, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(barra, text="Calidad:").pack(side=tk.LEFT)
        ttk.OptionMenu(barra, self.preset, "equilibrado",
                       "rapido", "equilibrado", "maximo").pack(side=tk.LEFT)

        ttk.Label(barra, text="IA:").pack(side=tk.LEFT, padx=(6, 0))
        ttk.OptionMenu(barra, self.backend_ia, motor.BACKEND_IA,
                       *backends.disponibles()).pack(side=tk.LEFT)

        self.btn_restaurar = ttk.Button(barra, text="Restaurar", command=self.restaurar)
        self.btn_restaurar.pack(side=tk.LEFT, padx=8)

        # --- panel derecho ---
        lateral = ttk.Frame(self, padding=8)
        lateral.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(lateral, text="Ver", font=("", 10, "bold")).pack(anchor=tk.W)
        self.radios = {}
        for k in ("original", "algoritmo", "ia", "combinado"):
            r = ttk.Radiobutton(lateral, text=NOMBRES[k], value=k,
                                variable=self.vista, command=self.redibujar)
            r.pack(anchor=tk.W)
            if k != "original":
                r.state(["disabled"])
            self.radios[k] = r

        # Mezcla en vivo: IA y algoritmo ya están calculados, así que combinarlos
        # es instantáneo. Se ajusta a ojo, que es lo que importa.
        self.marco_mezcla = ttk.Frame(lateral)
        self.marco_mezcla.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
        self.lbl_mezcla = ttk.Label(self.marco_mezcla, text="   mezcla: 90% IA")
        self.lbl_mezcla.pack(anchor=tk.W)
        self.sld_mezcla = ttk.Scale(self.marco_mezcla, from_=0, to=100,
                                    variable=self.peso_ia, orient=tk.HORIZONTAL,
                                    command=lambda _v: self._cambiar_mezcla())
        self.sld_mezcla.pack(fill=tk.X)
        self.sld_mezcla.state(["disabled"])

        ttk.Checkbutton(lateral, text="Ver máscara", variable=self.mostrar_mascara,
                        command=self.redibujar).pack(anchor=tk.W, pady=(8, 0))

        ttk.Separator(lateral, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(lateral, text="Zoom", font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(lateral, text="Ajustar", command=self.ajustar_zoom).pack(fill=tk.X, pady=1)
        ttk.Button(lateral, text="100 %", command=lambda: self.set_zoom(1.0)).pack(fill=tk.X, pady=1)
        ttk.Button(lateral, text="+", command=lambda: self.set_zoom(self.zoom * 1.25)).pack(fill=tk.X, pady=1)
        ttk.Button(lateral, text="−", command=lambda: self.set_zoom(self.zoom / 1.25)).pack(fill=tk.X, pady=1)

        ttk.Separator(lateral, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self.btn_guardar = ttk.Button(lateral, text="Guardar las 3…", command=self.guardar_todo)
        self.btn_guardar.pack(fill=tk.X)
        self.btn_guardar.state(["disabled"])
        self.btn_guardar_una = ttk.Button(lateral, text="Guardar la vista…", command=self.guardar_vista)
        self.btn_guardar_una.pack(fill=tk.X, pady=2)
        self.btn_guardar_una.state(["disabled"])
        # Guardar solo la máscara: sirve apenas hay algo marcado, sin restaurar.
        self.btn_guardar_mascara = ttk.Button(lateral, text="Guardar máscara…",
                                               command=self.guardar_mascara)
        self.btn_guardar_mascara.pack(fill=tk.X, pady=2)

        # --- lienzo ---
        cont = ttk.Frame(self)
        cont.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cont, bg="#2b2b2b", highlightthickness=0)
        hs = ttk.Scrollbar(cont, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vs = ttk.Scrollbar(cont, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<Configure>", lambda e: self.redibujar())

        # --- estado inferior ---
        pie = ttk.Frame(self, padding=(8, 4))
        pie.pack(side=tk.BOTTOM, fill=tk.X)
        self.barra_prog = ttk.Progressbar(pie, mode="determinate", maximum=1.0, length=220)
        self.barra_prog.pack(side=tk.LEFT)
        self.lbl_estado = ttk.Label(pie, text="Abrí una imagen para empezar.")
        self.lbl_estado.pack(side=tk.LEFT, padx=10)

        self.tam_pincel.trace_add("write", lambda *a: self.lbl_pincel.config(
            text=f"{self.tam_pincel.get()} px"))

    # ------------------------------------------------------------- acciones
    def estado(self, txt):
        self.lbl_estado.config(text=txt)

    def precargar_ia(self):
        """
        Carga torch en el HILO PRINCIPAL al arrancar, una sola vez.

        En Windows, importar torch por primera vez desde un hilo secundario
        (el worker de «Restaurar»/«Detectar») puede fallar al inicializar sus
        DLLs con «WinError 1114». Cargarlo acá, antes de que corra cualquier
        worker, evita ese choque: cuando el worker haga `import torch` ya está
        cargado (es un no-op).
        """
        self.estado("Cargando el motor de IA (solo la primera vez)…")
        self.update_idletasks()
        try:
            import torch  # noqa: F401
            self.estado("Motor de IA listo. Abrí una imagen para empezar.")
        except Exception:
            self.estado("No pude cargar el motor de IA (torch). Mirá la consola.")
            messagebox.showerror(
                "Error al cargar el motor de IA",
                "No se pudo inicializar torch en este equipo.\n\n"
                + traceback.format_exc()[-1200:])

    def abrir(self):
        ruta = filedialog.askopenfilename(
            title="Elegí la imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                       ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self.img = leer_imagen(ruta)
        except Exception as e:
            messagebox.showerror("No pude abrir la imagen", str(e))
            return
        self.ruta = ruta
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.resultados = {}
        self.undo_stack = []
        self.vista.set("original")
        for k in ("algoritmo", "ia", "combinado"):
            self.radios[k].state(["disabled"])
        self.btn_guardar.state(["disabled"])
        self.btn_guardar_una.state(["disabled"])
        h, w = self.img.shape[:2]
        self.estado(f"{os.path.basename(ruta)} — {w}x{h} px. Marcá el texto y tocá Restaurar.")
        self.fit = True
        self.ajustar_zoom()

    def _guardar_undo(self):
        if self.mask is None:
            return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def deshacer(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.redibujar()

    def limpiar_mascara(self):
        if self.mask is None:
            return
        self._guardar_undo()
        self.mask[:] = 0
        self.redibujar()

    def detectar(self):
        if self.img is None or self._trabajando:
            return
        self._trabajando = True
        self.btn_restaurar.state(["disabled"])
        self.estado("Detectando texto… (la primera vez descarga el modelo)")
        self.barra_prog.config(mode="indeterminate")
        self.barra_prog.start(12)

        modo = self.modo_deteccion.get()
        sensibilidad = {"baja": 0.25, "media": 0.5, "alta": 0.8}.get(
            self.sensib_txt.get(), 0.5)
        img = self.img.copy()

        def worker():
            try:
                m, cajas = deteccion.detectar_texto(img, modo=modo,
                                                    sensibilidad=sensibilidad)
                self.cola.put(("deteccion_ok", (m, len(cajas))))
            except Exception:
                self.cola.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def restaurar(self):
        if self.img is None or self._trabajando:
            return
        if np.count_nonzero(self.mask) == 0:
            messagebox.showinfo(
                "Falta marcar el texto",
                "No hay nada marcado.\n\nUsá «Detectar texto» o pintá encima del texto con el pincel.")
            return

        self._trabajando = True
        self.btn_restaurar.state(["disabled"])
        self.barra_prog.config(mode="determinate")
        self.barra_prog["value"] = 0
        self.estado("Restaurando…")

        img = self.img.copy()
        mask = self.mask.copy()
        preset = self.preset.get()
        backend = self.backend_ia.get()

        def worker():
            try:
                def prog(f, m):
                    self.cola.put(("progreso", (f, m)))
                res = motor.restaurar(img, mask, preset=preset, backend=backend,
                                      progress=prog)
                self.cola.put(("restauracion_ok", res))
            except Exception:
                self.cola.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _procesar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()

                if tipo == "progreso":
                    f, m = dato
                    self.barra_prog["value"] = f
                    self.estado(m)

                elif tipo == "deteccion_ok":
                    m, n = dato
                    self.barra_prog.stop()
                    self.barra_prog.config(mode="determinate")
                    self.barra_prog["value"] = 0
                    self._trabajando = False
                    self.btn_restaurar.state(["!disabled"])
                    motor_usado = deteccion.ultimo_motor or "?"
                    if deteccion.ultimo_error:
                        messagebox.showwarning(
                            "EasyOCR falló",
                            "La detección con IA falló y usé el detector básico.\n\n"
                            + str(deteccion.ultimo_error)[-900:])
                    if n == 0:
                        self.estado(f"No encontré texto (detector: {motor_usado}). "
                                    "Marcalo a mano con el pincel.")
                    else:
                        self._guardar_undo()
                        self.mask = cv2.bitwise_or(self.mask, m)
                        self.estado(f"Detecté {n} zona(s) con «{motor_usado}». "
                                    "Revisá y corregí con el pincel si hace falta.")
                    self.redibujar()

                elif tipo == "restauracion_ok":
                    self.resultados = dato
                    self._trabajando = False
                    self.btn_restaurar.state(["!disabled"])
                    for k in ("algoritmo", "ia", "combinado"):
                        self.radios[k].state(["!disabled"])
                    self.btn_guardar.state(["!disabled"])
                    self.btn_guardar_una.state(["!disabled"])
                    self.sld_mezcla.state(["!disabled"])
                    self.mostrar_mascara.set(False)
                    self.vista.set("ia")
                    self.estado("Listo. Compará IA / Combinado y movés la mezcla a gusto.")
                    self.redibujar()

                elif tipo == "error":
                    self.barra_prog.stop()
                    self.barra_prog.config(mode="determinate")
                    self.barra_prog["value"] = 0
                    self._trabajando = False
                    self.btn_restaurar.state(["!disabled"])
                    self.estado("Hubo un error.")
                    messagebox.showerror("Error", str(dato)[-1500:])

        except queue.Empty:
            pass
        self.after(80, self._procesar_cola)

    def _cambiar_mezcla(self):
        """Recalcula «combinado» al mover el deslizador. Es solo un addWeighted: instantáneo."""
        if not self.resultados or "ia" not in self.resultados or "algoritmo" not in self.resultados:
            return
        p = self.peso_ia.get() / 100.0
        self.lbl_mezcla.config(text=f"   mezcla: {int(round(p*100))}% IA")
        self.resultados["combinado"] = motor.mezclar(
            self.resultados["ia"], self.resultados["algoritmo"], p)
        if self.vista.get() == "combinado":
            self.redibujar()

    # ------------------------------------------------------------- guardado
    def _imagen_vista(self):
        v = self.vista.get()
        if v == "original" or v not in self.resultados:
            return self.img
        return self.resultados[v]

    def guardar_mascara(self):
        """Guarda SOLO la máscara (blanco = zona a reconstruir), sin restaurar.
        Sirve para reusarla con la CLI, en otra herramienta, o guardarla aparte."""
        if self.mask is None or np.count_nonzero(self.mask) == 0:
            messagebox.showinfo(
                "Máscara vacía",
                "No hay nada marcado todavía.\n\n"
                "Usá «Detectar texto» o pintá el texto con el pincel, y después guardá.")
            return
        base = os.path.splitext(os.path.basename(self.ruta or "imagen"))[0]
        ruta = filedialog.asksaveasfilename(
            title="Guardar la máscara",
            defaultextension=".png",
            initialfile=f"{base}_mascara.png",
            filetypes=[("PNG", "*.png")])
        if not ruta:
            return
        cv2.imwrite(ruta, self.mask)
        self.estado(f"Máscara guardada: {ruta}")

    def guardar_vista(self):
        if not self.resultados:
            return
        v = self.vista.get()
        base = os.path.splitext(os.path.basename(self.ruta or "imagen"))[0]
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{base}_{v}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not ruta:
            return
        guardar_imagen(ruta, self._imagen_vista())
        self.estado(f"Guardado: {ruta}")

    def guardar_todo(self):
        if not self.resultados:
            return
        carpeta = filedialog.askdirectory(title="¿Dónde guardo los resultados?")
        if not carpeta:
            return
        base = os.path.splitext(os.path.basename(self.ruta or "imagen"))[0]
        for k, v in self.resultados.items():
            guardar_imagen(os.path.join(carpeta, f"{base}_{k}.png"), v)
        cv2.imwrite(os.path.join(carpeta, f"{base}_mascara.png"), self.mask)
        self.estado(f"Guardé {len(self.resultados)} versiones + la máscara en {carpeta}")

    # ---------------------------------------------------------------- zoom
    def set_zoom(self, z):
        self.fit = False
        self.zoom = float(np.clip(z, 0.05, 8.0))
        self.redibujar()

    def ajustar_zoom(self):
        if self.img is None:
            return
        self.fit = True
        self.redibujar()

    def _zoom_efectivo(self):
        if self.img is None:
            return 1.0
        if not self.fit:
            return self.zoom
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        h, w = self.img.shape[:2]
        self.zoom = min(cw / w, ch / h, 1.0)
        return self.zoom

    # --------------------------------------------------------------- dibujo
    def redibujar(self):
        self.canvas.delete("all")
        if self.img is None:
            return

        z = self._zoom_efectivo()
        h, w = self.img.shape[:2]
        dw, dh = max(1, int(w * z)), max(1, int(h * z))

        vis = self._imagen_vista()
        disp = cv2.resize(vis, (dw, dh), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_NEAREST)

        if self.mostrar_mascara.get() and self.mask is not None and np.count_nonzero(self.mask):
            m = cv2.resize(self.mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
            capa = disp.copy()
            capa[m > 0] = (40, 40, 220)          # rojo en BGR
            disp = cv2.addWeighted(disp, 0.5, capa, 0.5, 0)

        pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        self._tkimg = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tkimg)
        self.canvas.config(scrollregion=(0, 0, dw, dh))

    # --------------------------------------------------------------- pincel
    def _a_coords_img(self, ev):
        x = self.canvas.canvasx(ev.x)
        y = self.canvas.canvasy(ev.y)
        z = self.zoom if self.zoom > 0 else 1.0
        return int(x / z), int(y / z)

    def _mouse_down(self, ev):
        if self.img is None or self._trabajando:
            return
        self._guardar_undo()
        self._pintando = True
        self.mostrar_mascara.set(True)
        self._pintar(ev)

    def _mouse_move(self, ev):
        if self._pintando:
            self._pintar(ev)

    def _mouse_up(self, _ev):
        self._pintando = False

    def _pintar(self, ev):
        x, y = self._a_coords_img(ev)
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        r = max(1, self.tam_pincel.get() // 2)
        color = 255 if self.modo_pincel.get() == "agregar" else 0
        cv2.circle(self.mask, (x, y), r, color, -1)
        self.redibujar()


def main():
    app = App()
    app.update()          # muestra la ventana antes de la carga pesada
    app.precargar_ia()    # torch en el hilo principal, antes de cualquier worker
    app.mainloop()


if __name__ == "__main__":
    main()
