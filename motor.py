"""
Motor de restauración de imágenes.

Genera tres reconstrucciones de la zona tapada:
  - "algoritmo"  : FSR (Frequency Selective Reconstruction) de OpenCV
  - "ia"         : modelo LaMa, UNA pasada limpia
  - "combinado"  : mezcla ajustable IA/algoritmo (por defecto 90% IA)

IMPORTANTE — corrección de criterio:
La primera versión de este motor se ajustó con PSNR/SSIM. Fue un error: esas
métricas premian el desenfoque, porque acertarle al PROMEDIO de todas las
texturas plausibles penaliza menos que arriesgar una textura nítida y errarle.
Siguiendo PSNR se llegaba a resultados con "manchón" liso.

Ahora se mide con LPIPS (métrica perceptual) y con la energía de textura del
relleno comparada con la de su entorno. Los dos criterios se contradicen:

    config                LPIPS(ojo)   textura   PSNR
    LaMa 1 pasada           0.081       0.88     22.04
    LaMa TTA8 (promedio)    0.132       0.53     23.13  <- gana PSNR, es una mancha

Por eso el promediado TTA se eliminó. Las decisiones de este módulo salen de
mediciones perceptuales:

  * FSR_FAST vs FSR_BEST -> FSR_BEST cuesta ~30x más tiempo para ganar 0.07 dB.
    Se usa FSR_FAST por defecto.
  * FSR y SHIFTMAP usan la máscara INVERTIDA respecto de cv2.inpaint
    (255 = píxel conocido). Es un detalle no obvio que produce basura si se ignora.
  * TTA (promediar rotaciones) ELIMINADO. Subía PSNR y destruía la textura:
    promediar 4 u 8 reconstrucciones distintas y plausibles no da una textura,
    da una mancha suave. LPIPS empeoraba 39% (0.081 -> 0.132 en textura de pelo).
  * Mezclar IA con el algoritmo también promedia, o sea también emborrona.
    Se midieron 0.9, 0.75, split de frecuencias y 2 pasadas de LaMa: NINGUNA
    le gana a LaMa 1 pasada. Por eso "combinado" es solo una mezcla ajustable
    por el usuario, con 90% IA por defecto (empate técnico con IA pura).
  * Reinyectar grano sintético o realce (unsharp) sube la métrica de textura
    pero EMPEORA LPIPS: ruido falso no es textura real. Descartado.
  * Reescalar la imagen antes de LaMa DEGRADA mucho el resultado (-1.7 dB a 768px).
    Se trabaja siempre a resolución nativa.
  * Contexto: medido con PSNR parecía que 15% bastaba. Con LPIPS es al revés,
    más contexto da mejor resultado (pad 1.0 -> 0.2411 vs pad 0.15 -> 0.2557).
    Por eso el contexto por defecto ahora es 1.0 (equivalente a la imagen casi
    entera en la mayoría de los casos).
  * LaMa rellena la imagen a múltiplos de 8 y devuelve ese tamaño: hay que
    recortar la salida o la imagen queda desalineada.
"""

from __future__ import annotations

import numpy as np
import cv2

from . import backends

# --- Parámetros derivados de las mediciones -------------------------------

PAD_RATIO = 1.0       # contexto alrededor de la máscara (medido con LPIPS)
PAD_MIN_PX = 64       # contexto mínimo absoluto, para máscaras chicas
PESO_IA = 0.90        # mezcla por defecto de "combinado"
FEATHER_PX = 1.5      # difuminado del borde al recomponer (solo hacia afuera)
BACKEND_IA = "lama"   # backend de IA por defecto (ver backends.py). LaMa es el
                      # medido y rápido; "sd_inpaint" y "lama_sd_refine" dan más
                      # textura pero necesitan diffusers y son lentos en CPU.

# Los presets ya no controlan el promediado (se eliminó), sino cuánto contexto
# recibe el modelo: más contexto = mejor resultado y más lento.
PRESETS_CONTEXTO = {
    "rapido": 0.2,        # para tantear la máscara
    "equilibrado": 1.0,
    "maximo": None,       # None = imagen completa
}
PRESETS_TTA = PRESETS_CONTEXTO   # alias viejo, por compatibilidad


def tiene_fsr() -> bool:
    """Indica si esta instalación de OpenCV trae el módulo xphoto con FSR."""
    return hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "INPAINT_FSR_FAST")


# --- Utilidades de máscara y recorte --------------------------------------

def normalizar_mascara(mask, shape) -> np.ndarray:
    """Devuelve la máscara como uint8 binaria (0/255) del tamaño de la imagen."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _caja_recorte(mask, shape, pad_ratio=PAD_RATIO):
    """
    Caja de trabajo alrededor de la máscara, a resolución nativa.

    Nunca se reescala: reducir la imagen antes de LaMa degrada mucho el
    resultado (-1.7 dB a 768 px).

    pad_ratio=None -> imagen completa (máximo contexto, más lento).
    """
    H, W = shape[:2]
    if pad_ratio is None:
        return 0, H, 0, W

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("La máscara está vacía: no hay nada marcado para reconstruir.")

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = max(int(max(y1 - y0, x1 - x0) * pad_ratio), PAD_MIN_PX)

    Y0, Y1 = max(0, y0 - pad), min(H, y1 + pad + 1)
    X0, X1 = max(0, x0 - pad), min(W, x1 + pad + 1)
    return Y0, Y1, X0, X1


def _componer(base, relleno, mask, feather=FEATHER_PX):
    """
    Pega `relleno` sobre `base` solo dentro de la máscara, con borde difuminado.

    Importante: fuera de la máscara la imagen queda idéntica al original.
    Los modelos devuelven la imagen entera y pueden alterarla levemente en
    todos lados; componer así garantiza que solo se toca lo marcado.

    El difuminado se aplica sobre una máscara dilatada, para que la banda de
    transición caiga FUERA del texto original y no lo reintroduzca.
    """
    dentro = (mask > 0).astype(np.float32)

    if feather and feather > 0:
        r = max(1, int(round(feather)))
        k = 2 * r + 1
        mask_d = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        alpha = cv2.GaussianBlur(mask_d.astype(np.float32) / 255.0, (0, 0), feather)
        # Sin esto, el desenfoque baja el alfa DENTRO de la máscara y los
        # píxeles del texto original se vuelven a mezclar en los bordes.
        alpha = np.maximum(alpha, dentro)
    else:
        alpha = dentro

    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    out = base.astype(np.float32) * (1 - alpha) + relleno.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# --- Rellenos -------------------------------------------------------------

def _rellenar_algoritmo(img, mask):
    """
    Inpainting clásico. Usa FSR de xphoto si está disponible (bastante mejor que
    Telea/NS en las pruebas), con respaldo a Telea+NS promediados.

    OJO: xphoto.inpaint espera la máscara INVERTIDA (255 = conocido) y escribe
    sobre un `dst` preasignado.
    """
    if tiene_fsr():
        try:
            conocido = cv2.bitwise_not(mask)
            dst = np.zeros_like(img)
            cv2.xphoto.inpaint(img, conocido, dst, cv2.xphoto.INPAINT_FSR_FAST)
            if dst.any():
                return dst
        except cv2.error:
            pass  # cae al respaldo

    a = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    b = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    return cv2.addWeighted(a, 0.5, b, 0.5, 0)


def _rellenar_ia(img, mask, backend=None, progress=None, base_prog=0.0, span=1.0):
    """
    Relleno por IA, delegado al backend elegido (ver backends.py).

    El default es "lama" (una sola pasada, sin promediar: el TTA se eliminó
    porque promediar reconstrucciones plausibles da una mancha, no textura —
    subía PSNR +1 dB y empeoraba LPIPS 39%). Los backends generativos
    ("sd_inpaint", "lama_sd_refine") muestrean textura real en vez de promediar,
    a cambio de inventar más y ser lentos en CPU.
    """
    nombre = backend or BACKEND_IA
    if progress:
        progress(base_prog + span * 0.05, f"IA ({nombre}): reconstruyendo…")

    def prog_backend(f, m):
        if progress:
            progress(base_prog + span * float(np.clip(f, 0.0, 1.0)), m)

    out = backends.rellenar(nombre, img, mask, progress=prog_backend)
    if progress:
        progress(base_prog + span, "IA: listo")
    return out


def mezclar(ia_img, algo_img, peso_ia=PESO_IA):
    """
    Mezcla ajustable IA/algoritmo (el "combinado").

    Se puede llamar en vivo desde la interfaz: como ambas imágenes ya están
    compuestas sobre el original, fuera de la máscara son idénticas y la mezcla
    no altera nada de lo que no se tocó.
    """
    p = float(np.clip(peso_ia, 0.0, 1.0))
    return cv2.addWeighted(ia_img, p, algo_img, 1.0 - p, 0)


# --- API principal --------------------------------------------------------

def restaurar(img_bgr, mask, metodos=("algoritmo", "ia", "combinado"),
              preset="equilibrado", backend=None, progress=None):
    """
    Restaura la zona marcada y devuelve un dict {metodo: imagen_bgr}.

    img_bgr : imagen original (BGR, uint8)
    mask    : máscara (255 = reconstruir). Se normaliza sola.
    metodos : cuáles generar.
    preset  : "rapido" | "equilibrado" | "maximo" (contexto que recibe la IA)
    backend : qué modelo de IA usar (None = BACKEND_IA, "lama" por defecto).
              Ver backends.disponibles(): "lama", "sd_inpaint", "lama_sd_refine".
    progress: callback(fraccion:float, mensaje:str)

    Todo el trabajo pesado se hace sobre un recorte alrededor de la máscara,
    a resolución nativa, y luego se compone sobre la imagen completa.
    """
    def rep(f, m):
        if progress:
            progress(min(1.0, max(0.0, f)), m)

    def _plena(base, parche, Y0, Y1, X0, X1):
        o = base.copy()
        o[Y0:Y1, X0:X1] = parche
        return o

    mask = normalizar_mascara(mask, img_bgr.shape)
    if np.count_nonzero(mask) == 0:
        raise ValueError("La máscara está vacía: marcá el texto que querés quitar.")

    pad_ratio = PRESETS_CONTEXTO.get(preset, PAD_RATIO)
    Y0, Y1, X0, X1 = _caja_recorte(mask, img_bgr.shape, pad_ratio)
    img_c = img_bgr[Y0:Y1, X0:X1].copy()
    mask_c = mask[Y0:Y1, X0:X1].copy()

    rep(0.02, f"Zona de trabajo: {X1-X0}x{Y1-Y0} px")

    necesita_algo = "algoritmo" in metodos or "combinado" in metodos
    necesita_ia = "ia" in metodos or "combinado" in metodos

    algo_c = ia_c = None
    if necesita_algo:
        rep(0.05, "Algoritmo clásico (FSR)...")
        algo_c = _rellenar_algoritmo(img_c, mask_c)
        rep(0.20, "Algoritmo clásico listo")

    if necesita_ia:
        ia_c = _rellenar_ia(img_c, mask_c, backend=backend,
                            progress=lambda f, m: rep(f, m),
                            base_prog=0.20, span=0.70)

    resultados = {}
    plena = np.zeros_like(img_bgr)

    if "algoritmo" in metodos:
        plena[:] = img_bgr
        plena[Y0:Y1, X0:X1] = algo_c
        resultados["algoritmo"] = _componer(img_bgr, plena, mask)

    if "ia" in metodos:
        plena = img_bgr.copy()
        plena[Y0:Y1, X0:X1] = ia_c
        resultados["ia"] = _componer(img_bgr, plena, mask)

    if "combinado" in metodos:
        rep(0.93, "Mezclando IA + algoritmo...")
        resultados["combinado"] = mezclar(resultados.get("ia") if "ia" in resultados
                                          else _componer(img_bgr, _plena(img_bgr, ia_c, Y0, Y1, X0, X1), mask),
                                          resultados.get("algoritmo") if "algoritmo" in resultados
                                          else _componer(img_bgr, _plena(img_bgr, algo_c, Y0, Y1, X0, X1), mask),
                                          PESO_IA)

    rep(1.0, "Listo")
    return resultados
