"""
Backends de relleno por IA — intercambiables y medibles.

La idea central del proyecto es que CADA decisión se mide. Este módulo hace que
el modelo de IA sea una pieza enchufable: se puede cambiar LaMa por difusión (o
por un híbrido) sin tocar el motor, y compararlos con el mismo arnés de medición.

Un backend es una función:

    fn(img_bgr_crop, mask_crop, progress=None, **kw) -> img_bgr_crop_rellenado

Recibe un RECORTE (BGR uint8) y su máscara (255 = reconstruir, la misma
convención de todo el proyecto — al revés que FSR) y devuelve el recorte
rellenado, del MISMO tamaño. Componer sobre la imagen completa y respetar la
zona fuera de la máscara es tarea del motor, no del backend.

Backends incluidos:
  - "lama"           : LaMa, una pasada limpia. El default histórico y medido.
  - "sd_inpaint"     : Stable Diffusion inpainting. Muestrea textura real (no
                       promedia como LaMa) => más nítido, pero inventa más.
  - "lama_sd_refine" : LaMa da la estructura, SD img2img (denoise bajo) le agrega
                       textura anclada a esa estructura. Menos alucinación que SD
                       puro; respeta la filosofía "reconstruir la foto, no inventar".

Los modelos pesados se importan y descargan de forma PEREZOSA: importar este
módulo no requiere torch ni diffusers. Solo el backend que uses trae sus deps.
"""
from __future__ import annotations

import numpy as np
import cv2

# --- Registro -------------------------------------------------------------

_BACKENDS = {}
BACKEND_DEFECTO = "lama"

# Stable Diffusion 1.x/2.x fue entrenado a 512 px. Trabajamos por tiles de este
# tamaño para NO reescalar la imagen (reescalar antes del modelo degrada mucho:
# medido -1.7 dB a 768 px, -3.5 dB a 512 px).
TILE = 512
SOLAPE = 96            # solape entre tiles para fundir costuras sin bordes visibles
PROMPT_NEG = "text, letters, words, watermark, logo, caption, subtitle, signature"


def backend(nombre):
    """Decorador para registrar un backend por nombre."""
    def deco(fn):
        _BACKENDS[nombre] = fn
        return fn
    return deco


def disponibles():
    """Nombres de backends registrados, en orden estable."""
    return sorted(_BACKENDS)


def rellenar(nombre, img, mask, progress=None, **kw):
    """Ejecuta el backend `nombre` sobre un recorte. Lanza si no existe."""
    fn = _BACKENDS.get(nombre)
    if fn is None:
        raise ValueError(
            f"Backend de IA desconocido: {nombre!r}. "
            f"Disponibles: {', '.join(disponibles())}")
    return fn(img, mask, progress=progress, **kw)


# --- Utilidad: aplicar un modelo 512x512 sobre un recorte de cualquier tamaño

def _a_multiplo_8(n):
    return int(np.ceil(n / 8.0) * 8)


def _ventana_fundido(h, w, borde):
    """
    Pesos con bordes suaves (coseno alzado) para fundir tiles solapados sin
    dejar costura. En el centro el peso es 1; decae hacia los bordes.
    """
    def rampa(n):
        v = np.ones(n, np.float32)
        b = min(borde, n // 2)
        if b > 0:
            r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, b, dtype=np.float32)))
            v[:b] = r
            v[-b:] = r[::-1]
        return v
    return np.outer(rampa(h), rampa(w))


def procesar_por_tiles(img, mask, fn_tile, tile=TILE, solape=SOLAPE, progress=None):
    """
    Recorre `img` en tiles de `tile` px que contengan máscara, llama
    `fn_tile(sub_bgr, sub_mask) -> sub_bgr` en cada uno y funde los resultados.

    - Nunca reescala: cada tile se procesa a resolución nativa.
    - Solo procesa tiles que tocan la máscara (los demás quedan como estaban).
    - Los solapes se funden con una ventana suave para que no se vean costuras.

    Si toda la imagen entra en un tile, se hace una sola pasada (caso común de
    una palabra o un renglón corto).
    """
    H, W = img.shape[:2]
    acum = np.zeros((H, W, 3), np.float32)
    peso = np.zeros((H, W), np.float32)

    paso = max(1, tile - solape)
    ys = list(range(0, max(1, H - solape), paso))
    xs = list(range(0, max(1, W - solape), paso))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    ys, xs = sorted(set(ys)), sorted(set(xs))

    total = len(ys) * len(xs)
    hecho = 0
    for y0 in ys:
        for x0 in xs:
            hecho += 1
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            sub_m = mask[y0:y1, x0:x1]
            if int(sub_m.max()) == 0:
                continue  # nada que reconstruir en este tile
            sub = img[y0:y1, x0:x1]
            res = fn_tile(sub, sub_m)
            res = res[: y1 - y0, : x1 - x0]
            w = _ventana_fundido(y1 - y0, x1 - x0, solape // 2)
            acum[y0:y1, x0:x1] += res.astype(np.float32) * w[..., None]
            peso[y0:y1, x0:x1] += w
            if progress:
                progress(hecho / total, f"IA por tiles {hecho}/{total}")

    salida = img.astype(np.float32).copy()
    tocado = peso > 1e-6
    salida[tocado] = acum[tocado] / peso[tocado][..., None]
    # rint (no truncar): la división por los pesos deja error de coma flotante y
    # astype() truncaría hacia abajo, metiendo ruido de ±1 en TODA la zona
    # procesada aunque el modelo no la haya cambiado.
    return np.clip(np.rint(salida), 0, 255).astype(np.uint8)


# --- Backend: LaMa (default histórico) ------------------------------------

_lama_cache = None


def _get_lama():
    """Carga LaMa una sola vez (~200 MB, se descarga al primer uso)."""
    global _lama_cache
    if _lama_cache is None:
        from simple_lama_inpainting import SimpleLama
        _lama_cache = SimpleLama()
    return _lama_cache


@backend("lama")
def lama(img, mask, progress=None, **kw):
    """
    Una pasada de LaMa. Recorta el padding a múltiplos de 8 que el modelo agrega
    (si no, la imagen queda desalineada).

    Es un modelo de REGRESIÓN: en zonas ambiguas devuelve el promedio de todo lo
    plausible, lo que aplana la textura. Para arreglar eso están los backends
    generativos de abajo.
    """
    from PIL import Image
    lm = _get_lama()
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask)
    out = np.array(lm(pil_img, pil_mask))
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out[: img.shape[0], : img.shape[1]]


# --- Stable Diffusion (carga perezosa, cache por tipo de pipeline) ---------

_sd_cache = {}


def _dispositivo_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32     # en CPU float16 no sirve


def _get_sd(tipo, modelo=None):
    """
    Carga y cachea un pipeline de diffusers.

    tipo = "inpaint" | "img2img"
    En CPU tarda; por eso se usa solo en el preset de máxima calidad.
    """
    clave = (tipo, modelo)
    if clave in _sd_cache:
        return _sd_cache[clave]

    import torch  # noqa: F401
    dispositivo, dtype = _dispositivo_dtype()

    # Modelos SD 1.5 en mirrors PÚBLICOS (sin cuenta ni token de HuggingFace):
    # los repos originales de runwayml/stabilityai quedaron borrados o "gated".
    # SD 1.5 además es más liviano y rápido en CPU que SD2, y su inpainting es de
    # 9 canales (hecho para esto). Se puede sobreescribir con el argumento `modelo`.
    if tipo == "inpaint":
        from diffusers import StableDiffusionInpaintPipeline
        mid = modelo or "stable-diffusion-v1-5/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            mid, torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False)
    elif tipo == "img2img":
        from diffusers import StableDiffusionImg2ImgPipeline
        mid = modelo or "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            mid, torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False)
    else:
        raise ValueError(f"Tipo de pipeline SD desconocido: {tipo}")

    pipe = pipe.to(dispositivo)
    pipe.set_progress_bar_config(disable=True)
    # En 24 GB de RAM y CPU conviene ir por partes para no reventar la memoria.
    # (vae.enable_slicing en vez de pipe.enable_vae_slicing, deprecado en 0.40.)
    try:
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    except Exception:
        pass
    _sd_cache[clave] = pipe
    return pipe


def _bgr_a_pil(img):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _pil_a_bgr(pil):
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def _pad_a_8(img, mask):
    """SD exige dimensiones múltiplo de 8. Rellena por reflexión y devuelve el
    tamaño original para recortar después."""
    h, w = img.shape[:2]
    H8, W8 = _a_multiplo_8(h), _a_multiplo_8(w)
    if (H8, W8) == (h, w):
        return img, mask, (h, w)
    img_p = cv2.copyMakeBorder(img, 0, H8 - h, 0, W8 - w, cv2.BORDER_REFLECT)
    mask_p = cv2.copyMakeBorder(mask, 0, H8 - h, 0, W8 - w, cv2.BORDER_CONSTANT, value=0)
    return img_p, mask_p, (h, w)


# --- Backend: Stable Diffusion inpainting ---------------------------------

@backend("sd_inpaint")
def sd_inpaint(img, mask, progress=None, prompt="", pasos=25, guidance=5.0,
               modelo=None, semilla=None, **kw):
    """
    Inpainting con Stable Diffusion. A diferencia de LaMa, MUESTREA una textura
    concreta en vez de promediar => resultado nítido. Como contrapartida inventa
    más: ideal para fondos (cielo, pared, césped), arriesgado sobre contenido único.

    El prompt negativo evita que el modelo vuelva a "escribir" texto donde justo
    lo estamos borrando.
    """
    import torch
    pipe = _get_sd("inpaint", modelo)
    gen = None
    if semilla is not None:
        gen = torch.Generator(device=pipe.device).manual_seed(int(semilla))

    def _tile(sub, sub_m):
        sub_p, m_p, (h, w) = _pad_a_8(sub, sub_m)
        out = pipe(
            prompt=prompt or "",
            negative_prompt=PROMPT_NEG,
            image=_bgr_a_pil(sub_p),
            mask_image=_bgr_a_pil(cv2.cvtColor(m_p, cv2.COLOR_GRAY2BGR)),
            num_inference_steps=int(pasos),
            guidance_scale=float(guidance),
            generator=gen,
            height=sub_p.shape[0], width=sub_p.shape[1],
        ).images[0]
        return _pil_a_bgr(out)[:h, :w]

    return procesar_por_tiles(img, mask, _tile, progress=progress)


# --- Backend: híbrido LaMa -> refinado con SD -----------------------------

@backend("lama_sd_refine")
def lama_sd_refine(img, mask, progress=None, prompt="", pasos=30, strength=0.35,
                   guidance=6.0, modelo=None, semilla=None, **kw):
    """
    LaMa pone la ESTRUCTURA coherente; encima SD img2img con denoise bajo
    (strength ~0.35) le devuelve TEXTURA nítida sin partir de cero. Como arranca
    de la reconstrucción de LaMa, inventa mucho menos que sd_inpaint: es la
    versión que mejor respeta "reconstruir la foto, no inventar el secreto".

    El motor compone solo la máscara sobre el original, así que los píxeles
    conocidos quedan intactos aunque img2img toque todo el tile.
    """
    import torch
    if progress:
        progress(0.05, "Refinado: estructura con LaMa…")
    base = lama(img, mask)

    pipe = _get_sd("img2img", modelo)
    gen = None
    if semilla is not None:
        gen = torch.Generator(device=pipe.device).manual_seed(int(semilla))

    def _tile(sub, sub_m):
        sub_p, _m_p, (h, w) = _pad_a_8(sub, sub_m)
        out = pipe(
            prompt=prompt or "",
            negative_prompt=PROMPT_NEG,
            image=_bgr_a_pil(sub_p),
            strength=float(strength),
            num_inference_steps=int(pasos),
            guidance_scale=float(guidance),
            generator=gen,
        ).images[0]
        return _pil_a_bgr(out)[:h, :w]

    # Solo refinamos donde LaMa reconstruyó; el resto del tile no se toca.
    return procesar_por_tiles(base, mask, _tile, progress=progress)
