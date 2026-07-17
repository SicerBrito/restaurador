"""
Métricas para comparar reconstrucciones — el termómetro del proyecto.

La regla de oro acá es NO usar PSNR/SSIM para decidir calidad de textura:
premian el desenfoque (acertarle al promedio penaliza menos que arriesgar una
textura nítida y errarle), y por eso la primera versión terminó en un manchón.

Se miden tres cosas, en orden de importancia para este problema:

  1) LPIPS  ↓  — distancia perceptual (se acerca a lo que ve el ojo). Necesita
                 torch + el paquete `lpips`. Es opcional: si no está, se omite.
  2) textura     — energía de textura del relleno / la de su entorno. 1.0 = misma
                 textura que alrededor; <1 = más liso (mancha); >1 = más ruidoso.
  3) nitidez     — varianza del Laplaciano dentro de la zona. Sube con el detalle;
                 se compara contra la del entorno como referencia.

textura y nitidez son numpy puro (corren sin cv2 ni torch), para poder validarlas
en cualquier entorno. Solo LPIPS trae dependencias pesadas.
"""
from __future__ import annotations

import numpy as np


def _a_gris(img):
    """BGR/RGB uint8 -> gris float32. (El orden de canales no cambia el Laplaciano.)"""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def _laplaciano(gris):
    """Laplaciano 4-vecinos, sin cv2 (bordes por reflexión con np.pad)."""
    p = np.pad(gris, 1, mode="reflect")
    return (-4 * gris
            + p[:-2, 1:-1] + p[2:, 1:-1]
            + p[1:-1, :-2] + p[1:-1, 2:])


def _mascara_bool(mask, shape):
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def _anillo(mask_bool, grosor=12):
    """
    Anillo de contexto alrededor de la máscara (dilatación menos la máscara),
    sin cv2: se dilata con desplazamientos acumulados. Es la referencia de
    "cómo es la textura real del material de al lado".
    """
    d = mask_bool.copy()
    for _ in range(int(grosor)):
        s = d.copy()
        s[:-1, :] |= d[1:, :]
        s[1:, :] |= d[:-1, :]
        s[:, :-1] |= d[:, 1:]
        s[:, 1:] |= d[:, :-1]
        d = s
    return d & ~mask_bool


def energia_textura(img, mask):
    """std del Laplaciano dentro de la máscara. Cuanta más textura, más alto."""
    m = _mascara_bool(mask, img.shape)
    if not m.any():
        return 0.0
    lap = _laplaciano(_a_gris(img))
    return float(lap[m].std())


def ratio_textura(img, mask, grosor_anillo=12):
    """
    Textura del relleno relativa a la de su entorno inmediato.
    1.0 = igual que alrededor (ideal). <1 = más liso (mancha). >1 = más ruidoso.
    Es la métrica que delata el desenfoque de un modelo de regresión.
    """
    m = _mascara_bool(mask, img.shape)
    if not m.any():
        return 0.0
    lap = _laplaciano(_a_gris(img))
    dentro = lap[m].std()
    ring = _anillo(m, grosor_anillo)
    fuera = lap[ring].std() if ring.any() else 0.0
    if fuera < 1e-6:
        return 0.0
    return float(dentro / fuera)


def nitidez(img, mask):
    """Varianza del Laplaciano dentro de la máscara (medida clásica de foco)."""
    m = _mascara_bool(mask, img.shape)
    if not m.any():
        return 0.0
    lap = _laplaciano(_a_gris(img))
    return float(lap[m].var())


# --- LPIPS (opcional) -----------------------------------------------------

_lpips_cache = None


def hay_lpips() -> bool:
    try:
        import lpips, torch  # noqa: F401
        return True
    except Exception:
        return False


def _get_lpips():
    global _lpips_cache
    if _lpips_cache is None:
        import lpips
        _lpips_cache = lpips.LPIPS(net="alex", verbose=False)
    return _lpips_cache


def lpips_dist(a_bgr, b_bgr):
    """
    Distancia perceptual LPIPS entre dos imágenes (mismo tamaño). ↓ es mejor.
    Devuelve None si `lpips`/torch no están instalados.
    """
    if not hay_lpips():
        return None
    import torch

    def prep(x):
        x = np.asarray(x, np.float32)[..., ::-1]      # BGR -> RGB
        x = x / 127.5 - 1.0                            # a [-1, 1]
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
        return t

    net = _get_lpips()
    with torch.no_grad():
        d = net(prep(a_bgr), prep(b_bgr))
    return float(d.item())


# --- Comparación de una reconstrucción contra la verdad conocida ----------

def comparar(original, reconstruida, mask):
    """
    Compara `reconstruida` contra el `original` conocido dentro de `mask`.

    Pensado para el arnés: se tapa una zona de una foto real, se reconstruye y se
    mide contra lo que HABÍA de verdad. Devuelve un dict de métricas.
    """
    return {
        "lpips": lpips_dist(original, reconstruida),               # ↓ mejor
        "textura_ratio": ratio_textura(reconstruida, mask),        # →1.0 ideal
        "textura_ratio_ref": ratio_textura(original, mask),        # el objetivo real
        "nitidez": nitidez(reconstruida, mask),
        "nitidez_ref": nitidez(original, mask),
    }


def formatear(m):
    """Una línea legible a partir del dict de comparar()."""
    lp = "n/d" if m.get("lpips") is None else f"{m['lpips']:.4f}"
    return (f"LPIPS={lp}  "
            f"textura={m['textura_ratio']:.2f} (real {m['textura_ratio_ref']:.2f})  "
            f"nitidez={m['nitidez']:.0f} (real {m['nitidez_ref']:.0f})")


# --- Auto-test (numpy puro, sin modelos): valida el sentido de las métricas -

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H = W = 200
    tex = (rng.standard_normal((H, W, 3)) * 40 + 128).clip(0, 255).astype(np.uint8)
    mask = np.zeros((H, W), np.uint8)
    mask[70:130, 70:130] = 255

    liso = tex.copy()
    liso[70:130, 70:130] = 128           # relleno plano = mancha
    real = tex                            # relleno con textura real

    print("Zona rellenada LISA vs con TEXTURA REAL (misma máscara)")
    print("  liso :", formatear(comparar(tex, liso, mask)))
    print("  real :", formatear(comparar(tex, real, mask)))
    r_liso = ratio_textura(liso, mask)
    r_real = ratio_textura(real, mask)
    assert r_liso < 0.3, r_liso          # la mancha colapsa la textura
    assert 0.7 < r_real < 1.4, r_real    # la textura real ~ iguala al entorno
    print(f"\nOK: mancha ratio={r_liso:.2f} (colapsa), textura ratio={r_real:.2f} (~1.0)")
