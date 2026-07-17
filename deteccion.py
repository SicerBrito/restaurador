"""
Detección automática del texto superpuesto en una imagen.

Dos motores:
  - EasyOCR (detector CRAFT): preciso, requiere descargar modelos la primera vez.
  - Respaldo morfológico: sin modelos ni descargas; busca zonas de alto
    contraste con forma de renglón. Menos preciso, pero siempre disponible.

Sobre la máscara resultante hay dos modos, ambos medidos:
  - "trazos": dentro de cada caja detectada marca solo los píxeles de las letras.
    Midió mejor PSNR (15.89 vs 15.73 dB) usando la mitad del área, porque
    conserva el fondo real entre letra y letra.
  - "caja": marca el rectángulo completo. Más agresivo; útil cuando el texto
    tiene sombra fuerte, borde o una banda de fondo semitransparente.
"""

from __future__ import annotations

import traceback

import numpy as np
import cv2

_reader_cache = None
ultimo_motor = None      # qué detector se usó realmente
ultimo_error = None      # traceback si EasyOCR falló


def hay_easyocr() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def _get_reader(idiomas=("es", "en")):
    global _reader_cache
    if _reader_cache is None:
        import easyocr
        _reader_cache = easyocr.Reader(list(idiomas), gpu=False, verbose=False)
    return _reader_cache


def _cajas_easyocr(img_bgr, sensibilidad=0.5):
    """
    Devuelve polígonos de texto.

    Se unen dos fuentes porque buscan cosas distintas:
      - readtext(): detecta Y lee. Filtra por confianza de LECTURA, así que un
        texto con tipografía decorativa puede detectarse bien y descartarse igual.
      - detect(): solo el detector CRAFT, sin leer. No depende de que el texto
        sea legible.
    Nos interesan las cajas, no leer el texto, así que la unión da más alcance.
    Los solapamientos no molestan: fillPoly los resuelve.

    `sensibilidad` (0..1) mueve los umbrales de CRAFT: más alto detecta texto más
    tenue o pequeño (a costa de algún falso positivo); más bajo, solo lo evidente.
    """
    s = float(np.clip(sensibilidad, 0.0, 1.0))
    umbral_conf = 0.25 - 0.20 * s          # 0.05 (sensible) .. 0.25 (estricto)
    text_thr = 0.80 - 0.40 * s             # 0.40 .. 0.80
    low_text = 0.50 - 0.30 * s             # 0.20 .. 0.50
    link_thr = 0.50 - 0.30 * s

    reader = _get_reader()
    salida = []

    for box, _texto, conf in reader.readtext(img_bgr):
        if conf >= umbral_conf:
            salida.append(np.array(box).astype(np.int32))

    hor, free = reader.detect(img_bgr, text_threshold=text_thr, low_text=low_text,
                              link_threshold=link_thr)
    for x0, x1, y0, y1 in hor[0]:
        salida.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32))
    for poly in free[0]:
        if poly is not None and len(poly) >= 3:
            salida.append(np.array(poly).astype(np.int32))

    return salida


def _cajas_morfologico(img_bgr):
    """
    Respaldo sin IA: el texto superpuesto suele ser de alto contraste y
    agrupado en renglones. Se realza el gradiente, se unen caracteres en
    líneas y se filtran los blobs con forma de renglón de texto.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    H, W = gray.shape
    ancho_union = max(9, W // 60)
    conectado = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (ancho_union, 3)))

    contornos, _ = cv2.findContours(conectado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h < 8 or w < 12:
            continue
        if h > H * 0.5 or w > W * 0.98:
            continue
        if w / float(h) < 1.2:          # los renglones son más anchos que altos
            continue
        if cv2.contourArea(c) / float(w * h) < 0.25:
            continue
        cajas.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32))
    return cajas


def _polaridad_texto(sub, th):
    """
    Decide cuál de las dos clases de Otsu es el TEXTO y si es más claro que el fondo.

    Antes se asumía "texto = clase minoritaria", que falla con texto grande o en
    negrita (ocupa más de la mitad de la caja). Acá el criterio es el fondo real:
    se estima con el borde de la caja (casi siempre fondo, no letra) y el texto es
    la clase cuya intensidad se aleja MÁS de ese fondo.

    Devuelve (mascara_texto_uint8, texto_mas_claro_que_fondo: bool).
    """
    borde = np.concatenate([sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1]])
    fondo = float(np.median(borde))
    blancos = sub[th > 0]
    negros = sub[th == 0]
    m_blanco = float(blancos.mean()) if blancos.size else fondo
    m_negro = float(negros.mean()) if negros.size else fondo
    if abs(m_blanco - fondo) >= abs(m_negro - fondo):
        return th, (m_blanco >= fondo)                 # los blancos = texto
    return cv2.bitwise_not(th), (m_negro >= fondo)     # los negros = texto


def _mascara_trazos(img_bgr, cajas, dilatar=3):
    """
    Dentro de cada caja separa las letras del fondo con Otsu y decide la polaridad
    con el fondo real del borde (ver _polaridad_texto). Para cajas grandes con
    fondo no uniforme (degradados, sombras) intersecta con un umbral ADAPTATIVO,
    que sigue las variaciones locales del fondo y recorta más fino que Otsu solo.
    Luego dilata para cubrir el antialias y la sombra del borde.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    mask = np.zeros((H, W), np.uint8)

    for poly in cajas:
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        if w <= 1 or h <= 1:
            continue

        sub = gray[y:y + h, x:x + w]
        _, th = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        texto, texto_claro = _polaridad_texto(sub, th)

        # Fondo no uniforme: refinar con umbral adaptativo, que sigue el fondo local.
        # adap marca en blanco lo que está por encima del fondo local; si el texto
        # es más claro que el fondo, el texto ~ adap, y si es más oscuro, ~ not adap.
        if min(w, h) >= 24:
            blk = max(11, (min(w, h) // 2) | 1)       # ventana impar ~ medio alto
            adap = cv2.adaptiveThreshold(
                sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, 5)
            adap_texto = adap if texto_claro else cv2.bitwise_not(adap)
            inter = cv2.bitwise_and(texto, adap_texto)
            # Solo si conserva algo razonable (evita borrar la máscara si discrepan).
            if np.count_nonzero(inter) > 0.15 * max(1, np.count_nonzero(texto)):
                texto = inter

        roi = np.zeros((H, W), np.uint8)
        cv2.fillPoly(roi, [poly], 255)
        tmp = np.zeros((H, W), np.uint8)
        tmp[y:y + h, x:x + w] = texto
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(tmp, roi))

    if dilatar > 0:
        k = 2 * int(dilatar) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask


def _mascara_caja(shape, cajas, dilatar=2):
    mask = np.zeros(shape[:2], np.uint8)
    for poly in cajas:
        cv2.fillPoly(mask, [poly], 255)
    if dilatar > 0:
        k = 2 * int(dilatar) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask


def detectar_texto(img_bgr, modo="trazos", motor="auto", dilatar=3, sensibilidad=0.5):
    """
    Devuelve (mascara_uint8, cajas).

    modo         : "trazos" (recomendado) | "caja"
    motor        : "auto" | "easyocr" | "morfologico"
    sensibilidad : 0..1, cuánto texto tenue/pequeño intenta detectar EasyOCR.
    """
    if motor == "auto":
        motor = "easyocr" if hay_easyocr() else "morfologico"

    global ultimo_motor, ultimo_error
    ultimo_error = None

    if motor == "easyocr":
        try:
            cajas = _cajas_easyocr(img_bgr, sensibilidad=sensibilidad)
            ultimo_motor = "easyocr"
        except Exception as e:
            # Antes esto se tragaba el error en silencio y el usuario solo veía
            # "no encontré texto", sin saber que EasyOCR se había caído.
            ultimo_error = traceback.format_exc()
            ultimo_motor = "morfologico (EasyOCR falló: %s)" % str(e)[:80]
            cajas = _cajas_morfologico(img_bgr)
    else:
        cajas = _cajas_morfologico(img_bgr)
        ultimo_motor = "morfologico"

    if not cajas:
        return np.zeros(img_bgr.shape[:2], np.uint8), []

    if modo == "caja":
        mask = _mascara_caja(img_bgr.shape, cajas, dilatar=max(1, dilatar - 1))
    else:
        mask = _mascara_trazos(img_bgr, cajas, dilatar=dilatar)

    return mask, cajas
