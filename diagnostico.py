"""
Diagnóstico de la detección de texto.

Corre todos los detectores sobre tu imagen, sin ocultar errores, y guarda una
imagen con las cajas dibujadas para ver qué encontró cada uno.

Uso:
    python -m restaurador.diagnostico mi_foto.png
"""

from __future__ import annotations

import sys
import os
import traceback

import cv2
import numpy as np


def main(argv=None):
    from . import consola_segura
    consola_segura()
    argv = argv or sys.argv[1:]
    if not argv:
        print("Uso: python -m restaurador.diagnostico mi_foto.png")
        return 1

    ruta = argv[0]
    if not os.path.exists(ruta):
        print(f"No existe el archivo: {ruta}")
        return 1

    from .app import leer_imagen
    from . import deteccion, motor

    img = leer_imagen(ruta)
    h, w = img.shape[:2]
    print(f"Imagen        : {os.path.basename(ruta)}  {w}x{h} px")
    print(f"OpenCV FSR    : {'sí' if motor.tiene_fsr() else 'NO (usaría Telea/NS)'}")
    print(f"EasyOCR       : {'instalado' if deteccion.hay_easyocr() else 'NO instalado'}")
    print("-" * 58)

    vis = img.copy()
    total = {}

    # 1) EasyOCR, sin ocultar el error
    if deteccion.hay_easyocr():
        try:
            print("Cargando EasyOCR (la 1ra vez descarga ~100 MB)...")
            reader = deteccion._get_reader()

            rt = reader.readtext(img)
            buenos = [x for x in rt if x[2] >= 0.15]
            print(f"readtext()    : {len(rt)} crudas, {len(buenos)} con confianza >= 0.15")
            for _b, t, c in rt[:12]:
                marca = "" if c >= 0.15 else "   <- descartada por confianza"
                print(f"                 \"{t[:34]}\" conf={c:.2f}{marca}")
            total["readtext"] = len(buenos)
            for b, _t, _c in buenos:
                cv2.polylines(vis, [np.array(b).astype(np.int32)], True, (0, 0, 255), 3)

            hor, free = reader.detect(img, text_threshold=0.6, low_text=0.3,
                                      link_threshold=0.3)
            n = len(hor[0]) + len(free[0])
            print(f"detect()      : {n} cajas (solo detector, sin leer)")
            total["detect"] = n
            for x0, x1, y0, y1 in hor[0]:
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        except Exception:
            print("EasyOCR FALLÓ. Este es el error que la app se estaba tragando:")
            print(traceback.format_exc())
            total["readtext"] = -1

    # 2) Detector morfológico (el respaldo)
    try:
        cajas = deteccion._cajas_morfologico(img)
        print(f"morfológico   : {len(cajas)} cajas")
        total["morfologico"] = len(cajas)
        for p in cajas:
            cv2.polylines(vis, [p], True, (255, 200, 0), 2)
    except Exception:
        print("morfológico FALLÓ:")
        print(traceback.format_exc())

    salida = os.path.splitext(ruta)[0] + "_diagnostico.png"
    cv2.imwrite(salida, vis)
    print("-" * 58)
    print(f"Imagen con las cajas: {salida}")
    print("  rojo = readtext | verde = detect | celeste = morfológico")

    if all(v <= 0 for v in total.values()):
        print("\nNingún detector vio el texto. Suele pasar cuando el texto es")
        print("muy grande, tiene poco contraste, o no parece texto de escena.")
        print("Marcalo con el pincel: es el camino más confiable igual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
