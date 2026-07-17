"""
Restaurador de fotos — línea de comandos (para lotes o uso sin ventana).

Ejemplos:
    # detectar el texto solo y generar las 3 versiones
    python3 -m restaurador.cli foto.jpg --auto -o salida/

    # usar una máscara propia (blanco = reconstruir)
    python3 -m restaurador.cli foto.jpg --mascara mascara.png -o salida/

    # más calidad (más lento)
    python3 -m restaurador.cli foto.jpg --auto --preset maximo -o salida/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from . import motor, deteccion, backends, consola_segura
from .app import leer_imagen, guardar_imagen


def main(argv=None):
    consola_segura()
    p = argparse.ArgumentParser(
        prog="restaurador",
        description="Quita texto superpuesto de una imagen y reconstruye lo de abajo.")
    p.add_argument("imagen", help="Imagen de entrada")
    p.add_argument("-o", "--salida", default="salida", help="Carpeta de salida")
    p.add_argument("--mascara", help="Máscara propia (blanco = zona a reconstruir)")
    p.add_argument("--auto", action="store_true", help="Detectar el texto automáticamente")
    p.add_argument("--modo", choices=("trazos", "caja"), default="trazos",
                   help="trazos: solo las letras (mejor). caja: el rectángulo entero.")
    p.add_argument("--sensibilidad", choices=("baja", "media", "alta"), default="media",
                   help="Cuánto texto tenue/pequeño intenta detectar (solo con --auto)")
    p.add_argument("--solo-mascara", action="store_true",
                   help="Solo detectar y guardar la máscara (no restaura)")
    p.add_argument("--preset", choices=tuple(motor.PRESETS_TTA), default="equilibrado",
                   help="Contexto que recibe la IA (rapido/equilibrado/maximo)")
    p.add_argument("--backend", choices=tuple(backends.disponibles()),
                   default=motor.BACKEND_IA,
                   help="Modelo de IA: lama (rápido), sd_inpaint o lama_sd_refine "
                        "(más textura, necesitan diffusers y son lentos en CPU)")
    p.add_argument("--metodos", default="algoritmo,ia,combinado",
                   help="Cuáles generar, separados por coma")
    args = p.parse_args(argv)

    if not args.auto and not args.mascara:
        p.error("Indicá --auto para detectar el texto, o --mascara con tu propia máscara.")

    img = leer_imagen(args.imagen)
    h, w = img.shape[:2]
    print(f"Imagen: {os.path.basename(args.imagen)} ({w}x{h})")
    print(f"FSR (algoritmo avanzado) disponible: {'sí' if motor.tiene_fsr() else 'no, uso Telea/NS'}")
    print(f"Backend de IA: {args.backend}")

    if args.mascara:
        m = cv2.imread(args.mascara, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"No pude leer la máscara: {args.mascara}", file=sys.stderr)
            return 2
        mask = motor.normalizar_mascara(m, img.shape)
    else:
        print("Detectando texto…")
        sens = {"baja": 0.25, "media": 0.5, "alta": 0.8}[args.sensibilidad]
        t = time.time()
        mask, cajas = deteccion.detectar_texto(img, modo=args.modo, sensibilidad=sens)
        print(f"  {len(cajas)} zona(s) de texto en {time.time()-t:.1f}s")
        if not cajas:
            print("No encontré texto. Probá --modo caja o --sensibilidad alta, "
                  "o hacé la máscara a mano con la app.", file=sys.stderr)
            return 1

    area = int(np.count_nonzero(mask))
    print(f"Área a reconstruir: {area} px ({100*area/(w*h):.2f}% de la imagen)")

    os.makedirs(args.salida, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.imagen))[0]

    if args.solo_mascara:
        ruta_m = os.path.join(args.salida, f"{base}_mascara.png")
        cv2.imwrite(ruta_m, mask)
        print(f"Solo máscara. Guardada:\n  -> {ruta_m}")
        return 0

    metodos = tuple(x.strip() for x in args.metodos.split(",") if x.strip())

    t = time.time()
    res = motor.restaurar(img, mask, metodos=metodos, preset=args.preset,
                          backend=args.backend,
                          progress=lambda f, m: print(f"  [{f*100:3.0f}%] {m}", end="\r"))
    print(" " * 70, end="\r")
    print(f"Restaurado en {time.time()-t:.1f}s")

    for k, v in res.items():
        ruta = os.path.join(args.salida, f"{base}_{k}.png")
        guardar_imagen(ruta, v)
        print(f"  -> {ruta}")
    ruta_m = os.path.join(args.salida, f"{base}_mascara.png")
    cv2.imwrite(ruta_m, mask)
    print(f"  -> {ruta_m}")

    print("\nCompará las 3: cuál gana depende de la imagen. Son reconstrucciones "
          "plausibles, no los píxeles originales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
