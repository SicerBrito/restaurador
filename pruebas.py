"""
Arnés de medición — tapa fotos reales a propósito y compara backends.

Metodología del proyecto: la única forma honesta de saber si un modelo es mejor
es taparle a una foto REAL una zona conocida, reconstruirla, y comparar contra
lo que HABÍA de verdad. Nada de mirar el resultado "a ojo" y decidir.

Este script:
  1) Toma una carpeta de fotos LIMPIAS (sin texto encima ya).
  2) Sobre cada una dibuja texto sintético en una zona al azar -> queda una
     versión "tapada" + su máscara + la verdad conocida (la foto original).
  3) Reconstruye con cada backend pedido y mide contra la verdad con metricas.py.
  4) Imprime una tabla y (opcional) guarda las imágenes para inspección.

Uso:
    python -m restaurador.pruebas fotos_limpias/ --backends lama
    python -m restaurador.pruebas fotos_limpias/ --backends lama,sd_inpaint,lama_sd_refine --guardar salida_pruebas/

Nota: los backends de difusión necesitan `diffusers` instalado y son lentos en
CPU. "lama" corre siempre (si está simple-lama-inpainting).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import cv2

from . import motor, metricas, consola_segura
from .app import leer_imagen, guardar_imagen


EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")
FRASES = ["MUESTRA", "COPYRIGHT 2024", "Sin valor", "www.ejemplo.com", "PROHIBIDO"]


def _tapar_con_texto(img, rng):
    """
    Dibuja texto sintético sobre la imagen y devuelve (tapada, mascara).

    La máscara marca exactamente los píxeles del texto dibujado (trazos), que es
    justo lo que la app tendría que reconstruir. Así la verdad conocida = img.
    """
    h, w = img.shape[:2]
    tapada = img.copy()
    mask = np.zeros((h, w), np.uint8)

    escala = max(0.8, min(w, h) / 400.0)
    grosor = max(2, int(escala * 2))
    n = int(rng.integers(1, 4))
    for _ in range(n):
        texto = FRASES[int(rng.integers(0, len(FRASES)))]
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        x = int(rng.integers(5, max(6, w - tw - 5)))
        y = int(rng.integers(th + 5, max(th + 6, h - 5)))
        color = (255, 255, 255) if rng.random() < 0.5 else (0, 0, 0)
        cv2.putText(tapada, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, color, grosor, cv2.LINE_AA)
        cv2.putText(mask, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, 255, grosor, cv2.LINE_AA)

    # dilatar un poco: el antialias del texto ensucia el borde, igual que en la app
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return tapada, mask


def _promedio(dicts, clave):
    vals = [d[clave] for d in dicts if d.get(clave) is not None]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    consola_segura()
    p = argparse.ArgumentParser(
        prog="restaurador.pruebas",
        description="Tapa fotos reales y compara backends de reconstrucción.")
    p.add_argument("carpeta", help="Carpeta con fotos LIMPIAS (sin texto encima)")
    p.add_argument("--backends", default="lama",
                   help="Backends a comparar, separados por coma")
    p.add_argument("--preset", default="equilibrado",
                   choices=tuple(motor.PRESETS_CONTEXTO))
    p.add_argument("--n", type=int, default=0, help="Máx. de fotos a usar (0 = todas)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--guardar", default=None,
                   help="Carpeta donde guardar tapada/máscara/reconstrucciones")
    args = p.parse_args(argv)

    rutas = []
    for e in EXTS:
        rutas += glob.glob(os.path.join(args.carpeta, e))
    rutas = sorted(rutas)
    if args.n > 0:
        rutas = rutas[: args.n]
    if not rutas:
        print(f"No encontré fotos en {args.carpeta}")
        return 1

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    rng = np.random.default_rng(args.semilla)

    if not metricas.hay_lpips():
        print("Aviso: falta el paquete `lpips` (o torch). Mido textura y nitidez; "
              "LPIPS aparecerá como n/d.\n")

    if args.guardar:
        os.makedirs(args.guardar, exist_ok=True)

    por_backend = {b: [] for b in backends}
    print(f"Fotos: {len(rutas)} | backends: {', '.join(backends)} | preset: {args.preset}")
    print("=" * 78)

    for ruta in rutas:
        nombre = os.path.basename(ruta)
        original = leer_imagen(ruta)
        tapada, mask = _tapar_con_texto(original, rng)
        print(f"\n{nombre}  ({original.shape[1]}x{original.shape[0]}, "
              f"{int(np.count_nonzero(mask))} px tapados)")

        if args.guardar:
            base = os.path.splitext(nombre)[0]
            guardar_imagen(os.path.join(args.guardar, f"{base}_tapada.png"), tapada)
            cv2.imwrite(os.path.join(args.guardar, f"{base}_mascara.png"), mask)

        for b in backends:
            try:
                t = time.time()
                res = motor.restaurar(tapada, mask, metodos=("ia",),
                                      preset=args.preset, backend=b)
                dt = time.time() - t
                m = metricas.comparar(original, res["ia"], mask)
                por_backend[b].append(m)
                print(f"  {b:16s} {metricas.formatear(m)}  ({dt:.1f}s)")
                if args.guardar:
                    guardar_imagen(
                        os.path.join(args.guardar, f"{base}_{b}.png"), res["ia"])
            except Exception as e:
                print(f"  {b:16s} ERROR: {e}")

    # --- resumen ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("PROMEDIOS  (LPIPS menor=mejor | textura ideal=1.0 | nitidez mayor=mas detalle)")
    print("-" * 78)
    for b in backends:
        ms = por_backend[b]
        if not ms:
            print(f"  {b:16s} sin resultados")
            continue
        lp = _promedio(ms, "lpips")
        lp_s = "n/d" if lp is None else f"{lp:.4f}"
        tx = _promedio(ms, "textura_ratio")
        nz = _promedio(ms, "nitidez")
        print(f"  {b:16s} LPIPS={lp_s}   textura={tx:.2f}   nitidez={nz:.0f}   "
              f"(n={len(ms)})")
    print("\nRecordá: la máscara sigue siendo la palanca #1. Estos números comparan "
          "modelos con la MISMA máscara.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
