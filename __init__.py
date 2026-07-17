"""Restaurador de fotos: quita texto superpuesto y reconstruye lo de abajo."""
from . import motor, deteccion, backends, metricas  # noqa: F401

__version__ = "1.1"


def consola_segura():
    """
    La consola de Windows suele ser cp1252 y revienta (UnicodeEncodeError) al
    imprimir símbolos fuera de ese set: flechas, guiones largos, el signo menos.
    Esto reemplaza lo que no entre por '?' en vez de cortar el programa, sin
    romper los acentos (que sí están en cp1252). Llamar al inicio de cada CLI.
    """
    import sys
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(errors="replace")
        except Exception:
            pass
