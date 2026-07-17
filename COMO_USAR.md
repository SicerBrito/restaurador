# Cómo usar el Restaurador de fotos

Guía completa para instalar y usar la app en cualquier equipo (Windows, Mac o Linux).
Quita texto superpuesto de una foto y reconstruye lo que había debajo.

---

## 0. La regla de oro

Todo se ejecuta desde la carpeta que **contiene** a `restaurador`, **nunca** desde
adentro. Si la carpeta está en:

```
C:\Users\TuUsuario\Desktop\WORKSPACE\Desarrollo\restaurador
```

entonces te parás en `...\Desarrollo` y usás `python -m restaurador.algo`.

> **Por qué:** `restaurador` es un *paquete* de Python. El `-m restaurador.app`
> (con punto) significa "buscá el paquete `restaurador` en esta carpeta y corré su
> módulo `app`". Por eso funciona desde afuera y no desde adentro.

En Windows (cmd), para ubicarte:
```cmd
cd C:\Users\TuUsuario\Desktop\WORKSPACE\Desarrollo
```

---

## 1. Requisitos

- **Python 3.9 o más nuevo** (probado con 3.12). Bajalo de python.org.
  Al instalarlo en Windows, marcá **"Add Python to PATH"**.
- Comprobá que está:
  ```cmd
  python --version
  ```

---

## 2. Instalación (una sola vez por equipo)

Parado en la carpeta de arriba (`...\Desarrollo`), en orden:

```cmd
pip install -r restaurador/requirements.txt
pip install --no-deps simple-lama-inpainting==0.1.2
```

**El `--no-deps` de la segunda línea es a propósito** (no lo saques): esa librería
pide una versión vieja de Pillow que choca con las demás, pero funciona bien con la
actual. El `--no-deps` evita que la baje.

### Opcional — textura por IA generativa (Stable Diffusion)

Solo si querés los backends `sd_inpaint` / `lama_sd_refine` (más textura, pero
lentos en CPU):

```cmd
pip install diffusers transformers accelerate lpips
```

La **primera vez** que uses cada modelo se descargan solos (LaMa ~200 MB, SD ~4 GB,
EasyOCR ~100 MB) y quedan en caché. Después funcionan sin internet. No hace falta
cuenta de HuggingFace.

---

## 3. Cómo ejecutar

Siempre parado en `...\Desarrollo`:

### La ventana (lo más fácil — empezá por acá)
```cmd
python -m restaurador.app
```

### Una foto por línea de comandos
```cmd
python -m restaurador.cli mifoto.jpg --auto -o salida/
python -m restaurador.cli mifoto.jpg --auto --backend lama_sd_refine -o salida/
python -m restaurador.cli mifoto.jpg --mascara mimascara.png -o salida/
python -m restaurador.cli mifoto.jpg --auto --sensibilidad alta -o salida/
python -m restaurador.cli mifoto.jpg --auto --solo-mascara -o salida/   # solo la máscara, no restaura
```

### Ver qué texto detecta (si algo no anda)
```cmd
python -m restaurador.diagnostico mifoto.jpg
```

### Comparar/medir modelos con tus fotos
```cmd
python -m restaurador.pruebas fotos_limpias/ --backends lama,sd_inpaint,lama_sd_refine --guardar salida/
```
(`fotos_limpias/` = fotos **sin** texto encima; el arnés les tapa una zona conocida
y mide qué tan bien la reconstruye cada modelo.)

**Regla mental:** parado en `Desarrollo`, todo es `python -m restaurador.LOQUESEA`
(`app`, `cli`, `diagnostico`, `pruebas`).

---

## 4. Usar la ventana, paso a paso

1. **Abrir imagen…**
2. **Detectar texto** — marca el texto en rojo automáticamente. Al lado del modo
   (*trazos/caja*) hay un control de **sensibilidad** (baja/media/alta): subilo si
   el texto es tenue o chico y no lo detecta; bajalo si marca de más.
3. Corregí con el **pincel** si hace falta: *Agregar* pinta zona a reconstruir,
   *Borrar* la quita. (Zoom en el panel derecho; *Deshacer* en la barra.)
4. Elegí el modelo en **"IA:"** (ver abajo cuál conviene).
5. **Restaurar**.
6. Compará con los radios **Algoritmo / IA / Combinado** y guardá el que mejor quede.

> **Guardar solo la máscara:** el botón **"Guardar máscara…"** (panel derecho) exporta
> únicamente la zona marcada (blanco = reconstruir), sin restaurar. Sirve para reusarla
> con `--mascara` en la línea de comandos o en otra herramienta.

> **La máscara importa más que el modelo.** En las pruebas, una máscara floja daba
> ~13 dB y una bien puesta ~23 dB: una diferencia enorme. Tomate un minuto con el
> pincel; cubrí la sombra y el borde de las letras.

---

## 5. Qué modelo de IA elegir

| Backend | Qué hace | Cuándo usarlo | Velocidad (CPU) |
|---|---|---|---|
| **lama** (defecto) | Reconstrucción limpia, tiende a alisar la textura. | Siempre, y para previsualizar rápido. | ~2 s |
| **sd_inpaint** | Genera textura real desde cero. Inventa más. | Fondos con textura (pared, césped, tela). | ~2-3 min |
| **lama_sd_refine** | LaMa + textura encima. Menos invención que SD puro. | Cuando LaMa quedó borroso pero SD inventa de más. | ~1-1½ min |

En una máquina **sin GPU potente**, dejá `lama` para trabajar y usá los de SD solo
para la pasada final de una foto importante.

---

## 6. Rendimiento

- Funciona en **CPU**. Andará en segundos con `lama`; en minutos con los backends SD.
- Usa **GPU NVIDIA** automáticamente **solo si** tenés una potente (4 GB+ de VRAM) con
  PyTorch CUDA instalado. Placas viejas o con menos de ~2 GB no sirven para IA y se
  usa la CPU igual.

---

## 7. Solución de problemas

**"No encontré fotos en fotos_limpias/"**
La carpeta no existe o está vacía. Creala y poné adentro fotos **sin** texto encima.

**Error al Restaurar: `WinError 1114 ... c10.dll` (fallo de DLL de torch)**
Ocurría al cargar torch desde el hilo de trabajo. Ya está mitigado: la app ahora
carga el motor de IA al arrancar ("Cargando el motor de IA…"). Si aún aparece:
1. Confirmá que torch carga solo, parado en `...\Desarrollo`:
   ```cmd
   python -c "import torch; print(torch.__version__)"
   ```
   - Si eso **también** falla → instalá el "Visual C++ Redistributable x64" de
     Microsoft y reintentá; si sigue, reinstalá torch:
     `pip install --force-reinstall torch torchvision`.
   - Si eso **funciona** pero la ventana no → cerrá y reabrí la app (el arreglo
     precarga torch al inicio).

**La detección con IA (EasyOCR) falla**
Marcá el texto a mano con el pincel: es el camino más confiable igual. `diagnostico`
te muestra qué vio cada detector.

**Descarga de un modelo da error 401 / "gated" / "Repository Not Found"**
Se cambió a modelos públicos (SD 1.5) que no piden cuenta. Si editaste los IDs de
modelo en `backends.py`, volvé a los que estaban.

---

## 8. Archivos del proyecto

```
restaurador/
├── app.py          Interfaz gráfica (ventana)
├── cli.py          Línea de comandos
├── motor.py        Orquesta: recorte, algoritmo clásico, composición, mezcla
├── backends.py     Modelos de IA intercambiables (lama, sd_inpaint, lama_sd_refine)
├── deteccion.py    Detección automática de texto
├── metricas.py     Métricas de calidad (textura, nitidez, LPIPS)
├── pruebas.py      Arnés para comparar modelos con tus fotos
├── diagnostico.py  Depurar la detección de texto
├── requirements.txt
├── LEEME.md        Explicación técnica de las decisiones de diseño
└── COMO_USAR.md    Este archivo
```
