# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`restaurador` is a local photo tool that removes overlaid text from an image and reconstructs what was underneath. It produces **three versions** to compare — classic algorithm (FSR), AI (LaMa), and an adjustable blend — with a live slider in the GUI. Everything runs locally; images never leave the machine. Code, comments, docstrings, and UI strings are in Spanish — match that language when editing.

The whole codebase is one Python package (this directory *is* `restaurador`). Run commands from the **parent** directory so `restaurador` resolves as a package.

## Commands

```bash
# Install (note the two-step install — see below)
pip install -r requirements.txt
pip install --no-deps simple-lama-inpainting==0.1.2

# GUI (from the parent directory of this folder)
python3 -m restaurador.app

# CLI — auto-detect text, generate all 3 versions
python3 -m restaurador.cli foto.jpg --auto -o salida/
python3 -m restaurador.cli foto.jpg --mascara mascara.png -o salida/   # own mask (white = reconstruct)
python3 -m restaurador.cli foto.jpg --auto --preset maximo -o salida/  # max context, slower
python3 -m restaurador.cli foto.jpg --auto --backend lama_sd_refine -o salida/  # diffusion texture (needs diffusers)

# Diagnose why text detection did/didn't fire (writes *_diagnostico.png with boxes drawn)
python3 -m restaurador.diagnostico foto.jpg

# Measurement harness: cover clean photos, compare backends against known truth
python3 -m restaurador.pruebas fotos_limpias/ --backends lama,sd_inpaint,lama_sd_refine --guardar salida_pruebas/
python metricas.py   # numpy-only self-test of the texture/sharpness metrics
```

There is no unit-test framework or build step. Two runnable checks exist: `metricas.py`'s `__main__` self-test (numpy-only, no models needed) and the `pruebas.py` harness (needs the models). `diagnostico.py` debugs the detection stage specifically.

The diffusion backends and LPIPS are **optional installs** (see `requirements.txt` comments): `diffusers`/`transformers`/`accelerate` for `sd_inpaint`/`lama_sd_refine`, and `lpips` for the perceptual metric. On a machine without a usable GPU, keep the default `lama` backend for interactive use — diffusion is minutes/image on CPU.

### The `--no-deps` install is deliberate

`simple-lama-inpainting` pins `Pillow<10`, an old and unnecessary constraint that collides with EasyOCR. It is verified to work with Pillow 12, so it's installed separately with `--no-deps`. Don't "fix" this by relaxing the pin or merging it into `requirements.txt`.

Models download on first AI use (LaMa ~200 MB, EasyOCR ~100 MB), then cache and run offline. Works on CPU; uses an NVIDIA GPU automatically if PyTorch CUDA is present.

## Architecture

Modules, one clear dependency direction:

- **`motor.py`** — the reconstruction engine. `restaurar(img_bgr, mask, preset=, backend=, progress=)` returns `{"algoritmo", "ia", "combinado"}`. Owns cropping (`_caja_recorte`), the classic FSR fill, mask-restricted compositing (`_componer`), and the linear blend. Delegates the AI fill to `backends`. No UI/IO deps; reusable core.
- **`backends.py`** — pluggable AI fill backends behind a registry (`@backend("name")` / `backends.rellenar(...)`): `lama` (default, measured), `sd_inpaint`, `lama_sd_refine`. A backend is `fn(img_crop, mask_crop, progress=, **kw) -> filled_crop`. Heavy deps (torch/diffusers/simple-lama) are **lazy-imported inside each backend** — importing the module needs only numpy+cv2. `procesar_por_tiles` runs a 512px model over an arbitrary-size crop without rescaling.
- **`deteccion.py`** — automatic text detection → returns `(mask_uint8, boxes)`. No dependency on `motor`.
- **`metricas.py`** — quality metrics for the harness. Texture/sharpness are **numpy-only** (run without cv2/torch); LPIPS is lazy/optional. `comparar(original, reconstruida, mask)` → dict. Never PSNR/SSIM (see below). Has a `__main__` self-test.
- **`pruebas.py`** — measurement harness: covers real clean photos with synthetic text (known ground truth), restores with each backend, and reports LPIPS + texture + sharpness. This is how backend changes get decided — `python -m restaurador.pruebas fotos/ --backends lama,sd_inpaint,lama_sd_refine`.
- **`app.py`** — Tkinter GUI. Owns image/mask I/O (`leer_imagen` / `guardar_imagen`, which the CLI and harness reuse) and orchestrates detect → paint → restore → compare → save. Has an "IA:" backend dropdown.
- **`cli.py`** — batch/headless entry point over the same `motor` + `deteccion`; exposes `--backend`.

### Pluggable AI backend

The AI model is swappable so LaMa vs diffusion can be A/B-tested with the same harness, honoring the project's measure-everything DNA. Default is `lama` (`motor.BACKEND_IA`) — the fast, measured path; existing behavior is unchanged unless a backend is passed. **LaMa is a regression model: in ambiguous regions it returns the mean of all plausible textures, which flattens texture** — this is the same mean-regression that PSNR rewards, now internal to the model. `sd_inpaint`/`lama_sd_refine` *sample* a concrete texture instead (sharper, but invent more) at the cost of being slow on CPU and needing `diffusers`. Diffusion runs **tiled at 512px** to keep the never-rescale rule. SD's mask convention (255 = fill) matches the project's — no inversion (unlike FSR). A backend returns a full crop; `motor._componer` is what restricts changes to mask pixels.

Data conventions that thread through everything:
- Images are **BGR uint8** (OpenCV convention) end to end. `leer_imagen` converts from PIL RGB and applies EXIF rotation; `guardar_imagen` converts back.
- Masks are **uint8, 255 = reconstruct**. `motor.normalizar_mascara` binarizes and resizes any incoming mask to this convention.

### GUI threading model

Heavy work (detection, restoration) runs on daemon threads. Results come back to the Tk main loop through `self.cola` (a `queue.Queue`), drained by `_procesar_cola` on an 80 ms `after` poll. Never touch Tk widgets from a worker thread — push a `(tipo, dato)` tuple onto the queue instead. `self._trabajando` guards against overlapping jobs.

### Why the engine is built the way it is (do not "optimize" these away)

`motor.py` and `LEEME.md` document decisions that were measured against ground truth by covering real photos on purpose. The counter-intuitive ones bite anyone who reverts them:

- **Tuned on LPIPS + texture energy, not PSNR/SSIM.** PSNR rewards blur — hitting the average of all plausible textures beats risking a sharp one and missing. Optimizing PSNR produced a smooth smear. If you add a quality metric, do **not** reach for PSNR/SSIM.
- **AI is a single clean LaMa pass. No TTA averaging.** Averaging several plausible reconstructions makes a smudge, not texture (raised PSNR ~1 dB, worsened LPIPS 39%). The `combinado` blend is a user-adjustable knob (default 90% AI), not a quality win over pure AI.
- **Never rescale before LaMa** — always native resolution (rescaling cost −1.7 dB at 768 px). Presets control *context padding* (`PRESETS_CONTEXTO`), not TTA. `PRESETS_TTA` is a back-compat alias.
- **No synthetic grain / unsharp** — improves the texture metric but worsens LPIPS.

### Non-obvious correctness traps already fixed here

- **FSR (`cv2.xphoto.inpaint`) and SHIFTMAP take the mask INVERTED** (255 = *known* pixel), opposite to `cv2.inpaint`. Getting this wrong produces silent garbage. See `_rellenar_algoritmo`.
- **LaMa pads the image to multiples of 8 and returns that larger size** — the output must be cropped back (`out[:h, :w]`) or the photo shifts and misaligns.
- **Feathered compositing must not lower alpha *inside* the mask**, or original text pixels bleed back at the edges. `_componer` dilates the mask before blurring and clamps with `np.maximum(alpha, dentro)`.
- Outside the mask the image stays byte-identical to the original except a ~6 px transition edge — compositing only touches the marked region.

### Detection details

`deteccion.detectar_texto(img, modo=, motor=, dilatar=, sensibilidad=)` tries EasyOCR (CRAFT) and falls back to a morphological detector if EasyOCR is absent or throws — but it records the failure in `ultimo_motor` / `ultimo_error` and surfaces it (an earlier version swallowed the error, leaving users with a bare "no text found"). EasyOCR path unions `readtext()` (detect + read, filtered by reading confidence) with `detect()` (detector only) to catch decorative text that reads poorly; `sensibilidad` (0..1) scales the CRAFT thresholds. `"trazos"` mode masks only letter pixels (better, less area); `"caja"` masks the full rectangle (for strong shadows / semi-transparent bands).

Two levers set detection quality: which boxes (the detector) and which pixels inside them (`_mascara_trazos`). The pixel cut is higher-leverage (the mask is the #1 quality lever). `_mascara_trazos` picks text polarity via `_polaridad_texto` (text = the Otsu class whose intensity is farthest from the box-border background) — robust to bold/large text where the old "minority class = text" heuristic failed — and refines large boxes with an adaptive threshold for non-uniform backgrounds. A higher-precision detector (PaddleOCR/DocTR) is a candidate future backend but not yet added. The GUI can export the mask alone (`guardar_mascara`); the CLI has `--solo-mascara` and `--sensibilidad`.
