# Restaurador de fotos

Quita texto superpuesto de una imagen y reconstruye lo que había debajo.
Genera **tres versiones** para comparar: algoritmo clásico, IA, y una mezcla ajustable
de las dos con un deslizador en vivo.

Funciona sola: detecta el texto, te deja corregir a mano, restaura y guarda.
Todo local, sin enviar tus fotos a ningún servidor.

---

## Antes que nada: qué se puede y qué no

Si el texto es **opaco**, los píxeles de abajo **ya no existen** en el archivo. Ninguna
herramienta los recupera. Lo que hace esta app es **reconstruir algo plausible** a partir
del contexto. Con fondos simples (cielo, pared, ropa lisa, césped) queda muy convincente.
Con algo único e irrepetible tapado (una cara, un texto de fondo, un objeto específico)
va a inventar — coherente, pero inventado.

Traducido: sirve para recuperar **la foto**, no para revelar **un secreto** que estaba tapado.

---

## Instalación

Requiere Python 3.9+.

```bash
pip install -r requirements.txt
pip install --no-deps simple-lama-inpainting==0.1.2
```

> El `--no-deps` es a propósito: `simple-lama-inpainting` declara `Pillow<10`, un pin viejo
> e innecesario que choca con EasyOCR. Verificado que funciona bien con Pillow 12.

La primera vez que uses la IA se descargan los modelos solos (LaMa ~200 MB, EasyOCR ~100 MB).
Después ya quedan en caché y funciona sin internet.

Funciona en CPU. Si tenés GPU NVIDIA con PyTorch CUDA, la usa sola y vuela.

---

## Uso

### Con ventana (recomendado)

```bash
python3 -m restaurador.app
```

1. **Abrir imagen…**
2. **Detectar texto** — marca el texto en rojo automáticamente.
3. Corregí con el **pincel** si hace falta: *Agregar* pinta zona a reconstruir, *Borrar* la quita.
   (Zoom con los botones del panel derecho; *Deshacer* está en la barra.)
4. **Restaurar**.
5. Compará con los radios **Algoritmo / IA / Combinado** y guardá la que mejor quede.

### Por línea de comandos

```bash
# automático
python3 -m restaurador.cli foto.jpg --auto -o salida/

# con máscara propia (blanco = reconstruir)
python3 -m restaurador.cli foto.jpg --mascara mascara.png -o salida/

# máxima calidad (más lento)
python3 -m restaurador.cli foto.jpg --auto --preset maximo -o salida/
```

---

## Consejos que sí mueven la aguja

**1. La máscara importa más que el algoritmo.** Es el hallazgo más fuerte de las pruebas:
con una máscara que recortaba mal las letras el resultado daba ~13 dB; con la máscara bien
puesta, ~23 dB. Esa diferencia es enorme — mucho mayor que la que hay entre algoritmo e IA.
Vale la pena tomarse un minuto con el pincel.

**2. Cubrí la sombra y el borde del texto.** Si queda un halo gris, la máscara se quedó corta.
Agrandá un poco con el pincel, o usá el modo `caja`.

**3. Menos área es mejor.** Modo `trazos` (por defecto) marca solo las letras y conserva el
fondo real entre ellas. Modo `caja` borra el rectángulo completo: usalo solo si el texto trae
sombra fuerte o una banda semitransparente.

**4. Probá las tres.** No hay una que gane siempre — depende de la imagen (ver abajo).

---

## Por qué está hecho así

Cada decisión se midió tapando fotos reales a propósito y comparando la reconstrucción
con la verdad conocida.

### La corrección más importante

La primera versión se ajustó con **PSNR/SSIM**, y fue un error. Esas métricas **premian el
desenfoque**: acertarle al promedio de todas las texturas posibles penaliza menos que
arriesgar una textura nítida y errarle. Optimizando PSNR se llegaba a un **manchón liso**.

Ahora se mide con **LPIPS** (métrica perceptual, se acerca a lo que ve el ojo) y con la
**energía de textura** del relleno comparada con la de su entorno. Los criterios se
contradicen de forma brutal:

| config | LPIPS ↓ (el ojo) | textura | PSNR ↑ |
|---|---|---|---|
| LaMa **1 pasada** | **0.081** ✅ | **0.88** | 22.04 |
| LaMa **TTA8** (promedia 8 vistas) | 0.132 ❌ | 0.53 (mancha) | **23.13** 🏆 |

(textura del original = 1.04; 1.00 = misma textura que el entorno)

PSNR corona como campeón justo a la versión que se ve peor. Por eso el promediado se
eliminó, y ahora la IA hace **una sola pasada limpia**.

### El resto de las decisiones

| Decisión | Motivo medido |
|---|---|
| **Sin promediado (TTA)** | Promediar 4-8 reconstrucciones plausibles no da textura, da una mancha. LPIPS empeoraba 39%. |
| **Nada le gana a LaMa 1 pasada** | Se probaron mezclas 0.9 y 0.75, split de frecuencias y 2 pasadas de LaMa. Ninguna mejora. Por eso «combinado» es solo una mezcla que ajustás vos. |
| **Sin grano sintético ni realce** | Suben la métrica de textura pero **empeoran** LPIPS: ruido falso no es textura real. |
| **Contexto = 1.0** | Con PSNR parecía que 15% alcanzaba; con LPIPS es al revés (0.2411 vs 0.2557). Más contexto, mejor. |
| **FSR_FAST** como algoritmo clásico | Mejor que Telea/NS. `FSR_BEST` tarda ~30x más para ganar 0.07 dB. |
| **Nunca reescalar antes de la IA** | Reescalar a 768 px costaba −1.7 dB; a 512 px, −3.5 dB. |
| **Máscara por trazos** | Menos área que el rectángulo completo, conservando el fondo real entre letras. |
| **Detección: readtext + detect unidos** | `readtext()` filtra por confianza de *lectura* y descarta texto decorativo que sí se detectó. `detect()` no depende de que sea legible. |

Tres bugs que las pruebas encontraron y que están corregidos:

- **FSR y SHIFTMAP usan la máscara invertida** (255 = píxel *conocido*), al revés que
  `cv2.inpaint`. Ignorarlo produce basura en silencio.
- **LaMa rellena la imagen a múltiplos de 8 y devuelve ese tamaño**. Si no se recorta la
  salida, la foto cambia de tamaño y se desalinea. No se nota probando con 512×512.
- **Al componer con borde difuminado, el alfa bajaba dentro de la máscara** y los píxeles
  del texto se volvían a mezclar en los bordes.

Fuera de la máscara la imagen queda **intacta**, salvo ~6 px de transición en el borde.

### Presets

Ya no controlan el promediado (se eliminó), sino **cuánto contexto** recibe el modelo.

| Preset | Contexto | Para qué |
|---|---|---|
| `rapido` | 20% | Tantear si la máscara quedó bien |
| `equilibrado` | 100% | Por defecto |
| `maximo` | imagen completa | La pasada final |

---

## Backends de IA: LaMa vs difusión (nítido)

El desenfoque que puede quedar **no viene de la métrica** (eso ya se corrigió pasando
a LPIPS), sino del modelo: **LaMa es de regresión**, y en una zona ambigua devuelve el
*promedio* de todas las texturas plausibles → se aplana. Un modelo **generativo**
(difusión) *muestrea* una textura concreta en vez de promediar → sale nítido, a cambio
de inventar más y ser lento en CPU.

El modelo de IA ahora es **enchufable**. `--backend` (CLI) o el menú **IA:** (ventana):

| Backend | Qué hace | Costo |
|---|---|---|
| `lama` (defecto) | Una pasada de LaMa. Rápido, medido. | segundos |
| `sd_inpaint` | Stable Diffusion rellena de cero. Textura real, inventa más. | minutos en CPU |
| `lama_sd_refine` | LaMa pone la estructura, SD le agrega textura encima (denoise bajo). Menos invención. | minutos en CPU |

Los backends de difusión son **opcionales**: `pip install diffusers transformers accelerate`.
Corren por tiles de 512 px para no reescalar (reescalar degrada, ya medido). Sin GPU útil,
dejá `lama` para trabajar y usá difusión solo para la pasada final.

> Ojo: la difusión **no es magia**. Sigue reconstruyendo algo plausible, con más
> textura pero también con más libertad para inventar. La máscara sigue siendo la
> palanca #1.

### Medir antes de decidir

Igual que todo en este proyecto, cambiar de modelo se decide con números, no a ojo.
El arnés tapa fotos limpias tuyas a propósito (verdad conocida) y compara backends:

```bash
python -m restaurador.pruebas fotos_limpias/ --backends lama,sd_inpaint,lama_sd_refine --guardar salida/
```

Reporta **LPIPS** (perceptual, ↓ mejor), **textura** (→1.0 = igual al entorno) y
**nitidez**. LPIPS necesita `pip install lpips`; sin él, mide textura y nitidez igual.

## Archivos

```
restaurador/
├── app.py          Interfaz gráfica (Tkinter)
├── cli.py          Línea de comandos
├── motor.py        Orquesta: recorte, FSR, composición, mezcla
├── backends.py     Modelos de IA enchufables (lama, sd_inpaint, lama_sd_refine)
├── deteccion.py    Detección automática de texto
├── metricas.py     Métricas de calidad (textura, nitidez, LPIPS)
├── pruebas.py      Arnés: tapa fotos reales y compara backends
└── requirements.txt
```

`motor.restaurar(img_bgr, mask, preset=..., backend=..., progress=...)` devuelve
`{"algoritmo": ..., "ia": ..., "combinado": ...}` si querés usarlo desde tu propio código.
