# FineTunned CycleGAN

Fine-tuning de CycleGAN (horse2zebra → lion/cheetah) para traducción de imágenes
sin pares alineados, usando el dataset Wildlife Animals Images de Kaggle.

## Estructura

```
.
├── pyproject.toml                 # Dependencias y config del proyecto
├── uv.lock                        # Lockfile reproducible
├── .env.example                   # Template de variables de entorno
│
├── src/                           # Módulos Python (responsabilidad única)
│   ├── model.py                   # ResNetGenerator, PatchGANDiscriminator
│   ├── dataset.py                 # UnpairedDataset + get_dataloaders
│   ├── training.py                # train_cyclegan (loop + evaluación)
│   ├── inference.py               # CycleGANInference (carga/traducción)
│   ├── registry.py                # ModelRegistry (tracking de experimentos)
│   ├── evaluation.py              # FID / LPIPS / visual grids
│   ├── download_and_prepare.py    # Descarga y preparación del dataset
│   └── __init__.py
│
├── scripts/
│   └── train.py                   # Entrenamiento por CLI (background-ready)
│
├── fine_tuning_experiment.ipynb   # Notebook interactivo
│
├── checkpoints/
│   ├── horse2zebra/               # Checkpoints pre-entrenados
│   └── experiments/               # ModelRegistry (runs de fine-tuning)
├── data/
│   ├── train/{lion,cheetah}/      # Training set 256×256
│   └── test/{lion,cheetah}/       # Test set 256×256
└── results/                       # Grids de traducción generados
```

## Requisitos

- Python ≥ 3.12
- CUDA 12.1 (GPU con ≥ 4 GB VRAM recomendada)
- [uv](https://docs.astral.sh/uv/) como gestor de proyectos

```bash
uv sync
```

## Entrenamiento

> ⚠️ **Siempre usar `python -m scripts.train`**, no `python scripts/train.py`.
> El flag `-m` hace que Python agregue el directorio raíz al `sys.path`,
> permitiendo los imports `from src.xxx` sin necesidad de hacks ni variables
> de entorno. Funciona desde cualquier subdirectorio del proyecto.

### Opción 1 — Notebook interactivo

```bash
uv run jupyter lab --no-browser --port=8888
```

Abrir `fine_tuning_experiment.ipynb` y ejecutar las celdas en orden:
preparación del dataset → carga de checkpoints → entrenamiento →
evaluación y selección del mejor modelo.

### Opción 2 — Script CLI (recomendado para servidores/cluster)

```bash
# Preparar dataset + entrenar en background (redirigir a archivo de log)
mkdir -p logs
nohup uv run python -m scripts.train --prepare > logs/train_20260725.log 2>&1 &

# Dataset ya preparado, solo entrenar
nohup uv run python -m scripts.train > logs/train_20260725.log 2>&1 &

# Personalizar hiperparámetros
nohup uv run python -m scripts.train \
    --epochs 30 --lr 1e-4 --checkpoint-interval 5 \
    > logs/train_20260725.log 2>&1 &

# Uso interactivo (con barra de progreso tqdm)
uv run python -m scripts.train --progress
```

### Parámetros de `scripts/train.py`

| Flag | Default | Descripción |
|------|---------|-------------|
| `--prepare` | off | Ejecutar `prepare_dataset()` (descarga Kaggle + filtro + split + resize) antes de entrenar |
| | | |
| **Dataset** | | |
| `--data-dir` | `data/train` | Directorio con subcarpetas `{lion,cheetah}/` con imágenes 256×256 |
| `--test-dir-a` | `data/test/lion` | Imágenes de test para dominio A (león); usadas en evaluación post-entrenamiento |
| `--test-dir-b` | `data/test/cheetah` | Imágenes de test para dominio B (guepardo); usadas en evaluación post-entrenamiento |
| | | |
| **Hiperparámetros** | | |
| `--epochs` | `15` | Número total de epochs de entrenamiento |
| `--lr` | `2e-4` | Learning rate inicial para el optimizador Adam |
| `--betas` | `0.5 0.999` | Coeficientes beta1 y beta2 de Adam (dos valores separados por espacio) |
| `--lambda-cycle` | `10.0` | Peso de la loss de cycle-consistency (cuanto más alto, más se fuerza que A→B→A sea fiel) |
| `--lambda-identity` | `0.5` | Peso de la identity loss (ayuda a preservar colores/gamma del dominio original) |
| `--pool-size` | `50` | Tamaño del búfer de imágenes falsas para el discriminador (reduce oscilación) |
| `--batch-size` | `1` | Tamaño del batch (CycleGAN entrena con batch=1 por diseño) |
| `--img-size` | `256` | Tamaño final del recorte (crop) para las imágenes de entrenamiento |
| `--load-size` | `286` | Tamaño al que se redimensionan las imágenes antes del random crop (data augmentation) |
| `--checkpoint-interval` | `10` | Cada N epochs se guarda un checkpoint en el ModelRegistry |
| `--decay-epochs` | `5` | Número de epochs con LR constante antes de que empiece el decaimiento lineal |
| `--approach` | `frozen-encoder` | Modo de fine-tuning: `frozen-encoder` congela el encoder pre-entrenado y solo entrena decoder + discriminadores; `full` entrena todos los pesos |
| | | |
| **Rutas** | | |
| `--registry-dir` | `checkpoints/experiments` | Directorio donde se guardan los experimentos (run_id + checkpoints + metadatos) |
| `--base-checkpoint-dir` | `checkpoints/horse2zebra` | Directorio con los checkpoints pre-entrenados `gen_AB.pth` y `gen_BA.pth` |
| | | |
| **Runtime** | | |
| `--gpu` | `auto` | Selección de dispositivo: `auto` usa heurística (>1 GiB libre en GPU), `cuda` fuerza GPU, `cpu` fuerza CPU. Para elegir una GPU concreta usá `CUDA_VISIBLE_DEVICES=N` |
| `--progress` | off | Mostrar barra de progreso tqdm por epoch (útil en terminal interactiva; desactivado por defecto para no ensuciar logs de nohup) |

> Todos los flags: `uv run python -m scripts.train --help`

### Selección de GPU en servidores con múltiples GPUs

En un clúster con varias GPUs, `--gpu auto` elige la que tenga ≥1 GiB libre,
pero podés forzar una GPU específica con `CUDA_VISIBLE_DEVICES`:

```bash
# Ver GPUs disponibles
nvidia-smi

# Seleccionar GPU 1 y entrenar
export CUDA_VISIBLE_DEVICES=1
uv run python -m scripts.train

# O en una línea (sin export persistente)
CUDA_VISIBLE_DEVICES=2 nohup uv run python -m scripts.train --prepare \
    > logs/train_20260725.log 2>&1 &
```

> `CUDA_VISIBLE_DEVICES` funciona a nivel de proceso — cada script ve solo la GPU
> asignada, sin afectar otros procesos del usuario.

## Evaluación

Las métricas FID y LPIPS se calculan automáticamente al finalizar cada
entrenamiento y se almacenan en el registro del experimento (`ModelRegistry`).
Para inspeccionar resultados históricos y cargar el mejor modelo, usa el
notebook (`fine_tuning_experiment.ipynb`, sección 6.3).
