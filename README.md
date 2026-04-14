# Clairvoyance

Transfer-learning classifier for **trading card condition** labels. A frozen **VGG16** backbone feeds a small dense head; images are organized as **one subdirectory per class** (e.g. `HP`, `NM`, `LP+`).

## Requirements

- Python 3 (this repo pins **TensorFlow 2.6** in `requirements.txt`; use a Python version compatible with that release, e.g. 3.7–3.9).
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout

Training expects a **root folder** whose **immediate subfolders** are class names, each containing images:

```text
training_data/
  HP/
    sku_zz__foo-back-deskew.jpg
  NM/
    ...
```

The bundled CSV `INVENTORY-TENSORFLOW_DATA - 5.0.csv` provides columns **`Woo image back`** (image URL), **`Condition`** (label), and optional **`SKU`** (used to make downloaded filenames unique).

## Train (`train.py`)

**Download from CSV and train** (writes under `training_data/` by default):

```bash
python train.py --download --epochs 5
```

**Train only** from images already on disk:

```bash
python train.py --img-path ./training_data
```

If you omit `--img-path` and do **not** pass `--download`, training uses `./training_data` when that directory exists; otherwise the script exits with an error.

### Training options

| Option | Default | Description |
|--------|---------|-------------|
| `--download` | off | Download URLs from CSV into `--download-path` |
| `--csv-path` | bundled inventory CSV | Path to CSV (`Woo image back`, `Condition`; optional `SKU`) |
| `--download-path` | `training_data` | Where downloaded class folders are stored |
| `--img-path` | *(see above)* | Root directory with one subfolder per class |
| `--test-size` | `0.1` | Fraction of data used for validation |
| `--batch-size` | `32` | Batch size |
| `--epochs` | `1` | Training epochs |
| `--model-out` | `my_model.h5` | Saved Keras model path |
| `--labels-out` | `label.txt` | Class names in **output index order** (one per line) |
| `--gpu` | off | Use GPU; **default is CPU-only** |

### Training outputs

- **`my_model.h5`** — saved model  
- **`label.txt`** — class order; **required for correct inference** with `test.py`  
- **`loss.png`** — train vs validation loss curve  

Training and inference both use **`tf.keras.applications.vgg16.preprocess_input`**.

## Predict (`test.py`)

```bash
python test.py --image path/to/card_back.jpg --model my_model.h5
```

By default, **`label.txt`** is loaded from the **same directory as** `--model`. Override with `--labels` if it lives elsewhere.

### Predict options

| Option | Default | Description |
|--------|---------|-------------|
| `--image` | *(required)* | Image file to classify |
| `--model` | `my_model.h5` | Trained `.h5` model |
| `--labels` | next to model | Path to `label.txt` from training |

The predicted label is printed to stdout; the script exits non-zero if the model or labels file is missing.

## Troubleshooting

- **Validation split errors**: `image_dataset_from_directory` needs enough images per class for the requested `--test-size`. If a class has very few files, lower `--test-size` or add images.
- **Failed downloads**: URLs in the CSV may be dead or rate-limited; failed rows are skipped and logged. Re-run `--download` to retry; existing files are skipped.
- **Wrong labels at inference**: Output indices map to **lines in `label.txt`**, not alphabetical guesswork. Keep `label.txt` paired with the model you load.