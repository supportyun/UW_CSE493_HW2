## Report

Our results are summarized in `report.pdf`.

## How to Run the Code

All scripts are listed alphabetically below with a brief explanation and example command.

### `generate_data.py`

Generates modular arithmetic datasets (add, sub, div under mod 97 and 113).

```bash
python generate_data.py
```

---

### `inference.py`

Performs inference on a model trained with the "I love machine learning" sanity check task.

```bash
python inference.py
```

---

### `inference_problem2.py`

Performs inference for modular arithmetic problems (e.g., `12/25=`).

```bash
python inference_problem2.py
```

---

### `model.py`

Defines the GPT-style Transformer model and `GPTConfig`. Imported by training and inference scripts.

*No direct execution needed.*

---

### `muon.py`

Implements the Muon optimizer. Used by `train_muon.py`.

*No direct execution needed.*

---

### `plot_1.py`

Plots accuracy and loss curves for Sanity Check and Addition/Subtraction tasks using saved logs.

```bash
python plot_1.py
```

---

### `plot_2.py`

Plots accuracy curves (train/test) comparing optimizers (e.g., AdamW vs. Muon) for ablation studies.

```bash
python plot_2.py
```

---

### `train.py`

Trains a simple model to overfit the sentence `I love machine learning`.

```bash
python train.py
```

---

### `train_muon.py`

Trains a modular arithmetic Transformer using the Muon optimizer for hidden weights and AdamW for the rest.

```bash
python train_muon.py
```

---

### `train_problem2.py`

Trains a Transformer model for addition, subtraction, or division under mod p using the `GPT` model.

```bash
python train_problem2.py
```
