# 🌐 Multilingual Neural Machine Translation

### Arabic ↔ English ↔ French — Transformer from Scratch

A many-to-many neural machine translation system built entirely from scratch in PyTorch.
Supports all 6 directions between Arabic, English, and French using a single shared model.

---

## 📁 Project Structure

```
Multilingual_Translation/
│
├── src/
│   ├── app/
│   │   └── app.py                        # Gradio inference app (local demo)
│   │
│   └── Model_assets/
│       ├── checkpoints/
│       │   └── best_model.pt             # Best model checkpoint (lowest val loss)
│       ├── tokenizer/
│       │   ├── translation_spm.model     # Trained SentencePiece BPE model
│       │   └── translation_spm.vocab     # Vocabulary file
│       └── artifacts/
│           ├── model_config.json         # Architecture hyperparameters
│           ├── training_history.json     # Loss & BLEU per epoch
│           └── run_summary.json          # Full run metadata
│
└── Multilingual_Translation_notebook.ipynb   # Training notebook (Google Colab)
```

---

## 🏗️ Model Architecture

A standard encoder-decoder Transformer built on `torch.nn.Transformer` with tied input/output embeddings and sinusoidal positional encoding.

| Hyperparameter                  | Value                         |
| ------------------------------- | ----------------------------- |
| Model dimension (`d_model`)     | 256                           |
| Attention heads (`n_heads`)     | 4                             |
| Encoder / Decoder layers        | 4 / 4                         |
| Feed-forward dimension (`d_ff`) | 1024                          |
| Dropout                         | 0.1                           |
| Max sequence length             | 64 tokens                     |
| Shared vocabulary size          | 32,000 BPE subwords           |
| Weight tying                    | Embedding ↔ Output projection |
| Weight init                     | Xavier uniform                |

**Encoder input format:** `[<2tgt>]  token₁  token₂ … tokenₙ  [<eos>]`
The target language token is prepended to the source — this is how the model knows which language to produce.

---

## 🗃️ Data

**Source:** [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) via HuggingFace Datasets, streamed to avoid full downloads.

| Direction | Pairs (max) |
| --------- | ----------- |
| ar → en   | 15,000      |
| en → ar   | 7,500       |
| ar → fr   | 15,000      |
| fr → ar   | 7,500       |
| en → fr   | 15,000      |
| fr → en   | 7,500       |

**Train / Val / Test split:** 80% / 10% / 10% — stratified by language pair.

---

## 🧹 Preprocessing Pipeline

Data goes through two cleaning phases before and after tokenizer training:

**Phase 1 — Light Preprocessing** _(before tokenizer training)_

- Unicode NFC normalisation
- Control / non-printable character removal
- URL and e-mail removal
- HTML / XML tag stripping
- Whitespace collapse
- Language-aware quote normalisation (FR: `«»` → `"` / EN: smart quotes → `'`)
- Repeated character collapse (`"loooool"` → `"lool"`)

**Phase 2 — Task-specific Cleaning** _(after tokenizer training, before dataset build)_

- **Arabic:** strip tashkeel diacritics, remove kashida, normalise Alef/Hamza variants
- **English:** remove corpus section markers (`Chapter 1`, `§ 42`)
- **French:** remove bracketed translator notes `(voir note 3)`, `[NB: …]`

**Pair filtering** rejects pairs that are too short (`< 3` tokens), too long (`> 62` tokens), have extreme length ratio (`> 3.0`), are identical src/tgt, or contain only numbers/punctuation.

---

## 🔤 Tokenizer

A shared SentencePiece BPE tokenizer trained across all three languages.

| Setting            | Value                                              |
| ------------------ | -------------------------------------------------- |
| Type               | BPE                                                |
| Vocabulary size    | 32,000                                             |
| Character coverage | 99.95%                                             |
| Special tokens     | `<pad>` (0), `<bos>` (1), `<eos>` (2), `<unk>` (3) |
| Language tokens    | `<2ar>` (4), `<2en>` (5), `<2fr>` (6)              |

---

## 🏋️ Training

| Setting                    | Value                                          |
| -------------------------- | ---------------------------------------------- |
| Epochs                     | 15 (with early stopping)                       |
| Batch size                 | 32                                             |
| Optimizer                  | Adam with warmup schedule                      |
| Warmup steps               | 4,000                                          |
| Loss                       | Label-smoothed cross-entropy (smoothing = 0.1) |
| Gradient clipping          | max norm = 1.0                                 |
| Mixed precision            | AMP (fp16 on GPU)                              |
| Early stopping patience    | 4 epochs                                       |
| Evaluation metric          | sacreBLEU                                      |
| Decoding (training eval)   | Greedy (fast, batch-parallel)                  |
| Decoding (final test eval) | Beam search (beam = 4, length penalty = 0.6)   |

---

## 🚀 Running the Gradio App Locally

### 1. Install dependencies

```bash
pip install "gradio==4.44.1" "fastapi==0.112.0" "starlette==0.37.2" \
            "huggingface_hub==0.23.4" torch sentencepiece
```

### 2. Set paths in `app.py`

Open `src/app/app.py` and edit the USER CONFIGURATION block at the top:

```python
SPM_MODEL_PATH  = "path/to/tokenizer/translation_spm.model"
CHECKPOINT_PATH = "path/to/checkpoints/best_model.pt"
```

### 3. Run

```bash
cd src/app
py app.py
```

Then open **http://127.0.0.1:7860** in your browser.

---

## 🔁 Training From Scratch (Google Colab)

1. Open the notebook in Colab (requires GPU — T4 recommended)
2. Run all cells top to bottom
3. Adjust runtime options at the bottom of the notebook:

```python
MAX_SAMPLES_PER_PAIR = 15000   # pairs per language direction
RETRAIN_TOKENIZER    = False   # set True to retrain tokenizer from scratch
FORCE_REDOWNLOAD     = False   # set True to ignore cached data
RESUME_CHECKPOINT    = None    # e.g. Path("checkpoints/epoch_03.pt")
```

4. After training, download the output files:
   - `checkpoints/best_model.pt`
   - `tokenizer/translation_spm.model`
   - `tokenizer/translation_spm.vocab`
   - `artifacts/model_config.json`
   - `artifacts/training_history.json`
   - `artifacts/run_summary.json`

---

## 📦 Dependencies

| Package                   | Purpose                    |
| ------------------------- | -------------------------- |
| `torch`                   | Model, training, inference |
| `sentencepiece`           | BPE tokenizer              |
| `datasets`                | HuggingFace data streaming |
| `sacrebleu`               | BLEU evaluation            |
| `gradio==4.44.1`          | Local demo UI              |
| `fastapi==0.112.0`        | Gradio backend             |
| `huggingface_hub==0.23.4` | HuggingFace integration    |

---

## 👤 Author

**Beshoy** — built as an end-to-end NLP project covering data acquisition, preprocessing, tokenizer training, model architecture, training loop, evaluation, and local deployment.
