# Multimodal Emotion Recognition

A PyTorch-based system for recognizing emotions from **speech**, **text**, and both combined (multimodal), using the **TESS** (Toronto Emotional Speech Set) dataset.

## Will this run on another PC?

**Yes**, as long as that machine has:

| Requirement | Notes |
|-------------|--------|
| **Python 3.10–3.12** | Tested with 3.10+; avoid mixing system Python and venv |
| **Internet (first run)** | Downloads `bert-base-uncased` from Hugging Face |
| **TESS dataset** | Not in Git — download into `data/TESS/` (see [data/README.md](data/README.md)) |
| **GPU (optional)** | CUDA used if available; otherwise runs on CPU (slower) |

Paths are resolved from each script’s location (`__file__`), so the project works on **Windows, Linux, and macOS** without editing hardcoded drive letters.

## 📁 Project Structure
```
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py       # Bi-LSTM speech model training
│   │   └── test.py        # Evaluation + t-SNE
│   ├── text_pipeline/
│   │   ├── train.py       # Fine-tuned BERT training
│   │   └── test.py        # Evaluation + t-SNE
│   └── fusion_pipeline/
│       ├── train.py       # Cross-modal attention fusion training
│       └── test.py        # Evaluation + model comparison chart
├── Results/
│   ├── speech/            # Speech model outputs
│   ├── text/              # Text model outputs
│   └── fusion/            # Fusion model outputs (incl. comparison chart)
├── README.md
└── requirements.txt
```

## 🚀 Setup

### 1. Clone and create a virtual environment (recommended)
```bash
git clone <your-repo-url>
cd project
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download TESS dataset
- Go to [Kaggle TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- Download and extract into: `project/data/TESS/`

Expected structure:
```
data/TESS/
├── OAF_angry/
├── OAF_disgust/
├── YAF_happy/
└── ...
```

## 🏃 Running the Pipelines

### Speech-Only
```bash
cd models/speech_pipeline
python train.py   # Train Bi-LSTM model
python test.py    # Evaluate + generate t-SNE
```

### Text-Only
```bash
cd models/text_pipeline
python train.py   # Fine-tune BERT
python test.py    # Evaluate + generate t-SNE
```

### Multimodal Fusion
```bash
cd models/fusion_pipeline
python train.py   # Train cross-modal attention fusion
python test.py    # Evaluate + generate comparison chart
```

## 🏗️ Architecture Decisions

| Block | Speech | Text | Reasoning |
|---|---|---|---|
| Preprocessing | librosa (resample, trim, normalize) | Regex clean + BERT tokenize | Standard for each modality |
| Feature Extraction | MFCCs + Δ + ΔΔ (T × 120) | BERT last-hidden-state (T × 768) | MFCCs capture phonetic features; BERT captures semantic context |
| Temporal/Contextual Model | Bi-LSTM + self-attention | Fine-tuned BERT (last 4 layers) | Bi-LSTM handles temporal emotion patterns; BERT handles bidirectional context |
| Fusion | Cross-Modal Attention | — | Allows each modality to attend to the other, learning complementary cues |
| Classifier | 2-layer FC + Softmax | 2-layer FC + Softmax | Standard classification head |

## 📊 Emotions
`angry`, `disgust`, `fear`, `happy`, `neutral`, `ps` (pleasant surprise), `sad`

## 📈 Results
After running all pipelines, results are saved in `Results/`:
- Confusion matrices
- Training curves
- t-SNE cluster plots (Temporal / Contextual / Fusion representations)
- `fusion/model_comparison.png` — side-by-side accuracy comparison

## 🔗 Dataset
[TESS on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## Uploading to GitHub

From the `project` folder (where this README lives):

```bash
git init
git add .
git status   # confirm data/TESS/ and *.pt are NOT listed
git commit -m "Initial commit: multimodal emotion recognition pipelines"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### What to commit vs ignore

| Include in Git | Exclude (`.gitignore`) |
|----------------|-------------------------|
| `models/**/*.py` | `data/TESS/` (audio files) |
| `README.md`, `requirements.txt` | `.venv/` |
| `Results/*.json`, `*.txt`, `*.csv`, `*.png` | `Results/**/best_model.pt` (~400MB text model) |
| `data/README.md` | `.env`, `kaggle.json` |

**After cloning on a new PC:** `pip install -r requirements.txt` → download TESS → run `train.py` for each pipeline (checkpoints are not stored in the repo).

### Reproducing results without uploaded weights

```bash
cd models/speech_pipeline && python train.py && python test.py
cd ../text_pipeline && python train.py && python test.py
cd ../fusion_pipeline && python train.py && python test.py
```
