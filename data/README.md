# Dataset folder

This project uses the **TESS** (Toronto Emotional Speech Set) dataset.

## Download (required on every new machine)

1. Go to [TESS on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
2. Download and extract the archive
3. Copy the emotion folders so the layout is:

```text
data/TESS/
├── OAF_angry/
├── OAF_disgust/
├── YAF_happy/
└── ... (other speaker/emotion folders)
```

Each folder should contain `.wav` files named like `OAF_back_angry.wav`.

## Do not commit audio to Git

The `data/TESS/` folder is listed in `.gitignore` because:

- It is large (~500+ MB)
- Kaggle terms apply — link to the dataset in your README instead

After cloning the repo, download TESS locally before running `train.py`.
