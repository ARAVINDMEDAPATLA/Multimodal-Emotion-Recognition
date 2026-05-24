"""
Speech-Only Emotion Recognition Pipeline
Architecture:
  Preprocessing   -> librosa (resample, trim silence, normalize)
  Feature Extract -> MFCCs (40 coeffs) + Delta + Delta-Delta  => (time_steps x 120)
  Temporal Model  -> Bidirectional LSTM
  Classifier      -> FC + Softmax
Dataset: TESS (Toronto Emotional Speech Set)
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Project root = 2 levels up from this file (project/models/speech_pipeline/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "data_dir": os.path.join(PROJECT_ROOT, "data", "TESS"),
    "sample_rate": 22050,
    "max_len": 3,                     # seconds — pad/trim to this length
    "n_mfcc": 40,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "save_dir": os.path.join(PROJECT_ROOT, "Results", "speech"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# 1. PREPROCESSING + FEATURE EXTRACTION
# ─────────────────────────────────────────────
def load_tess_metadata(data_dir):
    """Walk TESS folder and build a DataFrame with (filepath, emotion)."""
    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                # TESS filename format: OAF_back_angry.wav
                emotion = f.split("_")[-1].replace(".wav", "").lower()
                if emotion in EMOTIONS:
                    records.append({"path": os.path.join(root, f), "emotion": emotion})
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"[ERROR] No .wav files found in: {data_dir}\nCheck that TESS is extracted to project/data/TESS/")
    print(f"[INFO] Loaded {len(df)} samples | Emotions: {df['emotion'].value_counts().to_dict()}")
    return df


def extract_features(path, sr=22050, max_len=3, n_mfcc=40):
    """
    Preprocessing: load, resample, trim silence, pad/truncate.
    Feature Extraction: MFCC + Delta + Delta-Delta -> shape (time_steps, 3*n_mfcc)
    """
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)       # trim silence

    # Pad or truncate to max_len seconds
    max_samples = sr * max_len
    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))
    else:
        y = y[:max_samples]

    y = y / (np.max(np.abs(y)) + 1e-8)              # normalize

    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (40, T)
    delta      = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=0)       # (120, T)
    features = features.T                                           # (T, 120)
    return features.astype(np.float32)


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = extract_features(
            row["path"],
            sr=CONFIG["sample_rate"],
            max_len=CONFIG["max_len"],
            n_mfcc=CONFIG["n_mfcc"],
        )
        label = self.le.transform([row["emotion"]])[0]
        return torch.tensor(feat), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
# 3. MODEL — Bidirectional LSTM
# ─────────────────────────────────────────────
class SpeechEmotionLSTM(nn.Module):
    """
    Architecture Decision:
    Bi-LSTM captures temporal dynamics in both directions, which is
    important for speech where context from future frames also matters.
    Two stacked LSTM layers learn hierarchical temporal patterns.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)   # self-attention over time
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, time_steps, features)
        out, _ = self.lstm(x)                            # (batch, T, 2*hidden)
        attn_w = torch.softmax(self.attention(out), dim=1)  # (batch, T, 1)
        context = (attn_w * out).sum(dim=1)              # (batch, 2*hidden)
        return self.classifier(context)


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def train():
    print(f"[INFO] Using device: {CONFIG['device']}")

    # Load data
    df = load_tess_metadata(CONFIG["data_dir"])
    le = LabelEncoder()
    le.fit(EMOTIONS)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = SpeechDataset(train_df, le)
    val_ds   = SpeechDataset(val_df, le)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # Model
    sample_feat = extract_features(df.iloc[0]["path"])
    input_size  = sample_feat.shape[1]   # 3 * n_mfcc = 120

    model = SpeechEmotionLSTM(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=len(EMOTIONS),
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}"):
            feats, labels = feats.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(CONFIG["device"]), labels.to(CONFIG["device"])
                logits = model(feats)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pt"))
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.4f})")

    # ── Save results ──
    with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
        json.dump(history, f)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("\n[Classification Report]\n", report)
    with open(os.path.join(CONFIG["save_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title("Speech Model — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "confusion_matrix.png"))
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.legend(); plt.title("Loss Curves"); plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy", color="green")
    plt.legend(); plt.title("Validation Accuracy"); plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"))
    plt.close()

    print(f"\n[INFO] Best Val Accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Results saved to: {CONFIG['save_dir']}")


if __name__ == "__main__":
    train()
