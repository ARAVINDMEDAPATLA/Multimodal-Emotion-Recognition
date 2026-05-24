"""
Multimodal Fusion Emotion Recognition Pipeline
Architecture:
  Speech branch : MFCCs + Delta -> Bi-LSTM -> context vector (256-d)
  Text branch   : BERT tokenizer -> Fine-tuned BERT -> [CLS] vector (768-d)
  Fusion        : Cross-modal attention + concatenation -> Fusion vector (512-d)
  Classifier    : FC + Softmax
Dataset: TESS — pairs audio with transcript (word from filename).
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Project root = 2 levels up from this file (project/models/fusion_pipeline/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from speech pipeline using importlib to avoid name collision with local train.py
import importlib.util
_speech_train_path = os.path.join(PROJECT_ROOT, "models", "speech_pipeline", "train.py")
_spec = importlib.util.spec_from_file_location("speech_train", _speech_train_path)
_speech_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_speech_train)
extract_speech_features = _speech_train.extract_features
EMOTIONS = _speech_train.EMOTIONS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir":     os.path.join(PROJECT_ROOT, "data", "TESS"),
    "bert_model":   "bert-base-uncased",
    "sample_rate":  22050,
    "max_len":      3,
    "n_mfcc":       40,
    "max_text_len": 64,
    "batch_size":   16,
    "epochs":       30,
    "lr":           1e-4,
    "dropout":      0.3,
    "lstm_hidden":  128,
    "lstm_layers":  2,
    "save_dir":     os.path.join(PROJECT_ROOT, "Results", "fusion"),
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_tess_multimodal(data_dir: str) -> pd.DataFrame:
    """Load TESS dataset with both audio path and text transcript."""
    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                parts = f.replace(".wav", "").split("_")
                if len(parts) >= 3:
                    emotion = parts[-1].lower()
                    word    = parts[-2].lower()
                    if emotion in EMOTIONS:
                        records.append({
                            "path":    os.path.join(root, f),
                            "text":    clean_text(word),
                            "emotion": emotion,
                        })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"[ERROR] No .wav files found in: {data_dir}")
    print(f"[INFO] Loaded {len(df)} multimodal samples")
    return df


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, label_encoder):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le        = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Speech features
        speech_feat = extract_speech_features(
            row["path"],
            sr=CONFIG["sample_rate"],
            max_len=CONFIG["max_len"],
            n_mfcc=CONFIG["n_mfcc"],
        )   # (T, 120)

        # Text features
        encoded = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_text_len"],
            return_tensors="pt",
        )

        label = self.le.transform([row["emotion"]])[0]
        return {
            "speech":       torch.tensor(speech_feat),
            "input_ids":    encoded["input_ids"].squeeze(0),
            "attn_mask":    encoded["attention_mask"].squeeze(0),
            "label":        torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 3. MODEL — Cross-Modal Attention Fusion
# ─────────────────────────────────────────────
class SpeechBranch(nn.Module):
    """Bi-LSTM temporal modelling branch for speech."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                           # (B, T, 2H)
        w      = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        return (w * out).sum(dim=1)                     # (B, 2H)


class TextBranch(nn.Module):
    """Fine-tuned BERT contextual modelling branch for text."""
    def __init__(self, bert_model_name, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        for name, param in self.bert.encoder.layer[-4:].named_parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]           # (B, 768)


class CrossModalAttention(nn.Module):
    """
    Architecture Decision — Cross-Modal Attention Fusion:
    Instead of simple concatenation (early fusion) or averaging predictions
    (late fusion), cross-modal attention lets the text representation attend
    to the speech context and vice versa. This is more expressive because
    it allows the model to learn WHICH aspects of speech and text are most
    relevant for each other's emotion signals, rather than treating them
    independently. This is especially powerful when one modality is ambiguous
    (e.g., neutral tone with happy words).
    """
    def __init__(self, speech_dim, text_dim, fusion_dim):
        super().__init__()
        # Project both to same fusion space
        self.speech_proj = nn.Linear(speech_dim, fusion_dim)
        self.text_proj   = nn.Linear(text_dim,   fusion_dim)
        # Cross-attention: speech attends to text
        self.s2t_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
        # Cross-attention: text attends to speech
        self.t2s_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim * 2)

    def forward(self, speech_vec, text_vec):
        # Project
        s = self.speech_proj(speech_vec).unsqueeze(1)   # (B, 1, fusion_dim)
        t = self.text_proj(text_vec).unsqueeze(1)       # (B, 1, fusion_dim)
        # Cross-attend
        s_attended, _ = self.s2t_attn(query=s, key=t, value=t)
        t_attended, _ = self.t2s_attn(query=t, key=s, value=s)
        # Fuse
        fused = torch.cat([s_attended.squeeze(1), t_attended.squeeze(1)], dim=-1)
        return self.norm(fused)                         # (B, 2*fusion_dim)


class MultimodalFusionModel(nn.Module):
    def __init__(self, speech_input_size, num_classes):
        super().__init__()
        lstm_hidden  = CONFIG["lstm_hidden"]
        speech_dim   = lstm_hidden * 2     # Bi-LSTM: 256
        text_dim     = 768                 # BERT hidden
        fusion_dim   = 256

        self.speech_branch = SpeechBranch(speech_input_size, lstm_hidden, CONFIG["lstm_layers"], CONFIG["dropout"])
        self.text_branch   = TextBranch(CONFIG["bert_model"], CONFIG["dropout"])
        self.fusion        = CrossModalAttention(speech_dim, text_dim, fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(256, num_classes),
        )

    def forward(self, speech, input_ids, attn_mask):
        speech_vec = self.speech_branch(speech)
        text_vec   = self.text_branch(input_ids, attn_mask)
        fused      = self.fusion(speech_vec, text_vec)
        return self.classifier(fused), fused


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def train():
    print(f"[INFO] Using device: {CONFIG['device']}")

    df = load_tess_multimodal(CONFIG["data_dir"])
    le = LabelEncoder()
    le.fit(EMOTIONS)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer    = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    train_ds     = MultimodalDataset(train_df, tokenizer, le)
    val_ds       = MultimodalDataset(val_df,   tokenizer, le)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # Determine speech feature input size
    sample_feat = extract_speech_features(df.iloc[0]["path"])
    speech_input_size = sample_feat.shape[1]   # 120

    model     = MultimodalFusionModel(speech_input_size, len(EMOTIONS)).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()

    history      = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}"):
            speech = batch["speech"].to(CONFIG["device"])
            ids    = batch["input_ids"].to(CONFIG["device"])
            mask   = batch["attn_mask"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])

            optimizer.zero_grad()
            logits, _ = model(speech, ids, mask)
            loss       = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                speech = batch["speech"].to(CONFIG["device"])
                ids    = batch["input_ids"].to(CONFIG["device"])
                mask   = batch["attn_mask"].to(CONFIG["device"])
                labels = batch["label"].to(CONFIG["device"])
                logits, _ = model(speech, ids, mask)
                val_loss += criterion(logits, labels).item()
                preds     = logits.argmax(dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pt"))
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.4f})")

    # Save results
    with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
        json.dump(history, f)

    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("\n[Classification Report]\n", report)
    with open(os.path.join(CONFIG["save_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Purples")
    plt.title("Fusion Model — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.legend(); plt.title("Fusion — Loss Curves"); plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy", color="purple")
    plt.legend(); plt.title("Fusion — Validation Accuracy"); plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"))
    plt.close()

    print(f"\n[INFO] Best Val Accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Results saved to: {CONFIG['save_dir']}")


if __name__ == "__main__":
    train()
