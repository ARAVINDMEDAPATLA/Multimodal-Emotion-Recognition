"""
Text-Only Emotion Recognition Pipeline
Architecture:
  Preprocessing       -> Clean + Tokenize with BERT tokenizer
  Feature Extraction  -> BERT last-hidden-state (tokens x 768)
  Contextual Model    -> Fine-tuned BERT (bert-base-uncased) + pooling
  Classifier          -> FC + Softmax
Dataset: TESS — uses transcripts paired with speech files.
"""

import os
import re
import json
import numpy as np
import pandas as pd
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

# Project root = 2 levels up from this file (project/models/text_pipeline/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "data_dir":   os.path.join(PROJECT_ROOT, "data", "TESS"),
    "bert_model": "bert-base-uncased",
    "max_length": 32,
    "batch_size": 32,
    "epochs":     30,
    "lr":         5e-5,
    "dropout":    0.3,
    "save_dir":   os.path.join(PROJECT_ROOT, "Results", "text"),
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove special chars, strip extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tess_text(data_dir: str) -> pd.DataFrame:
    """
    TESS filename format: OAF_back_angry.wav
    The word 'back' is the spoken word (the transcript).
    Build DataFrame: path, word (transcript), emotion.
    """
    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                parts   = f.replace(".wav", "").split("_")
                # parts = [speaker, word, emotion]
                if len(parts) >= 3:
                    emotion = parts[-1].lower()
                    word    = parts[-2].lower()
                    if emotion in EMOTIONS:
                        # Wrap the single word in a sentence template to give BERT
                        # more token context. TESS words are semantically neutral,
                        # so we create a simple carrier sentence.
                        text = f"the speaker said {word} out loud"
                        records.append({
                            "path":    os.path.join(root, f),
                            "text":    text,
                            "emotion": emotion,
                        })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"[ERROR] No .wav files found in: {data_dir}")
    print(f"[INFO] Loaded {len(df)} text samples | Emotions: {df['emotion'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, label_encoder, max_length):
        self.df          = df.reset_index(drop=True)
        self.tokenizer   = tokenizer
        self.le          = label_encoder
        self.max_length  = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        encoded = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = self.le.transform([row["emotion"]])[0]
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 3. MODEL — Fine-tuned BERT Classifier
# ─────────────────────────────────────────────
class BERTEmotionClassifier(nn.Module):
    """
    Architecture Decision:
    BERT's bidirectional transformer captures deep contextual meaning across
    all tokens simultaneously. Fine-tuning the last 4 layers allows the model
    to adapt its contextual representations to the emotion domain while
    retaining general language understanding from pre-training.
    The [CLS] token pooling gives a sentence-level representation.
    """
    def __init__(self, bert_model_name, num_classes, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Unfreeze last 6 BERT layers for better fine-tuning on this task
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        for name, param in self.bert.encoder.layer[-6:].named_parameters():
            param.requires_grad = True
        # Also unfreeze the pooler
        for name, param in self.bert.pooler.named_parameters():
            param.requires_grad = True

        hidden = self.bert.config.hidden_size   # 768 for bert-base
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 256),   # CLS + mean-pool concatenated
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out  = outputs.last_hidden_state[:, 0, :]          # [CLS] token
        mean_out = outputs.last_hidden_state.mean(dim=1)       # mean pooling
        pooled   = torch.cat([cls_out, mean_out], dim=-1)      # (batch, 1536)
        return self.classifier(pooled), pooled                 # return logits + representation


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def train():
    print(f"[INFO] Using device: {CONFIG['device']}")

    df = load_tess_text(CONFIG["data_dir"])
    le = LabelEncoder()
    le.fit(EMOTIONS)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])

    train_ds     = TextDataset(train_df, tokenizer, le, CONFIG["max_length"])
    val_ds       = TextDataset(val_df,   tokenizer, le, CONFIG["max_length"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model     = BERTEmotionClassifier(CONFIG["bert_model"], len(EMOTIONS), CONFIG["dropout"]).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()

    history       = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}"):
            ids   = batch["input_ids"].to(CONFIG["device"])
            mask  = batch["attention_mask"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])

            optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss      = criterion(logits, labels)
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
                ids    = batch["input_ids"].to(CONFIG["device"])
                mask   = batch["attention_mask"].to(CONFIG["device"])
                labels = batch["label"].to(CONFIG["device"])
                logits, _ = model(ids, mask)
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

    # Save history and plots
    with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
        json.dump(history, f)

    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("\n[Classification Report]\n", report)
    with open(os.path.join(CONFIG["save_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Greens")
    plt.title("Text Model — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "confusion_matrix.png"))
    plt.close()

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
