"""
Text Pipeline — Test / Inference
Loads best saved BERT model, evaluates on test set,
and generates t-SNE of Contextual Modelling (BERT [CLS]) representations.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from train import (
    CONFIG, EMOTIONS, load_tess_text,
    TextDataset, BERTEmotionClassifier,
)


def plot_tsne(representations, labels, label_names, title, save_path):
    print(f"[INFO] Running t-SNE on {representations.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb  = tsne.fit_transform(representations)
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hls", len(label_names))
    for i, name in enumerate(label_names):
        mask = labels == i
        plt.scatter(emb[mask, 0], emb[mask, 1], label=name, alpha=0.6, s=20, color=palette[i])
    plt.legend(markerscale=2)
    plt.title(title)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved to {save_path}")


def test():
    device   = CONFIG["device"]
    save_dir = CONFIG["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    df = load_tess_text(CONFIG["data_dir"])
    le = LabelEncoder()
    le.fit(EMOTIONS)

    _, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    print(f"[INFO] Test samples: {len(test_df)}")

    tokenizer   = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    test_ds     = TextDataset(test_df, tokenizer, le, CONFIG["max_length"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = BERTEmotionClassifier(CONFIG["bert_model"], len(EMOTIONS), CONFIG["dropout"]).to(device)
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_reps = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, reps = model(ids, mask)
            test_loss += criterion(logits, labels).item()
            preds      = logits.argmax(dim=1)
            correct   += (preds == labels).sum().item()
            total     += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_reps.append(reps.cpu().numpy())

    test_acc = correct / total
    print(f"\n[TEST] Loss: {test_loss/len(test_loader):.4f} | Accuracy: {test_acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("\n[Classification Report]\n", report)
    with open(os.path.join(save_dir, "test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Greens")
    plt.title("Text Model — Test Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()

    # Failure cases
    failures = []
    test_df = test_df.reset_index(drop=True)
    for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
        if pred != true:
            failures.append({
                "text":      test_df.iloc[i]["text"],
                "true":      le.classes_[true],
                "predicted": le.classes_[pred],
            })
    pd.DataFrame(failures[:20]).to_csv(os.path.join(save_dir, "failure_cases.csv"), index=False)
    print(f"[INFO] {len(failures)} failures. Saved top 20 to failure_cases.csv")

    # t-SNE of Contextual Modelling representations
    reps = np.concatenate(all_reps, axis=0)
    plot_tsne(
        reps, np.array(all_labels), le.classes_,
        "Text — Contextual Modelling Representations (BERT [CLS]) (t-SNE)",
        os.path.join(save_dir, "tsne_contextual.png"),
    )

    with open(os.path.join(save_dir, "test_accuracy.json"), "w") as f:
        json.dump({"test_accuracy": round(test_acc, 4)}, f)

    print(f"\n[INFO] All test results saved to: {save_dir}")


if __name__ == "__main__":
    test()
