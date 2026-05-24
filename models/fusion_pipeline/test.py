"""
Fusion Pipeline — Test / Inference
Evaluates the multimodal model and generates t-SNE for Fusion representations.
Also compares accuracy across all 3 model variants.
"""

import os
import sys
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (
    CONFIG, EMOTIONS, load_tess_multimodal,
    MultimodalDataset, MultimodalFusionModel,
    extract_speech_features,
)


def plot_tsne(representations, labels, label_names, title, save_path):
    print(f"[INFO] Running t-SNE...")
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


def plot_model_comparison(save_dir):
    """Bar chart comparing accuracy of all 3 models."""
    accs = {}
    for name, path in [
        ("Speech-only", os.path.join(PROJECT_ROOT, "Results", "speech", "test_accuracy.json")),
        ("Text-only",   os.path.join(PROJECT_ROOT, "Results", "text",   "test_accuracy.json")),
        ("Multimodal",  os.path.join(PROJECT_ROOT, "Results", "fusion", "test_accuracy.json")),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                accs[name] = json.load(f)["test_accuracy"]

    if not accs:
        print("[WARN] Could not find accuracy files for comparison. Run test.py for all pipelines first.")
        return

    plt.figure(figsize=(8, 5))
    colors = ["#4C9BE8", "#5DB87A", "#9B59B6"]
    bars   = plt.bar(accs.keys(), [v * 100 for v in accs.values()], color=colors, width=0.4)
    for bar, acc in zip(bars, accs.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
    plt.ylim(0, 105)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison — Speech vs Text vs Multimodal")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"))
    plt.close()
    print(f"[INFO] Model comparison chart saved.")


def test():
    device   = CONFIG["device"]
    save_dir = CONFIG["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    df = load_tess_multimodal(CONFIG["data_dir"])
    le = LabelEncoder()
    le.fit(EMOTIONS)

    _, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    print(f"[INFO] Test samples: {len(test_df)}")

    tokenizer   = BertTokenizer.from_pretrained(CONFIG["bert_model"])
    test_ds     = MultimodalDataset(test_df, tokenizer, le)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    sample_feat       = extract_speech_features(df.iloc[0]["path"])
    speech_input_size = sample_feat.shape[1]

    model = MultimodalFusionModel(speech_input_size, len(EMOTIONS)).to(device)
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_fused = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            speech = batch["speech"].to(device)
            ids    = batch["input_ids"].to(device)
            mask   = batch["attn_mask"].to(device)
            labels = batch["label"].to(device)
            logits, fused = model(speech, ids, mask)
            test_loss += criterion(logits, labels).item()
            preds      = logits.argmax(dim=1)
            correct   += (preds == labels).sum().item()
            total     += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_fused.append(fused.cpu().numpy())

    test_acc = correct / total
    print(f"\n[TEST] Loss: {test_loss/len(test_loader):.4f} | Accuracy: {test_acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("\n[Classification Report]\n", report)
    with open(os.path.join(save_dir, "test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Purples")
    plt.title("Fusion Model — Test Confusion Matrix")
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
    print(f"[INFO] {len(failures)} failures. Saved top 20.")

    # t-SNE of Fusion representations
    fused_reps = np.concatenate(all_fused, axis=0)
    plot_tsne(
        fused_reps, np.array(all_labels), le.classes_,
        "Multimodal — Fusion Block Representations (t-SNE)",
        os.path.join(save_dir, "tsne_fusion.png"),
    )

    with open(os.path.join(save_dir, "test_accuracy.json"), "w") as f:
        json.dump({"test_accuracy": round(test_acc, 4)}, f)

    # Final comparison chart across all 3 models
    plot_model_comparison(save_dir)

    print(f"\n[INFO] All test results saved to: {save_dir}")


if __name__ == "__main__":
    test()
