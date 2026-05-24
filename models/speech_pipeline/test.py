"""
Speech Pipeline — Test / Inference
Loads best saved model and evaluates on a held-out test set.
Also extracts Temporal Modelling (LSTM hidden) representations for t-SNE visualization.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Re-use from train.py
from train import (
    CONFIG, EMOTIONS, load_tess_metadata,
    extract_features, SpeechDataset, SpeechEmotionLSTM,
)


def extract_representations(model, loader, device):
    """Extract LSTM context vectors (Temporal Modelling outputs) for t-SNE."""
    model.eval()
    representations, all_labels = [], []
    # Hook to capture the output of the attention pooling step
    hooks = []

    def hook_fn(module, input, output):
        representations.append(output.detach().cpu().numpy())

    # Register hook on the attention linear layer input
    handle = model.lstm.register_forward_hook(
        lambda m, i, o: representations.append(o[0].mean(dim=1).detach().cpu().numpy())
    )

    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="Extracting representations"):
            feats = feats.to(device)
            model(feats)
            all_labels.extend(labels.numpy())

    handle.remove()
    return np.concatenate(representations, axis=0), np.array(all_labels)


def plot_tsne(representations, labels, label_names, title, save_path):
    """t-SNE cluster visualization of learned representations."""
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
    print(f"[INFO] Saved t-SNE plot to {save_path}")


def test():
    device   = CONFIG["device"]
    save_dir = CONFIG["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = load_tess_metadata(CONFIG["data_dir"])
    le = LabelEncoder()
    le.fit(EMOTIONS)

    # Use 20% as test set (same split as training)
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    print(f"[INFO] Test samples: {len(test_df)}")

    test_ds     = SpeechDataset(test_df, le)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # Load model
    sample_feat = extract_features(df.iloc[0]["path"])
    input_size  = sample_feat.shape[1]

    model = SpeechEmotionLSTM(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=len(EMOTIONS),
        dropout=CONFIG["dropout"],
    ).to(device)
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for feats, labels in tqdm(test_loader, desc="Testing"):
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            test_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct / total
    print(f"\n[TEST] Loss: {test_loss/len(test_loader):.4f} | Accuracy: {test_acc:.4f}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=le.classes_)
    print("\n[Classification Report]\n", report)
    with open(os.path.join(save_dir, "test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title("Speech Model — Test Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()

    # Failure case analysis
    failures = []
    test_df = test_df.reset_index(drop=True)
    for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
        if pred != true:
            failures.append({
                "file":      test_df.iloc[i]["path"],
                "true":      le.classes_[true],
                "predicted": le.classes_[pred],
            })
    pd.DataFrame(failures[:20]).to_csv(os.path.join(save_dir, "failure_cases.csv"), index=False)
    print(f"[INFO] {len(failures)} failure cases. Saved top 20 to failure_cases.csv")

    # t-SNE visualization of temporal modelling representations
    reps, rep_labels = extract_representations(model, test_loader, device)
    plot_tsne(
        reps, rep_labels, le.classes_,
        "Speech — Temporal Modelling Representations (t-SNE)",
        os.path.join(save_dir, "tsne_temporal.png"),
    )

    # Save accuracy summary
    with open(os.path.join(save_dir, "test_accuracy.json"), "w") as f:
        json.dump({"test_accuracy": round(test_acc, 4)}, f)

    print(f"\n[INFO] All test results saved to: {save_dir}")


if __name__ == "__main__":
    test()
