import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "Christines")
MAPPING_FILE = os.path.join(os.path.dirname(__file__), "mapping.json")

# Regex to match standard files: XX_YYYV_Z.txt
FILE_PATTERN = re.compile(r"^(\d+)_(\d+)V_(\d+)\.txt$")

with open(MAPPING_FILE) as f:
    POLLEN_MAP = json.load(f)  # str->str, e.g. "2" -> "fagus"


def load_single_file(filepath):
    """Load a single spectrum file. Returns (wavelengths, intensities) arrays."""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]


def load_all_spectra(data_dir=DATA_DIR):
    """
    Load all standard-named spectra from the Christines folder.
    Returns a list of dicts: {pollen_id, voltage, sample_num, species, wavelengths, intensities, filename}
    """
    records = []
    for fname in sorted(os.listdir(data_dir)):
        m = FILE_PATTERN.match(fname)
        if not m:
            continue
        pollen_id = m.group(1).lstrip("0") or "0"
        voltage = int(m.group(2))
        sample_num = int(m.group(3))
        species = POLLEN_MAP.get(pollen_id, f"unknown_{pollen_id}")
        wl, intensity = load_single_file(os.path.join(data_dir, fname))
        records.append({
            "pollen_id": pollen_id,
            "voltage": voltage,
            "sample_num": sample_num,
            "species": species,
            "wavelengths": wl,
            "intensities": intensity,
            "filename": fname,
        })
    return records


# ──────────────────────────────────────────────────────────────────────
# 2. PLOTTING RAW DATA
# ──────────────────────────────────────────────────────────────────────

def plot_raw_spectrum(record):
    """Plot a single raw spectrum."""
    plt.figure(figsize=(12, 4))
    plt.plot(record["wavelengths"], record["intensities"], linewidth=0.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Raw spectrum – {record['species']} (file: {record['filename']})")
    plt.tight_layout()
    plt.show()


def plot_all_samples_for_species(records, species_name):
    """Plot all samples for a given species overlaid."""
    subset = [r for r in records if r["species"] == species_name]
    if not subset:
        print(f"No records found for species '{species_name}'")
        return
    plt.figure(figsize=(12, 5))
    for r in subset:
        plt.plot(r["wavelengths"], r["intensities"], linewidth=0.5,
                 label=r["filename"])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"All raw spectra – {species_name} ({len(subset)} samples)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING: Savitzky-Golay + MinMax Scaling
# ──────────────────────────────────────────────────────────────────────

def preprocess_spectrum(intensities, sg_window=15, sg_polyorder=3):
    """
    Apply Savitzky-Golay smoothing, then Min-Max scale to [0, 1].
    Returns (smoothed, scaled) arrays.
    """
    smoothed = savgol_filter(intensities, window_length=sg_window, polyorder=sg_polyorder)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(smoothed.reshape(-1, 1)).ravel()
    return smoothed, scaled


def plot_preprocessing_comparison(record, sg_window=15, sg_polyorder=3):
    """Show raw vs smoothed vs scaled for a single record."""
    wl = record["wavelengths"]
    raw = record["intensities"]
    smoothed, scaled = preprocess_spectrum(raw, sg_window, sg_polyorder)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(wl, raw, linewidth=0.5, color="gray")
    axes[0].set_title(f"Raw – {record['species']} ({record['filename']})")
    axes[0].set_ylabel("Intensity")

    axes[1].plot(wl, smoothed, linewidth=0.8, color="blue")
    axes[1].set_title(f"After Savitzky-Golay (window={sg_window}, poly={sg_polyorder})")
    axes[1].set_ylabel("Intensity")

    axes[2].plot(wl, scaled, linewidth=0.8, color="green")
    axes[2].set_title("After Savitzky-Golay + Min-Max Scaling [0,1]")
    axes[2].set_ylabel("Scaled Intensity")
    axes[2].set_xlabel("Wavelength (nm)")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 4. PREPARE DATASET FOR CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────

def build_dataset(records, sg_window=15, sg_polyorder=3):
    """
    Preprocess all records and return:
      X: np.array of shape (N, 1388) – scaled spectra
      y: np.array of species labels (strings)
      groups: list of (pollen_id, sample_num) for group-aware splitting
    """
    X, y, groups = [], [], []
    for r in records:
        _, scaled = preprocess_spectrum(r["intensities"], sg_window, sg_polyorder)
        X.append(scaled)
        y.append(r["species"])
        groups.append((r["pollen_id"], r["sample_num"]))
    return np.array(X), np.array(y), groups


def train_val_test_split(records):
    """
    Split per species: for each pollen_id, sort samples by sample_num,
    use the first 3 for training, next 1 for validation, last 1 for test.
    Species with fewer than 3 samples get all samples in training.
    Returns (train_records, val_records, test_records).
    """
    from collections import defaultdict
    by_pollen = defaultdict(list)
    for r in records:
        by_pollen[r["pollen_id"]].append(r)

    train, val, test = [], [], []
    for pid, samples in by_pollen.items():
        samples_sorted = sorted(samples, key=lambda x: x["sample_num"])
        n = len(samples_sorted)
        if n >= 5:
            train.extend(samples_sorted[:3])
            val.append(samples_sorted[3])
            test.append(samples_sorted[4])
        elif n >= 3:
            train.extend(samples_sorted[:-2])
            val.append(samples_sorted[-2])
            test.append(samples_sorted[-1])
        elif n == 2:
            train.append(samples_sorted[0])
            val.append(samples_sorted[1])
        else:
            train.extend(samples_sorted)

    return train, val, test


# ──────────────────────────────────────────────────────────────────────
# 5. 1D CNN MODEL
# ──────────────────────────────────────────────────────────────────────

class SpectrumCNN(nn.Module):
    """Simple 1D CNN for spectral classification."""
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, length)
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, 128)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────
# 6. TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────

def make_tensors(X, y_encoded):
    """Convert numpy arrays to PyTorch tensors for the CNN (adds channel dim)."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    return X_t, y_t


def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, test_loader, label_encoder, device="cpu"):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    labels = label_encoder.classes_
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(all_true, all_preds, target_names=labels, zero_division=0))

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.6)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.show()

    return all_true, all_preds


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 7. BASELINE MODELS: SVM & MLP
# ──────────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train_enc, X_test, y_test_enc, label_encoder):
    """Train an SVM with RBF kernel + PCA dimensionality reduction."""
    print("\n" + "=" * 60)
    print("SVM BASELINE (RBF kernel + PCA)")
    print("=" * 60)
    # PCA to reduce 1388 features — keeps 95% variance
    svm_pipe = make_pipeline(
        PCA(n_components=0.95),
        SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovo")
    )
    svm_pipe.fit(X_train, y_train_enc)
    preds = svm_pipe.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    n_components = svm_pipe[0].n_components_
    print(f"PCA reduced {X_train.shape[1]} → {n_components} features")
    print(f"SVM Test Accuracy: {acc:.3f}")
    labels = label_encoder.classes_
    print(classification_report(y_test_enc, preds, target_names=labels, zero_division=0))
    return acc, preds


def train_mlp(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, label_encoder):
    """Train a scikit-learn MLP classifier."""
    print("\n" + "=" * 60)
    print("MLP BASELINE (scikit-learn)")
    print("=" * 60)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        random_state=42,
    )
    # Combine train+val for sklearn (it does its own internal early-stopping split)
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train_enc, y_val_enc])
    mlp.fit(X_fit, y_fit)
    preds = mlp.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"MLP Test Accuracy: {acc:.3f}")
    labels = label_encoder.classes_
    print(classification_report(y_test_enc, preds, target_names=labels, zero_division=0))
    return acc, preds


def print_comparison(results):
    """Print a summary table comparing all models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Test Accuracy':>15}")
    print("-" * 37)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 30)
        print(f"{name:<20} {acc:>13.1%}  {bar}")
    print()


# ──────────────────────────────────────────────────────────────────────
# 8. MAIN – RUN EVERYTHING STEP BY STEP
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Step 1: Load all data ──
    print("Loading data...")
    all_records = load_all_spectra()
    print(f"Loaded {len(all_records)} spectra from "
          f"{len(set(r['pollen_id'] for r in all_records))} pollen types")

    # ── Step 2: Plot a raw spectrum (fagus, sample 1) ──
    fagus_records = [r for r in all_records if r["species"] == "fagus"]
    print(f"\nFagus has {len(fagus_records)} samples")
    plot_raw_spectrum(fagus_records[0])

    # ── Step 3: Plot all fagus samples overlaid ──
    plot_all_samples_for_species(all_records, "fagus")

    # ── Step 4: Show preprocessing comparison ──
    plot_preprocessing_comparison(fagus_records[0])

    # ── Step 5: Split data into train/val/test ──
    train_recs, val_recs, test_recs = train_val_test_split(all_records)
    print(f"\nSplit: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test")

    # ── Step 6: Build preprocessed arrays ──
    X_train, y_train, _ = build_dataset(train_recs)
    X_val, y_val, _ = build_dataset(val_recs)
    X_test, y_test, _ = build_dataset(test_recs)

    # Encode labels
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    input_length = X_train.shape[1]
    print(f"Classes: {num_classes}, Input length: {input_length}")
    print(f"Species: {list(le.classes_)}")

    # ── Step 7: Create DataLoaders ──
    X_train_t, y_train_t = make_tensors(X_train, y_train_enc)
    X_val_t, y_val_t = make_tensors(X_val, y_val_enc)
    X_test_t, y_test_t = make_tensors(X_test, y_test_enc)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    # ── Step 8: Train the CNN ──
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model = SpectrumCNN(input_length, num_classes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    model, history = train_model(model, train_loader, val_loader,
                                  num_epochs=400, lr=1e-3, device=device)

    # ── Step 9: Plot training history ──
    plot_training_history(history)

    # ── Step 10: Evaluate CNN on test set ──
    true_labels, pred_labels = evaluate_model(model, test_loader, le, device=device)
    cnn_acc = accuracy_score(true_labels, pred_labels)

    # ── Step 11: SVM baseline ──
    X_train_all = np.vstack([X_train, X_val])  # SVM uses train+val
    y_train_all = np.concatenate([y_train_enc, y_val_enc])
    svm_acc, _ = train_svm(X_train_all, y_train_all, X_test, y_test_enc, le)

    # ── Step 12: MLP baseline ──
    mlp_acc, _ = train_mlp(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le)

    # ── Step 13: Comparison ──
    print_comparison({"1D-CNN": cnn_acc, "SVM (RBF+PCA)": svm_acc, "MLP (sklearn)": mlp_acc})
