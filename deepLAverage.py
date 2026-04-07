"""
Pollen classifier for average_data — 12 species, ~5 samples each.
Same pipeline as deepLChristine.py adapted for this dataset.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

N_PCA = 6  # number of PCA components (set to None for 95% variance)

# ──────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "average_data")

# Map folder names to species labels
FOLDER_TO_SPECIES = {
    "average_carpinus":       "carpinus_betulus",
    "average_cedrus@85V_2":   "cedrus_85V",
    "average_cedrus@8V":      "cedrus_8V",
    "average_mimosa":         "mimosa",
    "average_olea":           "olea_europea",
    "average_parietaria":     "parietaria",
    "average_planteago":      "plantago_lanceolata",
    "average_quercus_robur":  "quercus_robur",
    "average_salix":          "salix_caprea",
    "average_salsola_kali":   "salsola_kali",
    "castanea_average":       "castanea_sativa",
    "cypressus_average":      "cypressus_arizonica",
}


def load_single_file(filepath):
    """Load a single spectrum file. Returns (wavelengths, intensities) arrays."""
    data = np.loadtxt(filepath)
    return data[:, 0][391:1786], data[:, 1][391:1786]


def load_all_spectra(data_dir=DATA_DIR):
    """
    Load all spectra from the average_data subfolders.
    Skips .png files and tot_average_data files.
    Returns a list of dicts with species, wavelengths, intensities, filename.
    """
    records = []
    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        species = FOLDER_TO_SPECIES.get(folder_name)
        if species is None:
            print(f"  Warning: unknown folder '{folder_name}', skipping")
            continue

        sample_num = 0
        for fname in sorted(os.listdir(folder_path)):
            # Skip images and the tot_average_data file
            if fname.endswith(".png") or fname.startswith("tot_average"):
                continue
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                wl, intensity = load_single_file(fpath)
            except Exception as e:
                print(f"  Warning: could not load {fpath}: {e}")
                continue
            sample_num += 1
            records.append({
                "species": species,
                "sample_num": sample_num,
                "wavelengths": wl,
                "intensities": intensity,
                "filename": fname,
                "folder": folder_name,
            })
    return records


# ──────────────────────────────────────────────────────────────────────
# 2. PLOTTING
# ──────────────────────────────────────────────────────────────────────

def plot_raw_spectrum(record):
    """Plot a single raw spectrum."""
    plt.figure(figsize=(12, 4))
    plt.plot(record["wavelengths"], record["intensities"], linewidth=0.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Raw spectrum – {record['species']} ({record['filename']})")
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
# 3. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────

def preprocess_spectrum(intensities, sg_window=15, sg_polyorder=3):
    """Savitzky-Golay smoothing + Min-Max scale to [0, 1]."""
    smoothed = savgol_filter(intensities, window_length=sg_window, polyorder=sg_polyorder)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(smoothed.reshape(-1, 1)).ravel()
    return smoothed, scaled


# ──────────────────────────────────────────────────────────────────────
# 4. DATASET BUILDING & SPLITTING
# ──────────────────────────────────────────────────────────────────────

def build_dataset(records, sg_window=15, sg_polyorder=3):
    """Preprocess all records → (X, y) arrays."""
    X, y = [], []
    for r in records:
        _, scaled = preprocess_spectrum(r["intensities"], sg_window, sg_polyorder)
        X.append(scaled)
        y.append(r["species"])
    return np.array(X), np.array(y)


def train_val_test_split(records):
    """
    Per species: sort by sample_num, first 3 → train, 4th → val, 5th → test.
    Species with 4 samples: 2 train, 1 val, 1 test.
    """
    from collections import defaultdict
    by_species = defaultdict(list)
    for r in records:
        by_species[r["species"]].append(r)

    train, val, test = [], [], []
    for species, samples in sorted(by_species.items()):
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
# 5. MODELS
# ──────────────────────────────────────────────────────────────────────

class SpectrumCNN(nn.Module):
    """1D CNN for spectral classification."""
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
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class SpectrumFC(nn.Module):
    """Small FC network for PCA-reduced features."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────
# 6. TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────

def make_tensors(X, y_encoded, add_channel=True):
    """Convert numpy arrays to PyTorch tensors."""
    X_t = torch.tensor(X, dtype=torch.float32)
    if add_channel:
        X_t = X_t.unsqueeze(1)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    return X_t, y_t


def mixup_batch(xb, yb, num_classes, alpha=0.4):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = xb.size(0)
    index = torch.randperm(batch_size, device=xb.device)
    mixed_x = lam * xb + (1 - lam) * xb[index]
    y_onehot = torch.zeros(batch_size, num_classes, device=xb.device)
    y_onehot.scatter_(1, yb.unsqueeze(1), 1.0)
    y_onehot2 = torch.zeros(batch_size, num_classes, device=xb.device)
    y_onehot2.scatter_(1, yb[index].unsqueeze(1), 1.0)
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot2
    return mixed_x, mixed_y


def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3,
                device="cpu", use_mixup=True, label_smoothing=0.1):
    """Train with Adam + cosine annealing + optional mixup."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_soft = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    num_classes = list(model.classifier.children())[-1].out_features

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if use_mixup:
                mixed_x, mixed_y = mixup_batch(xb, yb, num_classes)
                out = model(mixed_x)
                log_probs = torch.log_softmax(out, dim=1)
                loss = criterion_soft(log_probs, mixed_y)
                with torch.no_grad():
                    out_clean = model(xb)
                    correct += (out_clean.argmax(1) == yb).sum().item()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                correct += (out.argmax(1) == yb).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step(epoch)

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


def train_model_lbfgs(model, X_train, y_train, X_val, y_val,
                      num_epochs=300, lr=1.0, weight_decay=1e-3, device="cpu"):
    """Full-batch LBFGS training for small FC models."""
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20,
                            history_size=50, line_search_fn="strong_wolfe")

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        loss_val = [0.0]

        def closure():
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            l2 = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + weight_decay * l2
            loss.backward()
            loss_val[0] = loss.item()
            return loss

        optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            train_out = model(X_train)
            train_acc = (train_out.argmax(1) == y_train).float().mean().item()
            train_loss = loss_val[0]
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_acc = (val_out.argmax(1) == y_val).float().mean().item()

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
    """Evaluate and print classification report."""
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
# 7. BASELINE MODELS
# ──────────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train_enc, X_test, y_test_enc, label_encoder):
    print("\n" + "=" * 60)
    print("SVM BASELINE (RBF kernel + PCA)")
    print("=" * 60)
    svm_pipe = make_pipeline(
        PCA(n_components=N_PCA if N_PCA else 0.95),
        SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovo")
    )
    svm_pipe.fit(X_train, y_train_enc)
    preds = svm_pipe.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    n_comp = svm_pipe[0].n_components_
    print(f"PCA reduced {X_train.shape[1]} → {n_comp} features")
    print(f"SVM Test Accuracy: {acc:.3f}")
    labels = label_encoder.classes_
    print(classification_report(y_test_enc, preds, target_names=labels, zero_division=0))
    return acc, preds


def train_mlp(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, label_encoder):
    print("\n" + "=" * 60)
    print("MLP BASELINE (scikit-learn)")
    print("=" * 60)
    mlp = make_pipeline(
        PCA(n_components=N_PCA if N_PCA else 0.95),
        MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="lbfgs",
            max_iter=5000,
            alpha=1.0,
            random_state=42,
        )
    )
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train_enc, y_val_enc])
    mlp.fit(X_fit, y_fit)
    preds = mlp.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"MLP Test Accuracy: {acc:.3f}")
    labels = label_encoder.classes_
    print(classification_report(y_test_enc, preds, target_names=labels, zero_division=0))
    return acc, preds


def train_knn(X_train, y_train_enc, X_test, y_test_enc, label_encoder):
    print("\n" + "=" * 60)
    print("KNN BASELINE (k=3 + PCA)")
    print("=" * 60)
    knn_pipe = make_pipeline(
        PCA(n_components=N_PCA if N_PCA else 0.95),
        KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean")
    )
    knn_pipe.fit(X_train, y_train_enc)
    preds = knn_pipe.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"KNN Test Accuracy: {acc:.3f}")
    labels = label_encoder.classes_
    print(classification_report(y_test_enc, preds, target_names=labels, zero_division=0))
    return acc, preds


def print_comparison(results):
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
# 8. LEAVE-ONE-OUT CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────

def cross_validate_svm(records, sg_window=15, sg_polyorder=3):
    """Leave-one-sample-out CV with SVM."""
    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (SVM)")
    print("=" * 60)

    X_all, y_all = [], []
    for r in records:
        _, scaled = preprocess_spectrum(r["intensities"], sg_window, sg_polyorder)
        X_all.append(scaled)
        y_all.append(r["species"])
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    print(f"Using {len(X_all)} samples from {len(le.classes_)} species")

    all_true, all_pred = [], []
    n = len(X_all)
    for i in range(n):
        X_train = np.delete(X_all, i, axis=0)
        y_train = np.delete(y_enc, i)
        X_test = X_all[i:i+1]
        y_test = y_enc[i:i+1]

        svm_pipe = make_pipeline(
            PCA(n_components=N_PCA if N_PCA else 0.95),
            SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovo")
        )
        svm_pipe.fit(X_train, y_train)
        pred = svm_pipe.predict(X_test)
        all_true.append(y_test[0])
        all_pred.append(pred[0])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = accuracy_score(all_true, all_pred)

    print(f"\nLOO-CV Accuracy: {acc:.3f} ({int(acc * n)}/{n} correct)")
    print(classification_report(all_true, all_pred, target_names=le.classes_, zero_division=0))

    print("Per-species accuracy:")
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = all_true == cls_idx
        if mask.sum() > 0:
            cls_acc = (all_pred[mask] == cls_idx).sum() / mask.sum()
            print(f"  {cls_name:<25} {cls_acc:.0%} ({(all_pred[mask] == cls_idx).sum()}/{mask.sum()})")

    return acc


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Step 1: Load all data ──
    print("Loading data from average_data/...")
    all_records = load_all_spectra()
    print(f"Total records loaded: {len(all_records)}")
    print(all_records)
    species_set = sorted(set(r["species"] for r in all_records))
    print(f"Loaded {len(all_records)} spectra from {len(species_set)} species")
    for sp in species_set:
        count = sum(1 for r in all_records if r["species"] == sp)
        print(f"  {sp:<25} {count} samples")

    # ── Step 2: Plot one example ──
    first_species = species_set[0]
    first_rec = [r for r in all_records if r["species"] == first_species][0]
    plot_raw_spectrum(first_rec)
    plot_all_samples_for_species(all_records, first_species)
    plot_preprocessing_comparison(first_rec)

    # ── Step 3: Split ──
    train_recs, val_recs, test_recs = train_val_test_split(all_records)
    print(f"\nSplit: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test")

    # ── Step 4: Build preprocessed arrays ──
    X_train, y_train = build_dataset(train_recs)
    X_val, y_val = build_dataset(val_recs)
    X_test, y_test = build_dataset(test_recs)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    input_length = X_train.shape[1]
    print(f"Classes: {num_classes}, Input length: {input_length}")
    print(f"Species: {list(le.classes_)}")

    # ── Step 5: Prepare data ──
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    X_cnn_train = np.vstack([X_train, X_val])
    y_cnn_train = np.concatenate([y_train_enc, y_val_enc])

    # CNN data
    X_cnn_t, y_cnn_t = make_tensors(X_cnn_train, y_cnn_train, add_channel=True)
    X_test_t, y_test_t = make_tensors(X_test, y_test_enc, add_channel=True)
    cnn_train_loader = DataLoader(TensorDataset(X_cnn_t, y_cnn_t), batch_size=16, shuffle=True)
    cnn_test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    # FC+PCA data
    pca_fc = PCA(n_components=N_PCA if N_PCA else 0.95)
    X_fc_train = pca_fc.fit_transform(X_cnn_train)
    X_fc_test = pca_fc.transform(X_test)
    n_pca = X_fc_train.shape[1]
    print(f"PCA for FC model: {X_train.shape[1]} → {n_pca} features")
    X_fc_t, y_fc_t = make_tensors(X_fc_train, y_cnn_train, add_channel=False)
    X_fc_test_t, y_fc_test_t = make_tensors(X_fc_test, y_test_enc, add_channel=False)
    fc_test_loader = DataLoader(TensorDataset(X_fc_test_t, y_fc_test_t), batch_size=32)

    # ── Step 6a: Train CNN ──
    model_cnn = SpectrumCNN(input_length, num_classes)
    print(f"\nCNN parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    print("Training CNN (mixup + label smoothing + cosine annealing)...")
    model_cnn, history_cnn = train_model(model_cnn, cnn_train_loader, cnn_test_loader,
                                          num_epochs=400, lr=1e-3, device=device,
                                          use_mixup=True, label_smoothing=0.1)
    plot_training_history(history_cnn)
    true_labels, pred_labels = evaluate_model(model_cnn, cnn_test_loader, le, device=device)
    cnn_acc = accuracy_score(true_labels, pred_labels)

    # ── Step 6b: Train FC+PCA (LBFGS) ──
    model_fc = SpectrumFC(n_pca, num_classes)
    print(f"\nFC parameters: {sum(p.numel() for p in model_fc.parameters()):,}")
    print("Training FC+PCA (LBFGS, full-batch)...")
    model_fc, history_fc = train_model_lbfgs(
        model_fc, X_fc_t, y_fc_t, X_fc_test_t, y_fc_test_t,
        num_epochs=300, lr=1.0, weight_decay=1e-3, device="cpu"
    )
    plot_training_history(history_fc)
    true_labels_fc, pred_labels_fc = evaluate_model(model_fc, fc_test_loader, le, device="cpu")
    fc_acc = accuracy_score(true_labels_fc, pred_labels_fc)

    # ── Step 7: SVM baseline ──
    X_train_all = np.vstack([X_train, X_val])
    y_train_all = np.concatenate([y_train_enc, y_val_enc])
    svm_acc, _ = train_svm(X_train_all, y_train_all, X_test, y_test_enc, le)

    # ── Step 8: MLP baseline ──
    mlp_acc, _ = train_mlp(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le)

    # ── Step 9: KNN baseline ──
    knn_acc, _ = train_knn(X_train_all, y_train_all, X_test, y_test_enc, le)

    # ── Step 10: Comparison ──
    print_comparison({
        "1D-CNN": cnn_acc,
        "FC+PCA (PyTorch)": fc_acc,
        "SVM (RBF+PCA)": svm_acc,
        "MLP (sklearn)": mlp_acc,
        "KNN (k=3+PCA)": knn_acc,
    })

    # ── Step 11: LOO-CV ──
    loo_acc = cross_validate_svm(all_records)
