#!/usr/bin/env python3
"""GRU training pipeline for tennis shot classification."""

import json
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TRAINING_DATA_DIR, MODELS_DIR, MODEL, SHOT_TYPES


# ── Helpers ──────────────────────────────────────────────────


def format_size(num_bytes):
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


# ── Data Discovery ───────────────────────────────────────────


def discover_training_data(training_dir, shot_types):
    """Find clip JSON files per shot type.

    Returns dict like {"forehand": [path1, path2, ...], ...}.
    Warns if any type has fewer than 50 clips.
    """
    data = {}
    for shot_type in shot_types:
        type_dir = os.path.join(training_dir, shot_type)
        if not os.path.isdir(type_dir):
            data[shot_type] = []
            continue
        files = sorted([
            os.path.join(type_dir, f)
            for f in os.listdir(type_dir)
            if f.endswith(".json")
        ])
        data[shot_type] = files
    return data


def load_clips(file_paths):
    """Read clip JSON files and validate structure.

    Returns list of parsed clip dicts. Skips invalid files with warnings.
    """
    clips = []
    for path in file_paths:
        try:
            with open(path, "r") as f:
                clip = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Cannot read {os.path.basename(path)}: {e}")
            continue

        # Validate required fields
        required = ("version", "shot_type", "start_frame", "end_frame", "frames")
        missing = [k for k in required if k not in clip]
        if missing:
            print(f"  [WARN] {os.path.basename(path)} missing keys: {missing}")
            continue

        if not clip["frames"]:
            print(f"  [WARN] {os.path.basename(path)} has no frames")
            continue

        clips.append(clip)
    return clips


# ── Array Conversion ─────────────────────────────────────────


def clips_to_arrays(clips_by_type, seq_len, shot_types):
    """Convert clip dicts to numpy arrays.

    Returns:
        X: numpy array shape (N, seq_len, 99)
        y: numpy array shape (N,) with integer labels
        label_map: dict mapping shot_type string to integer
    """
    import numpy as np

    label_map = {t: i for i, t in enumerate(shot_types)}

    X_list = []
    y_list = []

    for shot_type, clips in clips_by_type.items():
        label = label_map[shot_type]
        for clip in clips:
            frames = clip["frames"]

            # Pad or truncate to seq_len
            frame_data = []
            for frame in frames[:seq_len]:
                landmarks = frame.get("world_landmarks_xyz", [])
                # Flatten 33 keypoints × 3 coords = 99 features
                flat = []
                for kp in landmarks:
                    flat.extend(kp[:3])

                # Pad if fewer than 99 features
                while len(flat) < 99:
                    flat.append(0.0)
                frame_data.append(flat[:99])

            # Pad with zeros if fewer frames than seq_len
            while len(frame_data) < seq_len:
                frame_data.append([0.0] * 99)

            X_list.append(frame_data)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y, label_map


# ── Normalization ────────────────────────────────────────────


def compute_normalization(X):
    """Compute per-feature mean and std across the training set.

    X shape: (N, seq_len, n_features)
    Returns (mean, std) each shape (n_features,)
    """
    import numpy as np

    # Reshape to (N * seq_len, n_features) for per-feature stats
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # Avoid division by zero
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X, mean, std):
    """Apply z-score normalization: (X - mean) / std."""
    return (X - mean) / std


# ── Model Building ───────────────────────────────────────────


def build_gru_model(seq_len, n_features, n_classes, config):
    """Build and compile a 2-layer GRU model.

    Architecture:
        Input(seq_len, n_features)
        → GRU(hidden_units, return_sequences=True, dropout)
        → GRU(hidden_units, return_sequences=False, dropout)
        → Dense(64, relu)
        → Dropout(dropout)
        → Dense(n_classes, softmax)
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.GRU(
            config["hidden_units"],
            return_sequences=True,
            dropout=config["dropout"],
        ),
        layers.GRU(
            config["hidden_units"],
            return_sequences=False,
            dropout=config["dropout"],
        ),
        layers.Dense(64, activation="relu"),
        layers.Dropout(config["dropout"]),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ── Training ─────────────────────────────────────────────────


def train_model(model, X_train, y_train, X_val, y_val, config):
    """Train with early stopping and class weights.

    Returns keras History object.
    """
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping

    # Compute class weights for imbalanced data
    unique_classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(unique_classes)
    class_weight = {}
    for c in unique_classes:
        count = np.sum(y_train == c)
        class_weight[int(c)] = n_samples / (n_classes * count)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=config["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    return history


# ── Evaluation ───────────────────────────────────────────────


def evaluate_model(model, X_val, y_val, label_map):
    """Run classification report and confusion matrix.

    Returns dict with val_loss, val_accuracy, classification_report, confusion_matrix.
    """
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    inverse_map = {v: k for k, v in label_map.items()}
    target_names = [inverse_map[i] for i in range(len(label_map))]

    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    report = classification_report(
        y_val, y_pred_classes,
        target_names=target_names,
        output_dict=False,
        zero_division=0,
    )

    report_dict = classification_report(
        y_val, y_pred_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_val, y_pred_classes)

    return {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "classification_report": report,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names,
    }


# ── Model Saving ─────────────────────────────────────────────


def save_model(model, path, normalization, label_map, history, eval_results):
    """Save .h5 model and metadata JSON with atomic writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model (.h5) — save directly; Keras rejects non-.h5 extensions
    model.save(path)

    # Save metadata JSON
    meta_path = path.replace(".h5", "_meta.json")
    meta_part = meta_path + ".part"

    inverse_label_map = {str(v): k for k, v in label_map.items()}
    mean, std = normalization

    # Convert history to serializable lists
    training_history = {}
    for key, values in history.history.items():
        training_history[key] = [float(v) for v in values]

    # Find best epoch (lowest val_loss)
    val_losses = training_history.get("val_loss", [])
    best_epoch = int(val_losses.index(min(val_losses))) + 1 if val_losses else 0

    metadata = {
        "version": 1,
        "model_file": os.path.basename(path),
        "label_map": label_map,
        "inverse_label_map": inverse_label_map,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "sequence_length": model.input_shape[1],
        "num_features": model.input_shape[2],
        "training_stats": {
            "total_clips": int(len(history.history.get("loss", [0])) and
                               history.params.get("steps", 0) *
                               history.params.get("epochs", 0)) or 0,
            "best_epoch": best_epoch,
            "final_val_accuracy": eval_results["val_accuracy"],
        },
        "training_history": training_history,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_part, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(meta_part, meta_path)

    return path, meta_path


# ── Summary ──────────────────────────────────────────────────


def print_training_summary(history, eval_results, elapsed, model_path, total_clips):
    """Print final training summary."""
    val_acc = eval_results["val_accuracy"]
    val_loss = eval_results["val_loss"]

    # Best epoch
    val_losses = history.history.get("val_loss", [])
    best_epoch = val_losses.index(min(val_losses)) + 1 if val_losses else 0
    total_epochs = len(val_losses)

    print()
    print("=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"  Total clips:      {total_clips}")
    print(f"  Epochs:           {total_epochs} (best: {best_epoch})")
    print(f"  Training time:    {format_duration(elapsed)}")
    print(f"  Val loss:         {val_loss:.4f}")
    print(f"  Val accuracy:     {val_acc:.1%}")
    print()

    # Classification report
    print("Per-class metrics:")
    print(eval_results["classification_report"])

    # Confusion matrix
    print("Confusion matrix:")
    names = eval_results["target_names"]
    cm = eval_results["confusion_matrix"]
    header = "          " + "  ".join(f"{n[:6]:>6}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {names[i]:<8} {row_str}")
    print()

    # Model output
    model_size = os.path.getsize(model_path)
    meta_path = model_path.replace(".h5", "_meta.json")
    meta_size = os.path.getsize(meta_path) if os.path.exists(meta_path) else 0

    print(f"  Model:    {os.path.basename(model_path)} ({format_size(model_size)})")
    print(f"  Metadata: {os.path.basename(meta_path)} ({format_size(meta_size)})")

    if val_acc >= 0.85:
        print(f"\n  [OK] Validation accuracy {val_acc:.1%} meets 85% target")
    else:
        print(f"\n  [WARN] Validation accuracy {val_acc:.1%} below 85% target")
        print("  Consider labeling more clips or adjusting hyperparameters.")
    print()


# ── Main ─────────────────────────────────────────────────────


def main():
    # ── Startup checks ────────────────────────────────────────
    print("=" * 50)
    print("Tennis Shot Classifier — GRU Training")
    print("=" * 50)
    print()

    # Check imports
    try:
        import numpy as np
    except ImportError:
        print("[ERROR] numpy not available")
        sys.exit(1)

    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"GPU: {gpus[0].name}")
        else:
            print("GPU: None (training on CPU)")
    except ImportError:
        print("[ERROR] TensorFlow not available")
        sys.exit(1)

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("[ERROR] scikit-learn not available")
        sys.exit(1)

    print()

    config = MODEL
    seq_len = config["sequence_length"]

    # ── Discover training data ────────────────────────────────
    print("Discovering training data...")
    data_by_type = discover_training_data(TRAINING_DATA_DIR, SHOT_TYPES)

    total_clips = 0
    has_empty = False
    for shot_type in SHOT_TYPES:
        count = len(data_by_type[shot_type])
        total_clips += count
        status = ""
        if count == 0:
            status = " [ERROR: no clips]"
            has_empty = True
        elif count < 50:
            status = " [WARN: <50 clips]"
        print(f"  {shot_type:<12} {count:>4} clips{status}")

    print(f"  {'Total':<12} {total_clips:>4} clips")
    print()

    if total_clips == 0:
        print("[ERROR] No training data found.")
        sys.exit(1)

    # Filter to shot types that have clips
    active_types = [t for t in SHOT_TYPES if len(data_by_type[t]) > 0]
    if has_empty:
        print(f"  Skipping empty types, training on: {active_types}")
    print()

    # ── Load clips ────────────────────────────────────────────
    print("Loading clips...")
    clips_by_type = {}
    for shot_type in active_types:
        clips = load_clips(data_by_type[shot_type])
        clips_by_type[shot_type] = clips
        if len(clips) != len(data_by_type[shot_type]):
            print(f"  [WARN] {shot_type}: loaded {len(clips)}/{len(data_by_type[shot_type])}")

    # ── Convert to arrays ─────────────────────────────────────
    print("Converting to arrays...")
    X, y, label_map = clips_to_arrays(clips_by_type, seq_len, active_types)
    print(f"  X shape: {X.shape}  y shape: {y.shape}")
    print(f"  Label map: {label_map}")
    print()

    n_classes = len(active_types)
    if n_classes < 2:
        print("[ERROR] Need at least 2 shot types with clips to train a classifier.")
        print(f"  Currently have: {active_types}")
        print("  Label some neutral segments (gaps between serves) or other shot types.")
        sys.exit(1)

    # ── Normalize ─────────────────────────────────────────────
    print("Computing normalization...")

    # Stratified split first (normalize only on training set)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["validation_split"],
        stratify=y,
        random_state=42,
    )
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

    mean, std = compute_normalization(X_train)
    X_train = normalize(X_train, mean, std)
    X_val = normalize(X_val, mean, std)
    print(f"  Normalized (z-score, per-feature)")
    print()

    # ── Build model ───────────────────────────────────────────
    print("Building GRU model...")
    n_classes = len(active_types)
    model = build_gru_model(seq_len, 99, n_classes, config)
    model.summary()
    print()

    # ── Train ─────────────────────────────────────────────────
    print("Training...")
    start_time = time.time()
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {format_duration(elapsed)}")

    # ── Evaluate ──────────────────────────────────────────────
    print("\nEvaluating...")
    eval_results = evaluate_model(model, X_val, y_val, label_map)

    # ── Save ──────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "shot_classifier.h5")
    print(f"\nSaving model to {model_path}...")
    save_model(model, model_path, (mean, std), label_map, history, eval_results)

    # ── Summary ───────────────────────────────────────────────
    print_training_summary(history, eval_results, elapsed, model_path, total_clips)


if __name__ == "__main__":
    main()
