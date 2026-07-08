"""Hybrid Scheme 1 pipeline: adversarially train lightweight IDS models.

This script ties together the GAN/FGSM tooling from lab-ids-anta-main with the
lightweight SGD/Ridge classifiers from the SSRN paper. It loads a dataset,
trains a small DNN to craft FGSM adversarial samples, augments the training set,
and finally evaluates lightweight classifiers on both clean and adversarial
data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import feature selection module
from feature_selection import (
    ForwardSelection,
    BackwardElimination,
    CorrelationBasedSelection,
    ImportanceBasedSelection,
    compare_feature_selection_methods,
)

# Import OWC-SAWN module
from owc_sawn import (
    ConditionalGenerator,
    WeightedDiscriminator,
    OWCSAWNTrainer,
)


def build_dnn(input_dim: int, hidden_layers: List[int], learning_rate: float) -> tf.keras.Model:
    """Create a simple feed-forward network used to craft FGSM samples."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_fgsm_samples(
    model: tf.keras.Model,
    x_np: np.ndarray,
    y_np: np.ndarray,
    epsilon: float,
    clip_value: float,
) -> np.ndarray:
    """Create FGSM adversarial samples for a numpy batch."""
    x_tensor = tf.convert_to_tensor(x_np, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_np.reshape(-1, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor, training=False)
        loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)
    gradient = tape.gradient(loss, x_tensor)
    signed_grad = tf.sign(gradient)
    adv_tensor = x_tensor + epsilon * signed_grad
    if clip_value > 0:
        adv_tensor = tf.clip_by_value(adv_tensor, -clip_value, clip_value)
    return adv_tensor.numpy()


def generate_pgd_samples(
    model: tf.keras.Model,
    x_np: np.ndarray,
    y_np: np.ndarray,
    epsilon: float,
    step_size: float,
    num_steps: int,
    clip_value: float,
    random_start: bool = True,
    batch_size: int = 1024,
) -> np.ndarray:
    """Create PGD (L-infinity) adversarial samples.

    Notes:
        - Operates in the *scaled* feature space (after StandardScaler / feature selection).
        - Uses untargeted loss maximization on the true label.
        - Runs in mini-batches to keep memory bounded.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    x_np = x_np.astype(np.float32, copy=False)
    y_np = y_np.astype(np.int32, copy=False)

    adv_out = np.empty_like(x_np, dtype=np.float32)
    n = x_np.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x0 = tf.convert_to_tensor(x_np[start:end], dtype=tf.float32)
        y = tf.convert_to_tensor(y_np[start:end].reshape(-1, 1), dtype=tf.float32)

        if random_start:
            delta = tf.random.uniform(
                shape=tf.shape(x0),
                minval=-epsilon,
                maxval=epsilon,
                dtype=tf.float32,
            )
            x_adv = x0 + delta
        else:
            x_adv = tf.identity(x0)

        # Project to epsilon-ball around x0 and clip to allowed range.
        x_adv = tf.minimum(tf.maximum(x_adv, x0 - epsilon), x0 + epsilon)
        if clip_value > 0:
            x_adv = tf.clip_by_value(x_adv, -clip_value, clip_value)

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                pred = model(x_adv, training=False)
                loss = tf.keras.losses.binary_crossentropy(y, pred)
            grad = tape.gradient(loss, x_adv)
            x_adv = x_adv + step_size * tf.sign(grad)

            # Project back to the epsilon-ball around the clean input.
            x_adv = tf.minimum(tf.maximum(x_adv, x0 - epsilon), x0 + epsilon)
            if clip_value > 0:
                x_adv = tf.clip_by_value(x_adv, -clip_value, clip_value)

        adv_out[start:end] = x_adv.numpy()
    return adv_out


def train_owc_sawn_for_ids(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 128,
    latent_dim: int = 100,
    generator_lr: float = 0.00005,
    discriminator_lr: float = 0.00005,
) -> OWCSAWNTrainer:
    """Train OWC-SAWN adversarial network for IDS.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save checkpoints and logs
        epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Dimension of latent noise
        generator_lr: Generator learning rate
        discriminator_lr: Discriminator learning rate
    
    Returns:
        Trained OWCSAWNTrainer
    """
    print(f"\n{'='*80}")
    print("Training OWC-SAWN Adversarial Network")
    print(f"{'='*80}")
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    checkpoint_dir = output_dir / "owc_sawn_checkpoints"
    log_dir = output_dir / "owc_sawn_logs"
    
    print(f"\nConfiguration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Create models with simplified architecture
    generator = ConditionalGenerator(
        latent_dim=latent_dim,
        output_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=[128, 256, 128],  # Simpler architecture
        dropout_rate=0.2,  # Less dropout
        embedding_dim=32  # Smaller embedding
    )
    
    discriminator = WeightedDiscriminator(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=[128, 64],  # Much simpler
        dropout_rate=0.2,
        embedding_dim=32,
        use_class_weights=False  # Disable weights initially
    )
    
    # Create trainer with maximum stability
    trainer = OWCSAWNTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=latent_dim,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        beta_1=0.5,
        beta_2=0.999,
        use_gradient_penalty=False,  # Disable for stability
        gp_lambda=0.0,
        n_discriminator_steps=1,  # 1:1 ratio for balance
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir)
    )
    
    # Train
    print(f"\nStarting training...")
    trainer.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("\n✓ OWC-SAWN training completed!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    
    return trainer


def evaluate_model(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    split: str,
    scenario: str,
) -> Dict[str, float]:
    """Collect common metrics for a classifier."""
    y_pred = model.predict(X)
    metrics: Dict[str, float] = {
        "model": name,
        "split": split,
        "scenario": scenario,
        "accuracy": accuracy_score(y, y_pred),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    metrics.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)[:, 1]
        except Exception:  # pragma: no cover - fallback for unexpected shapes
            y_score = None
    elif hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X)
        except Exception:
            y_score = None

    if y_score is not None and len(np.unique(y)) > 1:
        try:
            metrics["auc"] = roc_auc_score(y, y_score)
        except ValueError:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
    return metrics


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "lab-ids-anta-main" / "Dataset" / "normalized_data_2017.csv"
    default_output = Path(__file__).resolve().parent / "results"

    parser = argparse.ArgumentParser(description="Scheme 1 hybrid IDS pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to CSV dataset with a 'Label' column",
    )
    parser.add_argument(
        "--label-column",
        default="Label",
        help="Name of the column containing binary labels",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Perturbation magnitude (used by FGSM/PGD in scaled feature space)",
    )
    parser.add_argument(
        "--clip-value",
        type=float,
        default=3.0,
        help="Clip adversarial samples to +/- this value after scaling (0 disables clipping)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and sklearn models",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Units per hidden layer in the FGSM helper DNN",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs for the FGSM helper DNN",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for the FGSM helper DNN",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the FGSM helper DNN",
    )
    parser.add_argument(
        "--eval-attacks",
        nargs="*",
        default=[],
        choices=["fgsm", "pgd", "transfer_fgsm", "transfer_pgd"],
        help=(
            "Additional attacks to evaluate on the test set. "
            "Writes results to results/metrics_attack_eval.csv"
        ),
    )
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=20,
        help="Number of PGD iterations for --eval-attacks pgd/transfer_pgd",
    )
    parser.add_argument(
        "--pgd-step-size",
        type=float,
        default=None,
        help=(
            "PGD step size (alpha). If omitted, uses 2*epsilon/pgd_steps. "
            "Applies to pgd and transfer_pgd."
        ),
    )
    parser.add_argument(
        "--pgd-random-start",
        action="store_true",
        help="Use random initialization within the epsilon-ball for PGD",
    )
    parser.add_argument(
        "--transfer-hidden-layers",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden units for the surrogate DNN used in transfer attacks",
    )
    parser.add_argument(
        "--transfer-epochs",
        type=int,
        default=10,
        help="Training epochs for the surrogate DNN used in transfer attacks",
    )
    parser.add_argument(
        "--transfer-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the surrogate DNN used in transfer attacks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to store adversarial samples and metrics",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Max iterations for SGDClassifier",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity passed to Keras fit",
    )
    parser.add_argument(
        "--feature-selection",
        type=str,
        default=None,
        choices=["forward", "backward", "correlation", "importance", "compare"],
        help="Feature selection method to use (None to skip feature selection)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20,
        help="Maximum number of features to select",
    )
    parser.add_argument(
        "--fs-cv",
        type=int,
        default=3,
        help="Cross-validation folds for feature selection",
    )
    parser.add_argument(
        "--fs-max-samples",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of training samples used *during feature selection* "
            "(subsamples X_train_scaled/y_train). This can drastically speed up forward/backward selection. "
            "Default: None (use full training set)."
        ),
    )
    parser.add_argument(
        "--adversarial-method",
        type=str,
        default="fgsm",
        choices=["none", "fgsm", "owc-sawn", "both"],
        help="Adversarial sample generation method (none, fgsm, owc-sawn, or both)",
    )
    parser.add_argument(
        "--owc-epochs",
        type=int,
        default=100,
        help="Training epochs for OWC-SAWN",
    )
    parser.add_argument(
        "--owc-latent-dim",
        type=int,
        default=100,
        help="Latent dimension for OWC-SAWN generator",
    )
    parser.add_argument(
        "--owc-augmentation-ratio",
        type=float,
        default=0.5,
        help="Ratio of generated samples to add when using OWC-SAWN (0.5 = add 50%% more samples)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Column '{args.label_column}' not present in dataset")

    # Extract labels first
    y_raw = df[args.label_column].values
    df_features = df.drop(columns=[args.label_column])
    
    # Drop non-numeric columns (timestamps, IDs, IPs, etc.)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_features.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_cols) > 0:
        print(f"\nDropping {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)[:5]}...")
        df_features = df_features[numeric_cols]
    
    X = df_features.values.astype(np.float32)
    
    # Handle label conversion (string to binary or already binary)
    if y_raw.dtype == object or y_raw.dtype.kind == 'U':  # String labels
        print(f"\nDetected string labels, converting to binary...")
        unique_labels = np.unique(y_raw)
        print(f"Unique labels: {unique_labels}")
        # BENIGN/Benign = 0, all attacks = 1
        benign_variants = ['BENIGN', 'Benign', 'benign', 'NORMAL', 'Normal', 'normal']
        y = np.where(np.isin(y_raw, benign_variants), 0, 1).astype(np.int32)
        print(f"Label distribution: BENIGN={np.sum(y==0)}, ATTACK={np.sum(y==1)}")
    else:  # Already numeric
        y = y_raw.astype(np.int32)
    
    print(f"\nDataset info:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Class balance: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    # Keep an immutable copy for augmentation bookkeeping.
    y_train_clean = y_train.copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Feature Selection (optional)
    selected_features = None
    feature_selector = None
    
    if args.feature_selection:
        print(f"\n{'='*80}")
        print(f"Feature Selection: {args.feature_selection}")
        print(f"{'='*80}")
        
        # Create a lightweight classifier for feature selection
        fs_clf = SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=args.random_state,
        )

        X_fs = X_train_scaled
        y_fs = y_train
        if args.fs_max_samples is not None and args.fs_max_samples > 0:
            if X_fs.shape[0] > args.fs_max_samples:
                rng = np.random.default_rng(args.random_state)
                idx = rng.choice(X_fs.shape[0], size=int(args.fs_max_samples), replace=False)
                X_fs = X_fs[idx]
                y_fs = y_fs[idx]
                print(
                    f"Using a feature-selection subsample: {X_fs.shape[0]}/{X_train_scaled.shape[0]} train rows"
                )
        
        if args.feature_selection == "compare":
            # Compare all methods
            fs_results = compare_feature_selection_methods(
                X_train_scaled,
                y_train,
                fs_clf,
                cv=args.fs_cv,
                verbose=True,
            )
            # Use the best method
            best_method = max(fs_results.items(), key=lambda x: x[1]['score'])
            feature_selector = best_method[1]['selector']
            selected_features = best_method[1]['features']
            print(f"\nBest method: {best_method[0]} (score: {best_method[1]['score']:.4f})")
        else:
            # Use specified method
            if args.feature_selection == "forward":
                feature_selector = ForwardSelection(
                    fs_clf,
                    max_features=args.max_features,
                    cv=args.fs_cv,
                    verbose=True,
                )
            elif args.feature_selection == "backward":
                feature_selector = BackwardElimination(
                    fs_clf,
                    min_features=max(1, args.max_features),
                    cv=args.fs_cv,
                    verbose=True,
                )
            elif args.feature_selection == "correlation":
                feature_selector = CorrelationBasedSelection(
                    fs_clf,
                    threshold=0.7,
                    max_features=args.max_features,
                    cv=args.fs_cv,
                    verbose=True,
                )
            elif args.feature_selection == "importance":
                from sklearn.ensemble import RandomForestClassifier
                importance_clf = RandomForestClassifier(
                    n_estimators=50,
                    random_state=args.random_state,
                    n_jobs=-1,
                )
                feature_selector = ImportanceBasedSelection(
                    importance_clf,
                    top_k=args.max_features,
                    cv=args.fs_cv,
                    verbose=True,
                )
            
            feature_selector.fit(X_fs, y_fs)
            selected_features = feature_selector.selected_features_
        
        # Apply feature selection
        print(f"\nSelected {len(selected_features)} features: {selected_features}")
        X_train_scaled = X_train_scaled[:, selected_features]
        X_test_scaled = X_test_scaled[:, selected_features]
        
        # Save feature selection info
        fs_info = {
            "method": args.feature_selection,
            "n_features_original": X.shape[1],
            "n_features_selected": len(selected_features),
            "selected_features": [int(f) for f in selected_features],
        }
        fs_json = output_dir / "feature_selection.json"
        fs_json.write_text(json.dumps(fs_info, indent=2))
        print(f"\nSaved feature selection info to {fs_json}")
    
    X_train_scaled64 = X_train_scaled.astype(np.float64)
    X_test_scaled64 = X_test_scaled.astype(np.float64)

    tf.random.set_seed(args.random_state)
    
    # Adversarial generation + attack evaluation helper models
    X_train_adv: Optional[np.ndarray] = None
    X_test_adv_primary: Optional[np.ndarray] = None
    owc_trainer = None

    need_attacker_dnn = (
        args.adversarial_method in ["fgsm", "both"]
        or any(attack in args.eval_attacks for attack in ["fgsm", "pgd"])
    )
    need_transfer_surrogate = any(
        attack in args.eval_attacks for attack in ["transfer_fgsm", "transfer_pgd"]
    )

    dnn_model = None
    if need_attacker_dnn:
        print(f"\n{'='*80}")
        print("Training helper DNN for gradient-based attacks (FGSM/PGD)")
        print(f"{'='*80}")
        dnn_model = build_dnn(X_train_scaled.shape[1], args.hidden_layers, args.learning_rate)
        dnn_model.fit(
            X_train_scaled,
            y_train_clean,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbosity,
        )

    # Choose adversarial method (training augmentation)
    X_train_adv_fgsm = None
    X_test_adv_fgsm = None
    if args.adversarial_method in ["fgsm", "both"]:
        print(f"\n{'='*80}")
        print("Generating FGSM Adversarial Samples")
        print(f"{'='*80}")
        if dnn_model is None:
            raise RuntimeError("Internal error: dnn_model is required for FGSM")

        X_train_adv_fgsm = generate_fgsm_samples(
            dnn_model,
            X_train_scaled,
            y_train_clean,
            epsilon=args.epsilon,
            clip_value=args.clip_value,
        )
        X_test_adv_fgsm = generate_fgsm_samples(
            dnn_model,
            X_test_scaled,
            y_test,
            epsilon=args.epsilon,
            clip_value=args.clip_value,
        )
        np.save(output_dir / "X_train_adv_fgsm.npy", X_train_adv_fgsm)
        np.save(output_dir / "X_test_adv_fgsm.npy", X_test_adv_fgsm)
        if args.adversarial_method == "fgsm":
            X_train_adv = X_train_adv_fgsm
            X_test_adv_primary = X_test_adv_fgsm
    
    if args.adversarial_method in ["owc-sawn", "both"]:
        print(f"\n{'='*80}")
        print("Training OWC-SAWN and Generating Adversarial Samples")
        print(f"{'='*80}")
        
        # Split validation set for OWC-SAWN
        X_train_owc, X_val_owc, y_train_owc, y_val_owc = train_test_split(
            X_train_scaled,
            y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=args.random_state,
        )
        
        # Additional normalization to [-1, 1] for GAN stability
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_owc_norm = minmax_scaler.fit_transform(X_train_owc)
        X_val_owc_norm = minmax_scaler.transform(X_val_owc)
        X_train_norm_for_gen = minmax_scaler.transform(X_train_scaled)
        X_test_norm_for_gen = minmax_scaler.transform(X_test_scaled)
        
        print(f"Data range after MinMax normalization: [{X_train_owc_norm.min():.2f}, {X_train_owc_norm.max():.2f}]")
        
        # Train OWC-SAWN
        owc_trainer = train_owc_sawn_for_ids(
            X_train_owc_norm,
            y_train_owc,
            X_val_owc_norm,
            y_val_owc,
            output_dir,
            epochs=args.owc_epochs,
            batch_size=args.batch_size,
            latent_dim=args.owc_latent_dim,
        )
        
        # Generate adversarial samples with OWC-SAWN
        print(f"\nGenerating adversarial samples with OWC-SAWN...")
        X_train_adv_owc_norm = owc_trainer.generate_samples(
            num_samples=len(X_train_scaled),
            labels=y_train
        )
        X_test_adv_owc_norm = owc_trainer.generate_samples(
            num_samples=len(X_test_scaled),
            labels=y_test
        )
        
        # Transform back to original StandardScaler scale
        X_train_adv_owc = minmax_scaler.inverse_transform(X_train_adv_owc_norm)
        X_test_adv_owc = minmax_scaler.inverse_transform(X_test_adv_owc_norm)
        
        np.save(output_dir / "X_train_adv_owc.npy", X_train_adv_owc)
        np.save(output_dir / "X_test_adv_owc.npy", X_test_adv_owc)

        if args.adversarial_method == "owc-sawn":
            X_train_adv = X_train_adv_owc
            X_test_adv_primary = X_test_adv_owc
        elif args.adversarial_method == "both":
            if X_train_adv_fgsm is None:
                raise RuntimeError("Internal error: FGSM samples missing in 'both' mode")
            # Combine FGSM and OWC-SAWN samples for training augmentation.
            X_train_adv = np.vstack([X_train_adv_fgsm, X_train_adv_owc])
            # Keep metrics.csv compatible: use FGSM as the primary adv test set.
            X_test_adv_primary = X_test_adv_fgsm
            print(
                f"\nCombined FGSM + OWC-SAWN for training: {X_train_adv.shape[0]} adversarial samples"
            )

    if args.adversarial_method != "none":
        if X_train_adv is None or X_test_adv_primary is None:
            raise RuntimeError("Failed to generate adversarial samples; check --adversarial-method")

    X_train_aug = None
    y_train_aug = None
    X_test_adv64 = None
    if args.adversarial_method != "none":
        X_train_aug = np.vstack([X_train_scaled, X_train_adv]).astype(np.float64)
        n_clean = y_train_clean.shape[0]
        n_adv = X_train_adv.shape[0]
        if n_adv % n_clean != 0:
            raise ValueError(
                f"Adversarial training samples count {n_adv} is not a multiple of clean count {n_clean}."
            )
        repeat_adv = n_adv // n_clean
        y_train_aug = np.concatenate([y_train_clean] * (1 + repeat_adv))
        X_test_adv64 = X_test_adv_primary.astype(np.float64)

    baseline_sgd = SGDClassifier(
        loss="log_loss",
        max_iter=args.max_iter,
        tol=1e-3,
        random_state=args.random_state,
    )
    baseline_sgd.fit(X_train_scaled64, y_train_clean)

    baseline_ridge = RidgeClassifier()
    baseline_ridge.fit(X_train_scaled64, y_train_clean)

    sgd = None
    ridge = None
    if args.adversarial_method != "none":
        sgd = SGDClassifier(
            loss="log_loss",
            max_iter=args.max_iter,
            tol=1e-3,
            random_state=args.random_state,
        )
        sgd.fit(X_train_aug, y_train_aug)

        ridge = RidgeClassifier()
        ridge.fit(X_train_aug, y_train_aug)

    metrics: List[Dict[str, float]] = []
    # Scenario 1: Clean training -> Clean test (no attack)
    for model_name, clf in [
        ("SGDClassifier", baseline_sgd),
        ("RidgeClassifier", baseline_ridge),
    ]:
        metrics.append(
            evaluate_model(
                model_name,
                clf,
                X_test_scaled64,
                y_test,
                split="clean_test",
                scenario="baseline_clean_train_clean_test",
            )
        )

    if args.adversarial_method != "none":
        # Scenario 2: Clean training -> Adversarial test (attack without defense)
        for model_name, clf in [
            ("SGDClassifier", baseline_sgd),
            ("RidgeClassifier", baseline_ridge),
        ]:
            metrics.append(
                evaluate_model(
                    model_name,
                    clf,
                    X_test_adv64,
                    y_test,
                    split="adv_test",
                    scenario="baseline_clean_train_adv_test",
                )
            )

        # Scenario 3: Adversarial training -> both clean/adversarial test
        for split_name, data in [
            ("clean_test", X_test_scaled64),
            ("adv_test", X_test_adv64),
        ]:
            for model_name, clf in [
                ("SGDClassifier", sgd),
                ("RidgeClassifier", ridge),
            ]:
                metrics.append(
                    evaluate_model(
                        model_name,
                        clf,
                        data,
                        y_test,
                        split=split_name,
                        scenario=f"adv_train_{split_name}",
                    )
                )

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = output_dir / "metrics.csv"
    metrics_json = output_dir / "metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2))

    if args.adversarial_method != "none":
        np.save(output_dir / "X_train_adv.npy", X_train_adv)
        np.save(output_dir / "X_test_adv.npy", X_test_adv_primary)
    np.save(output_dir / "y_train.npy", y_train_clean)
    np.save(output_dir / "y_test.npy", y_test)

    # Optional: PGD + black-box transfer (surrogate) evaluation suite
    if args.eval_attacks:
        print(f"\n{'='*80}")
        print("Running attack evaluation suite")
        print(f"{'='*80}")

        pgd_step_size = args.pgd_step_size
        if pgd_step_size is None:
            pgd_step_size = (2.0 * args.epsilon) / max(args.pgd_steps, 1)

        surrogate_model = None
        if need_transfer_surrogate:
            tf.random.set_seed(args.random_state + 1)
            surrogate_model = build_dnn(
                X_train_scaled.shape[1],
                args.transfer_hidden_layers,
                args.transfer_learning_rate,
            )
            surrogate_model.fit(
                X_train_scaled,
                y_train_clean,
                validation_split=0.1,
                epochs=args.transfer_epochs,
                batch_size=args.batch_size,
                verbose=args.verbosity,
            )

        attack_sets: Dict[str, np.ndarray] = {"clean": X_test_scaled}
        for attack in args.eval_attacks:
            if attack == "fgsm":
                if dnn_model is None:
                    raise RuntimeError("FGSM evaluation requested but helper DNN is missing")
                attack_sets[attack] = generate_fgsm_samples(
                    dnn_model,
                    X_test_scaled,
                    y_test,
                    epsilon=args.epsilon,
                    clip_value=args.clip_value,
                )
            elif attack == "pgd":
                if dnn_model is None:
                    raise RuntimeError("PGD evaluation requested but helper DNN is missing")
                attack_sets[attack] = generate_pgd_samples(
                    dnn_model,
                    X_test_scaled,
                    y_test,
                    epsilon=args.epsilon,
                    step_size=float(pgd_step_size),
                    num_steps=int(args.pgd_steps),
                    clip_value=args.clip_value,
                    random_start=bool(args.pgd_random_start),
                    batch_size=int(args.batch_size),
                )
            elif attack == "transfer_fgsm":
                if surrogate_model is None:
                    raise RuntimeError("transfer_fgsm requested but surrogate model is missing")
                attack_sets[attack] = generate_fgsm_samples(
                    surrogate_model,
                    X_test_scaled,
                    y_test,
                    epsilon=args.epsilon,
                    clip_value=args.clip_value,
                )
            elif attack == "transfer_pgd":
                if surrogate_model is None:
                    raise RuntimeError("transfer_pgd requested but surrogate model is missing")
                attack_sets[attack] = generate_pgd_samples(
                    surrogate_model,
                    X_test_scaled,
                    y_test,
                    epsilon=args.epsilon,
                    step_size=float(pgd_step_size),
                    num_steps=int(args.pgd_steps),
                    clip_value=args.clip_value,
                    random_start=bool(args.pgd_random_start),
                    batch_size=int(args.batch_size),
                )

        # Persist adversarial arrays for later analysis
        for attack_name, x_adv in attack_sets.items():
            if attack_name == "clean":
                continue
            np.save(output_dir / f"X_test_adv_{attack_name}.npy", x_adv)

        attack_metrics: List[Dict[str, float]] = []
        clf_groups = [
            ("baseline", [("SGDClassifier", baseline_sgd), ("RidgeClassifier", baseline_ridge)]),
        ]
        if args.adversarial_method != "none":
            clf_groups.append(("adv_train", [("SGDClassifier", sgd), ("RidgeClassifier", ridge)]))

        for attack_name, x_adv in attack_sets.items():
            x_eval = x_adv.astype(np.float64)
            split_name = "clean_test" if attack_name == "clean" else "adv_test"
            for train_regime, models in clf_groups:
                for model_name, clf in models:
                    row = evaluate_model(
                        model_name,
                        clf,
                        x_eval,
                        y_test,
                        split=split_name,
                        scenario=f"attack_eval_{train_regime}",
                    )
                    row.update(
                        {
                            "attack": attack_name,
                            "train_regime": train_regime,
                            "epsilon": float(args.epsilon),
                            "pgd_steps": int(args.pgd_steps) if "pgd" in attack_name else 0,
                            "pgd_step_size": float(pgd_step_size) if "pgd" in attack_name else 0.0,
                            "pgd_random_start": bool(args.pgd_random_start) if "pgd" in attack_name else False,
                            "surrogate": "dnn" if attack_name in ["fgsm", "pgd"] else ("surrogate_dnn" if attack_name.startswith("transfer_") else ""),
                        }
                    )
                    attack_metrics.append(row)

        attack_metrics_df = pd.DataFrame(attack_metrics)
        attack_csv = output_dir / "metrics_attack_eval.csv"
        attack_json = output_dir / "metrics_attack_eval.json"
        attack_metrics_df.to_csv(attack_csv, index=False)
        attack_json.write_text(json.dumps(attack_metrics, indent=2))

        attack_cfg = {
            "eval_attacks": list(args.eval_attacks),
            "epsilon": float(args.epsilon),
            "clip_value": float(args.clip_value),
            "pgd_steps": int(args.pgd_steps),
            "pgd_step_size": float(pgd_step_size),
            "pgd_random_start": bool(args.pgd_random_start),
            "transfer_hidden_layers": list(args.transfer_hidden_layers),
            "transfer_epochs": int(args.transfer_epochs),
            "transfer_learning_rate": float(args.transfer_learning_rate),
        }
        (output_dir / "attack_eval_config.json").write_text(json.dumps(attack_cfg, indent=2))
        print("\nSaved attack eval metrics to", attack_csv)
    
    # Save OWC-SAWN info if used
    if owc_trainer is not None:
        owc_info = {
            "method": "owc-sawn",
            "epochs": args.owc_epochs,
            "latent_dim": args.owc_latent_dim,
            "training_history": {
                "gen_loss": [float(x) for x in owc_trainer.training_history['gen_loss']],
                "disc_loss": [float(x) for x in owc_trainer.training_history['disc_loss']],
                "disc_real_acc": [float(x) for x in owc_trainer.training_history['disc_real_acc']],
                "disc_fake_acc": [float(x) for x in owc_trainer.training_history['disc_fake_acc']],
            },
            "final_metrics": {
                "gen_loss": float(owc_trainer.training_history['gen_loss'][-1]),
                "disc_loss": float(owc_trainer.training_history['disc_loss'][-1]),
            }
        }
        owc_json = output_dir / "owc_sawn_info.json"
        owc_json.write_text(json.dumps(owc_info, indent=2))
        print(f"\nSaved OWC-SAWN info to {owc_json}")

    print("\nSaved metrics to", metrics_csv)
    print(metrics_df)


if __name__ == "__main__":
    main()
