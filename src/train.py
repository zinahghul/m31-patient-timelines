import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)

from tqdm import tqdm

from dataset import prepare_dataloaders
from model import EHRTransformer


# ============================================================
# Configuration
# ============================================================

SEED = 42
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED):
    """
    Best-effort reproducibility.

    Note:
        Full bitwise reproducibility is not guaranteed across
        different CUDA/PyTorch/hardware configurations.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Only enable this if strict reproducibility is required.
    # It can significantly reduce performance.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Binary focal loss with optional positive-class weighting.

    logits:
        [B, C]

    targets:
        [B, C]
    """

    def __init__(
        self,
        pos_weight=None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'"
            )

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                pos_weight.float()
            )
        else:
            self.pos_weight = None

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        targets = targets.float()

        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Probability assigned to the true class
        probs = torch.sigmoid(logits)

        p_t = (
            probs * targets
            + (1.0 - probs) * (1.0 - targets)
        )

        focal_factor = (1.0 - p_t).pow(self.gamma)

        loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping based on a monitored metric.

    Designed for metrics where higher is better, e.g. AUROC.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-3,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, score: float):

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False


# ============================================================
# Metrics
# ============================================================

def compute_metrics(labels, predictions):
    """
    Computes macro metrics while safely handling classes
    containing only one label.
    """

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    aucs = []
    aps = []
    f1s = []
    briers = []

    per_class = {}

    n_classes = labels.shape[1]

    for class_idx in range(n_classes):

        y_true = labels[:, class_idx]
        y_prob = predictions[:, class_idx]

        # Remove invalid values.
        valid = np.isfinite(y_true) & np.isfinite(y_prob)

        y_true = y_true[valid]
        y_prob = y_prob[valid]

        # AUC/AP require both positive and negative examples.
        if len(np.unique(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        y_pred = (y_prob >= 0.5).astype(np.int32)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        brier = brier_score_loss(
            y_true,
            y_prob,
        )

        aucs.append(auc)
        aps.append(ap)
        f1s.append(f1)
        briers.append(brier)

        per_class[f"class_{class_idx}_auc"] = auc
        per_class[f"class_{class_idx}_ap"] = ap
        per_class[f"class_{class_idx}_f1"] = f1
        per_class[f"class_{class_idx}_brier"] = brier

    results = {
        "val_auc": float(np.mean(aucs)) if aucs else 0.0,
        "val_map": float(np.mean(aps)) if aps else 0.0,
        "val_f1": float(np.mean(f1s)) if f1s else 0.0,
        "val_brier": float(np.mean(briers)) if briers else 0.0,
        "valid_classes": len(aucs),
    }

    results.update(per_class)

    return results


# ============================================================
# Class Weight Calculation
# ============================================================

def calculate_pos_weights(train_loader, device):
    """
    Calculates positive-class weights from the training set.

    IMPORTANT:
        This assumes NaN labels represent missing values.
        Missing labels are NOT treated as negative examples.
    """

    positive_counts = None
    negative_counts = None

    print("Calculating class weights...")

    for batch in tqdm(
        train_loader,
        desc="Scanning training labels",
    ):

        labels = batch["labels"].float()

        # Valid labels only.
        valid = torch.isfinite(labels)

        positives = (
            (labels == 1) & valid
        ).sum(dim=0)

        negatives = (
            (labels == 0) & valid
        ).sum(dim=0)

        if positive_counts is None:
            positive_counts = positives
            negative_counts = negatives
        else:
            positive_counts += positives
            negative_counts += negatives

    # Avoid division by zero.
    pos_weights = (
        negative_counts.float()
        / positive_counts.float().clamp(min=1.0)
    )

    print(
        f"Positive counts: {positive_counts.tolist()}"
    )

    print(
        f"Negative counts: {negative_counts.tolist()}"
    )

    print(
        f"Positive weights: {pos_weights.tolist()}"
    )

    return pos_weights.to(device)


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    config,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict()
        if scaler is not None
        else None,
        "metrics": metrics,
        "config": dict(config),
        "seed": SEED,
    }

    torch.save(checkpoint, path)


# ============================================================
# Training
# ============================================================

def train_model():

    set_seed(SEED)

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    wandb.init(
        project="m31-patient-timelines-public",
        config={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 35,
            "d_model": 128,
            "max_seq_len": 512,
            "weight_decay": 1e-4,
            "focal_gamma": 2.0,
            "gradient_clip": 1.0,
            "early_stopping_patience": 5,
            "scheduler_patience": 2,
            "mixed_precision": True,
            "seed": SEED,
        },
    )

    config = wandb.config

    try:

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        use_amp = device.type == "cuda"

        print(f"\nDevice: {device}")
        print(f"Mixed precision: {use_amp}")

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        train_loader, val_loader = prepare_dataloaders(
            batch_size=config.batch_size
        )

        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        pos_weights = calculate_pos_weights(
            train_loader,
            device,
        )

        # ----------------------------------------------------
        # Determine number of classes
        # ----------------------------------------------------

        first_batch = next(iter(train_loader))

        num_classes = first_batch["labels"].shape[1]

        print(f"Number of classes: {num_classes}")

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = EHRTransformer(
            num_classes=num_classes,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        ).to(device)

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        criterion = FocalLoss(
            pos_weight=pos_weights,
            gamma=config.focal_gamma,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=1e-7,
        )

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=1e-3,
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        best_val_auc = -np.inf

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------

        for epoch in range(config.epochs):

            print(
                f"\n{'=' * 70}\n"
                f"Epoch {epoch + 1}/{config.epochs}\n"
                f"{'=' * 70}"
            )

            # =================================================
            # TRAIN
            # =================================================

            model.train()

            running_train_loss = 0.0
            train_batches = 0

            train_bar = tqdm(
                train_loader,
                desc="Training",
                leave=False,
            )

            for batch in train_bar:

                input_ids = batch["input_ids"].to(
                    device,
                    non_blocking=True,
                )

                attention_mask = batch[
                    "attention_mask"
                ].to(
                    device,
                    non_blocking=True,
                )

                labels = batch["labels"].to(
                    device,
                    non_blocking=True,
                ).float()

                # ------------------------------------------------
                # Handle missing labels properly
                # ------------------------------------------------

                valid_labels = torch.isfinite(labels)

                safe_labels = torch.nan_to_num(
                    labels,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):

                    logits = model(
                        input_ids,
                        attention_mask=attention_mask,
                    )

                    # Calculate element-wise loss.
                    loss_matrix = criterion(
                        logits,
                        safe_labels,
                    )

                    # Ignore missing labels.
                    if valid_labels.any():

                        # If criterion returns scalar because of
                        # reduction='mean', use standard loss.
                        if loss_matrix.ndim == 0:
                            loss = loss_matrix
                        else:
                            loss = loss_matrix[
                                valid_labels
                            ].mean()

                    else:
                        continue

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.gradient_clip,
                )

                scaler.step(optimizer)
                scaler.update()

                running_train_loss += loss.item()
                train_batches += 1

                train_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            avg_train_loss = (
                running_train_loss / max(train_batches, 1)
            )

            # =================================================
            # VALIDATION
            # =================================================

            model.eval()

            running_val_loss = 0.0
            val_batches = 0

            all_predictions = []
            all_labels = []

            with torch.inference_mode():

                val_bar = tqdm(
                    val_loader,
                    desc="Validation",
                    leave=False,
                )

                for batch in val_bar:

                    input_ids = batch[
                        "input_ids"
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    attention_mask = batch[
                        "attention_mask"
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    labels = batch[
                        "labels"
                    ].to(
                        device,
                        non_blocking=True,
                    ).float()

                    valid_labels = torch.isfinite(
                        labels
                    )

                    safe_labels = torch.nan_to_num(
                        labels,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):

                        logits = model(
                            input_ids,
                            attention_mask=attention_mask,
                        )

                        loss = criterion(
                            logits,
                            safe_labels,
                        )

                    running_val_loss += loss.item()
                    val_batches += 1

                    probabilities = torch.sigmoid(
                        logits
                    )

                    all_predictions.append(
                        probabilities.cpu().float().numpy()
                    )

                    # Preserve NaNs for metric masking.
                    all_labels.append(
                        labels.cpu().float().numpy()
                    )

            avg_val_loss = (
                running_val_loss
                / max(val_batches, 1)
            )

            # =================================================
            # Metrics
            # =================================================

            all_predictions = np.concatenate(
                all_predictions,
                axis=0,
            )

            all_labels = np.concatenate(
                all_labels,
                axis=0,
            )

            metrics = compute_metrics(
                all_labels,
                all_predictions,
            )

            # Scheduler monitors AUROC.
            scheduler.step(
                metrics["val_auc"]
            )

            current_lr = optimizer.param_groups[0]["lr"]

            # =================================================
            # Logging
            # =================================================

            print(
                f"\nEpoch {epoch + 1}"
                f" | Train Loss: {avg_train_loss:.4f}"
                f" | Val Loss: {avg_val_loss:.4f}"
            )

            print(
                f"AUROC: {metrics['val_auc']:.4f}"
                f" | mAP: {metrics['val_map']:.4f}"
                f" | F1: {metrics['val_f1']:.4f}"
                f" | Brier: {metrics['val_brier']:.4f}"
            )

            print(
                f"LR: {current_lr:.2e}"
                f" | Valid classes: "
                f"{metrics['valid_classes']}"
            )

            wandb_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_macro_auc": metrics["val_auc"],
                "val_mean_ap": metrics["val_map"],
                "val_macro_f1": metrics["val_f1"],
                "val_brier_score": metrics["val_brier"],
                "learning_rate": current_lr,
            }

            # Add per-class metrics.
            wandb_metrics.update(
                {
                    k: v
                    for k, v in metrics.items()
                    if k.startswith("class_")
                }
            )

            wandb.log(wandb_metrics)

            # =================================================
            # Best model
            # =================================================

            improved = (
                metrics["val_auc"]
                > best_val_auc
            )

            if improved:

                best_val_auc = metrics[
                    "val_auc"
                ]

                save_checkpoint(
                    CHECKPOINT_DIR
                    / "best_model.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    metrics,
                    config,
                )

                print(
                    f"\n✓ New best model saved "
                    f"(AUROC = {best_val_auc:.4f})"
                )

            # =================================================
            # Latest checkpoint
            # =================================================

            save_checkpoint(
                CHECKPOINT_DIR
                / "latest_model.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                config,
            )

            # =================================================
            # Early stopping
            # =================================================

            if early_stopping.step(
                metrics["val_auc"]
            ):

                print(
                    f"\nEarly stopping triggered "
                    f"at epoch {epoch + 1}."
                )

                break

        print("\nTraining complete!")
        print(
            f"Best validation AUROC: "
            f"{best_val_auc:.4f}"
        )

    finally:

        wandb.finish()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_model()