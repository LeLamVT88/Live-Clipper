from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from utils import PROJECT_ROOT, ensure_dir, resolve_project_path


DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "all_frame_labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_MODEL_OUTPUT = DEFAULT_OUTPUT_DIR / "mobilenet_v3_small_lineup.pt"
DEFAULT_METRICS_OUTPUT = DEFAULT_OUTPUT_DIR / "mobilenet_v3_small_metrics.csv"
DEFAULT_HISTORY_OUTPUT = DEFAULT_OUTPUT_DIR / "mobilenet_v3_small_history.csv"
DEFAULT_PREDICTIONS_OUTPUT = DEFAULT_OUTPUT_DIR / "mobilenet_v3_small_predictions.csv"
REQUIRED_COLUMNS = {
    "split",
    "video_id",
    "video",
    "frame_path",
    "timestamp",
    "timestamp_seconds",
    "label",
}
PREDICTION_COLUMNS = [
    "split",
    "video_id",
    "video",
    "frame_path",
    "timestamp",
    "timestamp_seconds",
    "label",
]


class MobileNetTrainingError(Exception):
    """Raised when MobileNet training input or configuration is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an ImageNet-pretrained MobileNetV3-Small for lineup frame "
            "classification, select checkpoint/threshold on validation, then evaluate test."
        )
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY_OUTPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=8)
    parser.add_argument("--fine-tune-blocks", type=int, default=3)
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise MobileNetTrainingError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise MobileNetTrainingError("MPS was requested but is not available.")
    return torch.device(requested)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise MobileNetTrainingError(
            f"Dataset CSV does not exist: {csv_path}. "
            "Run python src/lineup/build_dataset_index.py first."
        )

    dataset = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(dataset.columns))
    if missing:
        raise MobileNetTrainingError(
            f"Missing columns in {csv_path}: {', '.join(missing)}"
        )

    dataset = dataset.copy()
    dataset["label"] = pd.to_numeric(dataset["label"], errors="raise").astype(int)
    dataset["timestamp_seconds"] = pd.to_numeric(
        dataset["timestamp_seconds"], errors="raise"
    )
    if not set(dataset["label"]).issubset({0, 1}):
        raise MobileNetTrainingError("Column label must contain only 0 and 1.")

    expected_splits = {"train", "val", "test"}
    actual_splits = set(dataset["split"].astype(str))
    missing_splits = sorted(expected_splits - actual_splits)
    if missing_splits:
        raise MobileNetTrainingError(
            "Dataset must contain train, val, and test splits. Missing: "
            + ", ".join(missing_splits)
        )

    split_counts = dataset.groupby("video_id")["split"].nunique()
    leaked_videos = split_counts[split_counts > 1].index.tolist()
    if leaked_videos:
        raise MobileNetTrainingError(
            "The same video appears in multiple splits: " + ", ".join(leaked_videos)
        )

    duplicates = dataset.duplicated(["video_id", "timestamp_seconds"])
    if duplicates.any():
        raise MobileNetTrainingError(
            f"Dataset has {int(duplicates.sum())} duplicate video/timestamp row(s)."
        )
    return dataset


def sample_training_rows(
    train_rows: pd.DataFrame,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    if negative_ratio <= 0:
        raise MobileNetTrainingError("--negative-ratio must be greater than 0.")

    positives = train_rows[train_rows["label"] == 1]
    negatives = train_rows[train_rows["label"] == 0]
    if positives.empty or negatives.empty:
        raise MobileNetTrainingError(
            "Training split must contain both lineup and non-lineup frames."
        )

    negative_count = min(len(negatives), len(positives) * negative_ratio)
    sampled_negatives = negatives.sample(
        n=negative_count,
        random_state=seed,
        replace=False,
    )
    sampled = pd.concat([positives, sampled_negatives], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    if image_size <= 0:
        raise MobileNetTrainingError("--image-size must be greater than 0.")

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.RandomAffine(
                degrees=1.5,
                translate=(0.015, 0.015),
                scale=(0.98, 1.02),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


class LineupFrameDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: pd.DataFrame, transform: transforms.Compose) -> None:
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        image_path = resolve_project_path(row["frame_path"])
        if not image_path.exists():
            raise MobileNetTrainingError(f"Frame does not exist: {image_path}")
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except OSError as exc:
            raise MobileNetTrainingError(f"Cannot read frame: {image_path}") from exc
        return tensor, int(row["label"])


def make_loader(
    rows: pd.DataFrame,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, int]]:
    if batch_size <= 0:
        raise MobileNetTrainingError("--batch-size must be greater than 0.")
    if num_workers < 0:
        raise MobileNetTrainingError("--num-workers cannot be negative.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        LineupFrameDataset(rows, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def create_model() -> nn.Module:
    torch.hub.set_dir(str(ensure_dir(DEFAULT_OUTPUT_DIR / ".torch_cache")))
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise MobileNetTrainingError("Unexpected MobileNetV3 classifier structure.")
    model.classifier[-1] = nn.Linear(final_layer.in_features, 2)
    return model


def configure_stage(model: nn.Module, fine_tune_blocks: int) -> None:
    features = model.features
    if not isinstance(features, nn.Sequential):
        raise MobileNetTrainingError("Unexpected MobileNetV3 feature structure.")
    if fine_tune_blocks < 0 or fine_tune_blocks > len(features):
        raise MobileNetTrainingError(
            f"--fine-tune-blocks must be between 0 and {len(features)}."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if fine_tune_blocks:
        for block in features[-fine_tune_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def set_stage_train_mode(model: nn.Module, fine_tune_blocks: int) -> None:
    model.train()
    features = model.features
    if fine_tune_blocks == 0:
        features.eval()
        return
    for block in features[:-fine_tune_blocks]:
        block.eval()


def build_class_weights(rows: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = rows["label"].value_counts().to_dict()
    if 0 not in counts or 1 not in counts:
        raise MobileNetTrainingError("Training sample must contain both labels.")
    total = float(len(rows))
    weights = [total / (2.0 * counts[label]) for label in (0, 1)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fine_tune_blocks: int,
) -> float:
    set_stage_train_mode(model, fine_tune_blocks)
    total_loss = 0.0
    total_rows = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_rows += batch_size

    return total_loss / total_rows


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        scores = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_rows += batch_size
        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(scores.detach().cpu().numpy())

    return (
        total_loss / total_rows,
        np.concatenate(all_labels).astype(np.int32),
        np.concatenate(all_scores).astype(np.float32),
    )


def compute_metrics(
    split: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (scores >= threshold).astype(np.int32)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "split": split,
        "rows": len(labels),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(len(labels) - labels.sum()),
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    candidates = [
        compute_metrics("val", labels, scores, float(threshold))
        for threshold in np.arange(0.05, 0.951, 0.01)
    ]
    return max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )


def smooth_scores(
    rows: pd.DataFrame,
    scores: np.ndarray,
    window: int,
) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise MobileNetTrainingError("Smoothing window must be a positive odd number.")

    ordered = rows[["video_id", "timestamp_seconds"]].reset_index(drop=True).copy()
    ordered["_position"] = np.arange(len(ordered))
    ordered["_score"] = scores
    ordered = ordered.sort_values(["video_id", "timestamp_seconds"])
    ordered["_smoothed_score"] = ordered.groupby("video_id", sort=False)[
        "_score"
    ].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).mean()
    )
    return (
        ordered.sort_values("_position")["_smoothed_score"]
        .to_numpy(dtype=np.float32)
    )


def select_temporal_postprocess(
    rows: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    windows: tuple[int, ...] = (1, 3, 5, 7),
) -> tuple[int, float, np.ndarray, dict[str, object]]:
    best: tuple[
        tuple[float, float, float, float],
        int,
        float,
        np.ndarray,
        dict[str, object],
    ] | None = None
    for window in windows:
        smoothed_scores = smooth_scores(rows, scores, window)
        metrics = select_threshold(labels, smoothed_scores)
        key = (
            float(metrics["f1"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
            -abs(float(metrics["threshold"]) - 0.5),
        )
        if best is None or key > best[0]:
            best = (
                key,
                window,
                float(metrics["threshold"]),
                smoothed_scores,
                metrics,
            )

    if best is None:
        raise MobileNetTrainingError("No temporal postprocessing configuration found.")
    _, window, threshold, smoothed_scores, metrics = best
    metrics = metrics.copy()
    metrics["split"] = "val_smoothed"
    metrics["smoothing_window"] = window
    return window, threshold, smoothed_scores, metrics


def prediction_frame(
    rows: pd.DataFrame,
    scores: np.ndarray,
    raw_threshold: float,
    smoothed_scores: np.ndarray,
    smoothed_threshold: float,
) -> pd.DataFrame:
    predictions = rows[PREDICTION_COLUMNS].reset_index(drop=True).copy()
    predictions["score"] = scores
    predictions["raw_pred_label"] = (scores >= raw_threshold).astype(np.int32)
    predictions["smoothed_score"] = smoothed_scores
    predictions["pred_label"] = (smoothed_scores >= smoothed_threshold).astype(np.int32)
    return predictions


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def print_metrics(row: dict[str, object]) -> None:
    print(
        f"{row['split']}: accuracy={float(row['accuracy']):.4f} "
        f"precision={float(row['precision']):.4f} "
        f"recall={float(row['recall']):.4f} "
        f"f1={float(row['f1']):.4f} "
        f"threshold={float(row['threshold']):.2f} "
        f"tp={row['tp']} fp={row['fp']} tn={row['tn']} fn={row['fn']}"
    )


def main() -> int:
    args = parse_args()
    try:
        if args.epochs_head < 0 or args.epochs_finetune < 0:
            raise MobileNetTrainingError("Epoch counts cannot be negative.")
        if args.epochs_head + args.epochs_finetune <= 0:
            raise MobileNetTrainingError("At least one training epoch is required.")
        if args.patience <= 0:
            raise MobileNetTrainingError("--patience must be greater than 0.")

        seed_everything(args.seed)
        device = select_device(args.device)
        print(f"Device: {device}")

        dataset = load_dataset(args.dataset_csv)
        train_rows = dataset[dataset["split"] == "train"].reset_index(drop=True)
        val_rows = dataset[dataset["split"] == "val"].reset_index(drop=True)
        test_rows = dataset[dataset["split"] == "test"].reset_index(drop=True)
        sampled_train_rows = sample_training_rows(
            train_rows,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )

        print(
            f"Videos: train={train_rows['video_id'].nunique()} "
            f"val={val_rows['video_id'].nunique()} test={test_rows['video_id'].nunique()}"
        )
        print(
            f"Training sample: rows={len(sampled_train_rows)} "
            f"lineup={int(sampled_train_rows['label'].sum())} "
            f"non_lineup={int((sampled_train_rows['label'] == 0).sum())}"
        )

        train_transform, eval_transform = build_transforms(args.image_size)
        train_loader = make_loader(
            sampled_train_rows,
            train_transform,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            seed=args.seed,
        )
        val_loader = make_loader(
            val_rows,
            eval_transform,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            seed=args.seed,
        )
        test_loader = make_loader(
            test_rows,
            eval_transform,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            seed=args.seed,
        )

        model = create_model().to(device)
        class_weights = build_class_weights(sampled_train_rows, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        stages = [
            ("head", args.epochs_head, 0, args.learning_rate_head),
            (
                "finetune",
                args.epochs_finetune,
                args.fine_tune_blocks,
                args.learning_rate_finetune,
            ),
        ]

        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, object] | None = None
        best_epoch = 0
        history: list[dict[str, object]] = []
        global_epoch = 0

        for stage_name, stage_epochs, fine_tune_blocks, learning_rate in stages:
            if stage_epochs == 0:
                continue
            configure_stage(model, fine_tune_blocks)
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=learning_rate,
                weight_decay=args.weight_decay,
            )
            epochs_without_improvement = 0
            print(
                f"Stage={stage_name} epochs={stage_epochs} "
                f"trainable_parameters={sum(p.numel() for p in trainable_parameters):,}"
            )

            for stage_epoch in range(1, stage_epochs + 1):
                global_epoch += 1
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    fine_tune_blocks,
                )
                val_loss, val_labels, val_scores = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                )
                val_metrics = select_threshold(val_labels, val_scores)
                history.append(
                    {
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "stage_epoch": stage_epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **val_metrics,
                    }
                )
                print(
                    f"epoch={global_epoch:02d} stage={stage_name} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )
                print_metrics(val_metrics)

                is_better = best_metrics is None or (
                    float(val_metrics["f1"]),
                    float(val_metrics["recall"]),
                    float(val_metrics["precision"]),
                ) > (
                    float(best_metrics["f1"]),
                    float(best_metrics["recall"]),
                    float(best_metrics["precision"]),
                )
                if is_better:
                    best_state = cpu_state_dict(model)
                    best_metrics = val_metrics.copy()
                    best_epoch = global_epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.patience:
                        print(f"Early stopping stage={stage_name} at epoch={global_epoch}")
                        break

        if best_state is None or best_metrics is None:
            raise MobileNetTrainingError("Training did not produce a valid checkpoint.")

        model.load_state_dict(best_state)
        raw_threshold = float(best_metrics["threshold"])
        val_loss, val_labels, val_scores = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        test_loss, test_labels, test_scores = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        raw_val_metrics = compute_metrics(
            "val_raw", val_labels, val_scores, raw_threshold
        )
        raw_test_metrics = compute_metrics(
            "test_raw", test_labels, test_scores, raw_threshold
        )
        (
            smoothing_window,
            threshold,
            smoothed_val_scores,
            smoothed_val_metrics,
        ) = select_temporal_postprocess(val_rows, val_labels, val_scores)
        smoothed_test_scores = smooth_scores(test_rows, test_scores, smoothing_window)
        smoothed_test_metrics = compute_metrics(
            "test_smoothed",
            test_labels,
            smoothed_test_scores,
            threshold,
        )
        smoothed_test_metrics["smoothing_window"] = smoothing_window
        for metrics, loss in (
            (raw_val_metrics, val_loss),
            (raw_test_metrics, test_loss),
            (smoothed_val_metrics, val_loss),
            (smoothed_test_metrics, test_loss),
        ):
            metrics["loss"] = loss
            metrics["best_epoch"] = best_epoch

        ensure_dir(args.model_output.parent)
        torch.save(
            {
                "architecture": "mobilenet_v3_small",
                "pretrained_weights": "MobileNet_V3_Small_Weights.DEFAULT",
                "model_state_dict": best_state,
                "image_size": args.image_size,
                "threshold": threshold,
                "raw_threshold": raw_threshold,
                "smoothing_window": smoothing_window,
                "smoothing_mode": "centered_mean",
                "class_names": ["non_lineup", "lineup"],
                "best_epoch": best_epoch,
                "seed": args.seed,
            },
            args.model_output,
        )
        ensure_dir(args.metrics_output.parent)
        pd.DataFrame(
            [
                raw_val_metrics,
                raw_test_metrics,
                smoothed_val_metrics,
                smoothed_test_metrics,
            ]
        ).to_csv(
            args.metrics_output,
            index=False,
        )
        ensure_dir(args.history_output.parent)
        pd.DataFrame(history).to_csv(args.history_output, index=False)
        predictions = pd.concat(
            [
                prediction_frame(
                    val_rows,
                    val_scores,
                    raw_threshold,
                    smoothed_val_scores,
                    threshold,
                ),
                prediction_frame(
                    test_rows,
                    test_scores,
                    raw_threshold,
                    smoothed_test_scores,
                    threshold,
                ),
            ],
            ignore_index=True,
        )
        ensure_dir(args.predictions_output.parent)
        predictions.to_csv(args.predictions_output, index=False)

        print(f"Best validation checkpoint: epoch={best_epoch}")
        print_metrics(raw_val_metrics)
        print_metrics(raw_test_metrics)
        print(
            f"Temporal postprocess selected on validation: "
            f"window={smoothing_window} threshold={threshold:.2f}"
        )
        print_metrics(smoothed_val_metrics)
        print_metrics(smoothed_test_metrics)
        print(f"Saved model: {args.model_output}")
        print(f"Saved metrics: {args.metrics_output}")
        print(f"Saved history: {args.history_output}")
        print(f"Saved predictions: {args.predictions_output}")
        return 0
    except (MobileNetTrainingError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
