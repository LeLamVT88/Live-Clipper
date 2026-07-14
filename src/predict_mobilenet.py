from __future__ import annotations

import argparse
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


DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "extracted_frames.csv"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "predictions" / "mobilenet_v3_small_lineup.pt"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "predictions" / "mobilenet_v3_small_inference.csv"
)
REQUIRED_COLUMNS = {"video", "frame_path", "timestamp", "timestamp_seconds"}


class MobileNetPredictionError(Exception):
    """Raised when MobileNet inference input or configuration is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict lineup scores with a trained MobileNetV3-Small checkpoint."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the smoothed-score threshold saved in the checkpoint.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Override the odd temporal smoothing window saved in the checkpoint.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise MobileNetPredictionError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise MobileNetPredictionError("MPS was requested but is not available.")
    return torch.device(requested)


def load_frames(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise MobileNetPredictionError(
            f"Input CSV does not exist: {csv_path}. "
            "Run python src/extract_frames.py first, or pass --input-csv."
        )

    frames = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(frames.columns))
    if missing:
        raise MobileNetPredictionError(
            f"Missing columns in {csv_path}: {', '.join(missing)}"
        )
    if frames.empty:
        raise MobileNetPredictionError(f"Input CSV has no frame rows: {csv_path}")

    frames = frames.copy()
    frames["timestamp_seconds"] = pd.to_numeric(
        frames["timestamp_seconds"], errors="raise"
    )
    if "video_id" not in frames.columns:
        frames["video_id"] = frames["video"].map(lambda value: Path(str(value)).stem)
    return frames


def load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise MobileNetPredictionError(
            f"Model checkpoint does not exist: {checkpoint_path}. "
            "Run python src/train_mobilenet.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise MobileNetPredictionError("Model checkpoint must contain a dictionary.")

    required_keys = {
        "architecture",
        "model_state_dict",
        "image_size",
        "threshold",
        "raw_threshold",
        "smoothing_window",
    }
    missing = sorted(required_keys - set(checkpoint))
    if missing:
        raise MobileNetPredictionError(
            "Model checkpoint is missing: " + ", ".join(missing)
        )
    if checkpoint["architecture"] != "mobilenet_v3_small":
        raise MobileNetPredictionError(
            f"Unsupported checkpoint architecture: {checkpoint['architecture']}"
        )
    return checkpoint


def create_model(checkpoint: dict[str, object]) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise MobileNetPredictionError("Unexpected MobileNetV3 classifier structure.")
    model.classifier[-1] = nn.Linear(final_layer.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def build_transform(image_size: int) -> transforms.Compose:
    if image_size <= 0:
        raise MobileNetPredictionError("Checkpoint image_size must be greater than 0.")
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class PredictionDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, frames: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frames = frames.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = resolve_project_path(self.frames.iloc[index]["frame_path"])
        if not image_path.exists():
            raise MobileNetPredictionError(f"Frame does not exist: {image_path}")
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except OSError as exc:
            raise MobileNetPredictionError(f"Cannot read frame: {image_path}") from exc
        return tensor, index


@torch.inference_mode()
def predict_scores(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    row_count: int,
) -> np.ndarray:
    model.eval()
    scores = np.empty(row_count, dtype=np.float32)
    for images, positions in loader:
        logits = model(images.to(device))
        batch_scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        scores[positions.numpy()] = batch_scores
    return scores


def smooth_scores(frames: pd.DataFrame, scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise MobileNetPredictionError(
            "Smoothing window must be a positive odd number."
        )

    ordered = frames[["video_id", "timestamp_seconds"]].reset_index(drop=True).copy()
    ordered["_position"] = np.arange(len(ordered))
    ordered["_score"] = scores
    ordered = ordered.sort_values(["video_id", "timestamp_seconds"])
    ordered["_smoothed_score"] = ordered.groupby("video_id", sort=False)[
        "_score"
    ].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).mean()
    )
    return ordered.sort_values("_position")["_smoothed_score"].to_numpy(
        dtype=np.float32
    )


def main() -> int:
    args = parse_args()
    try:
        if args.batch_size <= 0:
            raise MobileNetPredictionError("--batch-size must be greater than 0.")
        if args.num_workers < 0:
            raise MobileNetPredictionError("--num-workers cannot be negative.")

        frames = load_frames(args.input_csv)
        checkpoint = load_checkpoint(args.model)
        device = select_device(args.device)
        image_size = int(checkpoint["image_size"])
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(checkpoint["threshold"])
        )
        raw_threshold = float(checkpoint["raw_threshold"])
        smoothing_window = (
            args.smoothing_window
            if args.smoothing_window is not None
            else int(checkpoint["smoothing_window"])
        )
        if not 0.0 <= threshold <= 1.0:
            raise MobileNetPredictionError("--threshold must be between 0 and 1.")

        dataset = PredictionDataset(frames, build_transform(image_size))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = create_model(checkpoint).to(device)
        scores = predict_scores(model, loader, device, len(frames))
        smoothed_scores = smooth_scores(frames, scores, smoothing_window)

        predictions = frames.copy()
        predictions["score"] = scores
        predictions["raw_pred_label"] = (scores >= raw_threshold).astype(np.int32)
        predictions["smoothed_score"] = smoothed_scores
        predictions["pred_label"] = (smoothed_scores >= threshold).astype(np.int32)

        ensure_dir(args.output_csv.parent)
        predictions.to_csv(args.output_csv, index=False)
        print(f"Device: {device}")
        print(
            f"Frames: {len(predictions)} | predicted_lineup: "
            f"{int(predictions['pred_label'].sum())}"
        )
        print(
            f"Postprocess: smoothing_window={smoothing_window} "
            f"threshold={threshold:.2f}"
        )
        print(f"Saved predictions: {args.output_csv}")
        return 0
    except (
        MobileNetPredictionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
