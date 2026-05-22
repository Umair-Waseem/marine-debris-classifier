from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms as T

try:
    from torchvision.models import resnet18
except Exception as exc:
    raise ImportError(
        "torchvision is required for inference. Install it using: "
        "python -m pip install torchvision"
    ) from exc


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

DEFAULT_CHECKPOINT_PATH = MODELS_DIR / "resnet18_finetuned_layer4_best.pth"


# =========================================================
# Fixed Project Configuration
# =========================================================

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

CLASS_TO_IDX = {
    "plastic": 0,
    "foam": 1,
    "metal": 2,
    "other_debris": 3,
}

IDX_TO_CLASS = {
    0: "plastic",
    1: "foam",
    2: "metal",
    3: "other_debris",
}

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# Helper Functions
# =========================================================

def get_device(device=None):
    """
    Return a valid torch device.
    """
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(path_like) -> Path:
    """
    Resolve a path from the current directory or project root.
    """
    path = Path(path_like).expanduser()

    if path.exists():
        return path.resolve()

    if not path.is_absolute():
        project_candidate = PROJECT_ROOT / path

        if project_candidate.exists():
            return project_candidate.resolve()

    return path.resolve()


def get_inference_transforms():
    """
    Inference transforms must match validation/test transforms.

    No random augmentation is used during inference.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_inference_model(num_classes=4):
    """
    Recreate ResNet18 architecture for inference.

    weights=None is used because we load our trained checkpoint weights.
    This avoids downloading pretrained weights during deployment/inference.
    """
    model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def extract_state_dict(checkpoint):
    """
    Extract a model state dict from common checkpoint formats.

    Supports:
    - training checkpoint with model_state_dict
    - checkpoint with state_dict
    - raw state_dict
    - DataParallel checkpoints with module. prefixes
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dictionary.")

    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_checkpoint_safely(checkpoint_path: Path, device):
    """
    Load checkpoint safely.

    weights_only=False is used where supported because the checkpoint contains
    both model weights and metadata.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Expected selected model checkpoint at:\n"
            f"{DEFAULT_CHECKPOINT_PATH}"
        )

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


# =========================================================
# Model Loading
# =========================================================

def load_trained_model(checkpoint_path=None, device=None):
    """
    Load the trained ResNet18 model.

    Args:
        checkpoint_path: Optional path to checkpoint. Defaults to final selected model.
        device: Optional torch device.

    Returns:
        model, device, checkpoint
    """
    device = get_device(device)

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    else:
        checkpoint_path = resolve_path(checkpoint_path)

    checkpoint_path = Path(checkpoint_path)

    checkpoint = load_checkpoint_safely(checkpoint_path, device)

    state_dict = extract_state_dict(checkpoint)

    model = create_inference_model(num_classes=len(TARGET_CLASSES))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device, checkpoint


# =========================================================
# Prediction Functions
# =========================================================

def predict_pil_image(image, model=None, device=None, top_k=4):
    """
    Predict class for a PIL image.

    Args:
        image: PIL Image.
        model: Optional loaded model. If None, model is loaded automatically.
        device: Optional torch device.
        top_k: Number of top predictions to return.

    Returns:
        Prediction dictionary.
    """
    if model is not None and device is None:
        device = next(model.parameters()).device
    else:
        device = get_device(device)

    if model is None:
        model, device, _ = load_trained_model(device=device)

    if not isinstance(image, Image.Image):
        raise TypeError("predict_pil_image expects a PIL.Image.Image object.")

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    transform = get_inference_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_k = int(top_k)
    top_k = max(1, min(top_k, len(TARGET_CLASSES)))

    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    predicted_index = int(top_indices[0].item())
    predicted_label = IDX_TO_CLASS[predicted_index]
    confidence = float(top_probs[0].item())

    top_k_predictions = []

    for prob, idx in zip(top_probs, top_indices):
        idx_int = int(idx.item())

        top_k_predictions.append({
            "label": IDX_TO_CLASS[idx_int],
            "index": idx_int,
            "probability": float(prob.item()),
        })

    return {
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "top_k_predictions": top_k_predictions,
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": {
            str(idx): class_name
            for idx, class_name in IDX_TO_CLASS.items()
        },
    }


def predict_image(image_path, model=None, device=None, top_k=4):
    """
    Predict class for an image file path.

    Args:
        image_path: Path to image.
        model: Optional loaded model.
        device: Optional torch device.
        top_k: Number of top predictions.

    Returns:
        Prediction dictionary.
    """
    image_path = resolve_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            prediction = predict_pil_image(
                image=image,
                model=model,
                device=device,
                top_k=top_k,
            )

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc

    prediction["image_path"] = str(image_path)

    return prediction


# =========================================================
# Command-Line Interface
# =========================================================

def save_prediction_json(prediction: dict, output_path) -> Path:
    """Save a prediction dictionary as JSON."""
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(prediction, indent=4),
        encoding="utf-8",
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run single-label inference for the Marine Debris Classifier."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Optional model checkpoint path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of class probabilities to display.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("Marine Debris Classifier Inference Module")
    print("Project root:", PROJECT_ROOT)
    print("Checkpoint:", resolve_path(args.checkpoint))
    print("Input image:", resolve_path(args.image))

    model, device, checkpoint = load_trained_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded successfully.")
    print("Device:", device)
    print("Checkpoint model name:", checkpoint.get("model_name", "unknown"))

    prediction = predict_image(
        image_path=args.image,
        model=model,
        device=device,
        top_k=args.top_k,
    )

    print("\nPrediction:")
    print("Class:", prediction["predicted_label"])
    print("Confidence:", f"{prediction['confidence']:.4f}")

    print("\nTop predictions:")
    for item in prediction["top_k_predictions"]:
        print(f"- {item['label']}: {item['probability']:.4f}")

    if args.output:
        output_path = save_prediction_json(prediction, args.output)
        print("\nSaved prediction JSON:")
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
