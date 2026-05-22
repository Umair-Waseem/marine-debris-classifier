from pathlib import Path
import argparse
import json

from PIL import Image, ImageOps, UnidentifiedImageError

import torch
import torch.nn as nn
from torchvision import transforms

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
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Model / Threshold Paths
# =========================================================

SINGLE_LABEL_CHECKPOINT_PATH = MODELS_DIR / "resnet18_finetuned_layer4_best.pth"
MULTILABEL_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_layer4_best.pth"
MULTILABEL_THRESHOLDS_PATH = REPORTS_DIR / "multilabel_layer4_tuned_thresholds.json"

DUAL_MODEL_EXAMPLE_OUTPUT_JSON = REPORTS_DIR / "dual_model_prediction_example.json"


# =========================================================
# Fixed Configuration
# =========================================================

IMAGE_SIZE = 224

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# Global Lazy Cache
# =========================================================

_MODEL_CACHE = {}


# =========================================================
# Helper Functions
# =========================================================

def get_device(device=None):
    """Return selected device, CUDA if available, otherwise CPU."""
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(path_like) -> Path:
    """
    Resolve image path.

    If path is relative, first try current working directory behavior,
    then project-root-relative behavior.
    """
    path = Path(path_like).expanduser()

    if path.exists():
        return path.resolve()

    if not path.is_absolute():
        project_candidate = PROJECT_ROOT / path

        if project_candidate.exists():
            return project_candidate.resolve()

    return path.resolve()


def create_resnet18_head(num_outputs: int):
    """
    Create ResNet18 architecture with a custom output head.

    weights=None is used because we load trained checkpoints from disk.
    """
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_outputs)

    return model


def load_checkpoint(path: Path, device):
    """Load checkpoint safely across PyTorch versions."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    return checkpoint


def extract_state_dict(checkpoint):
    """
    Extract model state dict from common checkpoint formats.

    Supports:
    - checkpoint dict with model_state_dict
    - checkpoint dict with state_dict
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


def load_single_label_model(device):
    """Load official single-label model."""
    model = create_resnet18_head(num_outputs=len(TARGET_CLASSES))

    checkpoint = load_checkpoint(
        path=SINGLE_LABEL_CHECKPOINT_PATH,
        device=device,
    )

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def load_multilabel_model(device):
    """Load selected multi-label model."""
    model = create_resnet18_head(num_outputs=len(TARGET_CLASSES))

    checkpoint = load_checkpoint(
        path=MULTILABEL_CHECKPOINT_PATH,
        device=device,
    )

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def load_multilabel_thresholds():
    """Load tuned per-class multi-label thresholds."""
    if not MULTILABEL_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Tuned thresholds JSON not found: {MULTILABEL_THRESHOLDS_PATH}"
        )

    with open(MULTILABEL_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    thresholds = threshold_data.get("thresholds")

    if thresholds is None:
        thresholds = threshold_data.get("tuned_thresholds_by_class")

    if thresholds is None:
        raise KeyError(
            "Threshold file does not contain 'thresholds' or "
            "'tuned_thresholds_by_class'. "
            f"File: {MULTILABEL_THRESHOLDS_PATH}"
        )

    threshold_vector = []

    for class_name in TARGET_CLASSES:
        if class_name not in thresholds:
            raise KeyError(
                f"Threshold for class '{class_name}' not found in {MULTILABEL_THRESHOLDS_PATH}"
            )

        threshold_vector.append(float(thresholds[class_name]))

    return {
        "thresholds_by_class": {
            class_name: float(thresholds[class_name])
            for class_name in TARGET_CLASSES
        },
        "threshold_vector": threshold_vector,
        "raw_threshold_file": threshold_data,
    }


def load_prediction_resources(device=None):
    """
    Load models and thresholds lazily.

    This prevents reloading models repeatedly in Streamlit.
    """
    if device is None:
        device = get_device()

    cache_key = str(device)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    single_label_model = load_single_label_model(device)
    multilabel_model = load_multilabel_model(device)
    threshold_config = load_multilabel_thresholds()

    resources = {
        "device": device,
        "single_label_model": single_label_model,
        "multilabel_model": multilabel_model,
        "threshold_config": threshold_config,
    }

    _MODEL_CACHE[cache_key] = resources

    return resources


def build_preprocess_transform():
    """Create ImageNet preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def load_and_preprocess_image(image_path: Path):
    """Load image, convert RGB, and preprocess for ResNet18."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc

    preprocess = build_preprocess_transform()
    image_tensor = preprocess(image).unsqueeze(0)

    return image_tensor


def probability_dict(probabilities):
    """Convert probability tensor/list to class-probability dictionary."""
    return {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(TARGET_CLASSES)
    }


def sorted_probability_list(probabilities_dict):
    """Sort probabilities from highest to lowest."""
    return sorted(
        [
            {
                "class_name": class_name,
                "probability": float(probability),
            }
            for class_name, probability in probabilities_dict.items()
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )


@torch.no_grad()
def predict_single_label(model, image_tensor, device):
    """
    Single-label prediction.

    Uses softmax over four classes.
    """
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = TARGET_CLASSES[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    probabilities_by_class = probability_dict(probabilities.tolist())

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities_by_class": probabilities_by_class,
        "probabilities_sorted": sorted_probability_list(probabilities_by_class),
    }


@torch.no_grad()
def predict_multilabel(model, image_tensor, device, threshold_config):
    """
    Multi-label prediction.

    Uses sigmoid per class and applies tuned per-class thresholds.
    """
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probabilities = torch.sigmoid(logits)[0].detach().cpu()

    thresholds_by_class = threshold_config["thresholds_by_class"]

    probabilities_by_class = probability_dict(probabilities.tolist())

    binary_decisions_by_class = {}
    positive_labels = []

    for class_name in TARGET_CLASSES:
        probability = float(probabilities_by_class[class_name])
        threshold = float(thresholds_by_class[class_name])
        decision = probability >= threshold

        binary_decisions_by_class[class_name] = bool(decision)

        if decision:
            positive_labels.append(class_name)

    fallback_used = False
    fallback_class = None

    if len(positive_labels) == 0:
        fallback_used = True

        fallback_class = max(
            probabilities_by_class,
            key=probabilities_by_class.get,
        )

        positive_labels = [fallback_class]
        binary_decisions_by_class[fallback_class] = True

    per_class_details = []

    for class_name in TARGET_CLASSES:
        per_class_details.append({
            "class_name": class_name,
            "probability": float(probabilities_by_class[class_name]),
            "threshold": float(thresholds_by_class[class_name]),
            "predicted_positive": bool(binary_decisions_by_class[class_name]),
        })

    return {
        "positive_labels": positive_labels,
        "fallback_used": bool(fallback_used),
        "fallback_class": fallback_class,
        "probabilities_by_class": probabilities_by_class,
        "thresholds_by_class": thresholds_by_class,
        "binary_decisions_by_class": binary_decisions_by_class,
        "per_class_details": per_class_details,
        "probabilities_sorted": sorted_probability_list(probabilities_by_class),
    }


def predict_dual(image_path):
    """
    Run dual-model inference on one image.

    Returns:
    - single-label primary prediction
    - multi-label additional debris predictions
    """
    image_path = resolve_path(image_path)

    resources = load_prediction_resources()
    device = resources["device"]

    image_tensor = load_and_preprocess_image(image_path)

    single_label_result = predict_single_label(
        model=resources["single_label_model"],
        image_tensor=image_tensor,
        device=device,
    )

    multilabel_result = predict_multilabel(
        model=resources["multilabel_model"],
        image_tensor=image_tensor,
        device=device,
        threshold_config=resources["threshold_config"],
    )

    result = {
        "image_path": str(image_path),
        "device": str(device),
        "target_classes": TARGET_CLASSES,
        "single_label_model": {
            "checkpoint": str(SINGLE_LABEL_CHECKPOINT_PATH),
            "purpose": "primary_dominant_debris_class_prediction",
        },
        "multilabel_model": {
            "checkpoint": str(MULTILABEL_CHECKPOINT_PATH),
            "thresholds_json": str(MULTILABEL_THRESHOLDS_PATH),
            "purpose": "possible_additional_visible_debris_prediction",
        },
        "single_label_prediction": single_label_result,
        "multilabel_prediction": multilabel_result,
        "interpretation": {
            "primary_prediction": single_label_result["predicted_class"],
            "primary_confidence": single_label_result["confidence"],
            "possible_visible_debris": multilabel_result["positive_labels"],
            "fallback_used_for_multilabel": multilabel_result["fallback_used"],
            "important_note": (
                "The single-label model predicts the dominant class. "
                "The multi-label model predicts possible visible debris classes. "
                "This is an experimental dual-model inference branch, not a new official test result."
            ),
        },
        "training_performed": False,
        "test_set_evaluated": False,
        "raw_data_modified": False,
    }

    return result


def save_prediction_example(result, output_path: Path):
    """Save prediction result as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Run dual-model inference for Marine Debris Classifier."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an input image.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DUAL-MODEL MARINE DEBRIS PREDICTION")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Input image:", args.image)
    print("Single-label checkpoint:", SINGLE_LABEL_CHECKPOINT_PATH)
    print("Multi-label checkpoint:", MULTILABEL_CHECKPOINT_PATH)
    print("Thresholds:", MULTILABEL_THRESHOLDS_PATH)

    result = predict_dual(args.image)

    save_prediction_example(
        result=result,
        output_path=DUAL_MODEL_EXAMPLE_OUTPUT_JSON,
    )

    print("\nPrimary single-label prediction:")
    print("Class:", result["single_label_prediction"]["predicted_class"])
    print("Confidence:", f"{result['single_label_prediction']['confidence']:.4f}")

    print("\nMulti-label prediction:")
    print("Positive labels:", result["multilabel_prediction"]["positive_labels"])
    print("Fallback used:", result["multilabel_prediction"]["fallback_used"])

    print("\nMulti-label per-class details:")
    for item in result["multilabel_prediction"]["per_class_details"]:
        print(
            f"- {item['class_name']}: "
            f"prob={item['probability']:.4f}, "
            f"threshold={item['threshold']:.2f}, "
            f"positive={item['predicted_positive']}"
        )

    print("\nSaved example JSON:")
    print(DUAL_MODEL_EXAMPLE_OUTPUT_JSON)

    print("\nDUAL-MODEL PREDICTION COMPLETED.")


if __name__ == "__main__":
    main()
