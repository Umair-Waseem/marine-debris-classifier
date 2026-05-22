from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

SINGLE_LABEL_EVALUATION_SUMMARY_PATH = (
    REPORTS_DIR / "final_test_evaluation_summary.json"
)
SINGLE_LABEL_CONFUSION_MATRIX_PLOT_PATH = (
    PLOTS_DIR / "final_test_confusion_matrix.png"
)
SINGLE_LABEL_NORMALIZED_CONFUSION_MATRIX_PLOT_PATH = (
    PLOTS_DIR / "final_test_confusion_matrix_normalized.png"
)

MULTILABEL_THRESHOLD_TUNING_SUMMARY_PATH = (
    REPORTS_DIR / "multilabel_threshold_tuning_summary.json"
)


def load_json_file(path: Path) -> dict:
    """Load a JSON report file if it exists."""
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_single_label_confusion_report() -> dict:
    """Load official single-label final-test confusion matrix data."""
    summary = load_json_file(SINGLE_LABEL_EVALUATION_SUMMARY_PATH)
    target_classes = summary.get("target_classes", TARGET_CLASSES)

    return {
        "summary_path": SINGLE_LABEL_EVALUATION_SUMMARY_PATH,
        "plot_path": SINGLE_LABEL_CONFUSION_MATRIX_PLOT_PATH,
        "normalized_plot_path": SINGLE_LABEL_NORMALIZED_CONFUSION_MATRIX_PLOT_PATH,
        "target_classes": target_classes,
        "matrix": summary.get("confusion_matrix", []),
        "normalized_matrix": summary.get("normalized_confusion_matrix", []),
        "test_size": summary.get("test_size"),
        "test_accuracy": summary.get("test_accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "weighted_f1": summary.get("weighted_f1"),
        "per_class_metrics": summary.get("per_class_metrics", {}),
    }


def load_multilabel_confusion_report() -> dict:
    """
    Load tuned multi-label validation confusion data.

    Multi-label classification has one binary confusion matrix per class:
    rows are actual negative/positive, columns are predicted negative/positive.
    """
    summary = load_json_file(MULTILABEL_THRESHOLD_TUNING_SUMMARY_PATH)
    target_classes = summary.get("target_classes", TARGET_CLASSES)
    tuned_metrics = summary.get("tuned_metrics", {})
    per_class_metrics = tuned_metrics.get("per_class_metrics", {})

    matrices = {}
    rows = []

    for class_name in target_classes:
        metrics = per_class_metrics.get(class_name, {})

        true_negative = int(metrics.get("true_negative", 0))
        false_positive = int(metrics.get("false_positive", 0))
        false_negative = int(metrics.get("false_negative", 0))
        true_positive = int(metrics.get("true_positive", 0))

        matrices[class_name] = [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ]

        rows.append({
            "class": class_name,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "support": int(metrics.get("support", 0)),
        })

    return {
        "summary_path": MULTILABEL_THRESHOLD_TUNING_SUMMARY_PATH,
        "target_classes": target_classes,
        "matrices": matrices,
        "rows": rows,
        "validation_size": summary.get("validation_size"),
        "thresholds_by_class": summary.get("tuned_thresholds_by_class", {}),
        "subset_accuracy": tuned_metrics.get("subset_accuracy"),
        "hamming_accuracy": tuned_metrics.get("hamming_accuracy"),
        "macro_f1": tuned_metrics.get("macro_f1"),
        "micro_f1": tuned_metrics.get("micro_f1"),
        "weighted_f1": tuned_metrics.get("weighted_f1"),
        "test_set_used": summary.get("test_set_used", False),
    }
