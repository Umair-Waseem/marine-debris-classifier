from pathlib import Path
import argparse


REQUIRED_PATHS = [
    "app.py",
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "LICENSE",
    ".streamlit/config.toml",
    "app/streamlit_app.py",
    "app/streamlit_dual_model_app.py",
    "src/predict.py",
    "src/predict_dual_model.py",
    "src/report_helpers.py",
    "src/streamlit_ui.py",
    "assets/sample_images/plastic_bottles_dominant.jpg",
    "outputs/models/resnet18_finetuned_layer4_best.pth",
    "outputs/models/multilabel_resnet18_layer4_best.pth",
    "outputs/reports/final_test_evaluation_summary.json",
    "outputs/reports/multilabel_layer4_tuned_thresholds.json",
    "outputs/reports/multilabel_threshold_tuning_summary.json",
    "outputs/plots/final_test_confusion_matrix.png",
    "outputs/plots/final_test_confusion_matrix_normalized.png",
]

ALLOW_PATTERNS = [
    "app.py",
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "LICENSE",
    ".dockerignore",
    ".gitignore",
    ".streamlit/**",
    "app/**",
    "src/**",
    "assets/**",
    "outputs/models/**",
    "outputs/reports/class_mapping.json",
    "outputs/reports/final_test_evaluation_summary.json",
    "outputs/reports/multilabel_layer4_tuned_thresholds.json",
    "outputs/reports/multilabel_threshold_tuning_summary.json",
    "outputs/plots/final_test_confusion_matrix.png",
    "outputs/plots/final_test_confusion_matrix_normalized.png",
]

IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "venv/**",
    "data/**",
    "outputs/temp_dual_model_uploads/**",
    "outputs/temp_hf_uploads/**",
    "outputs/report_assets/**",
    "outputs/reports/ml_format_pdf_preview/**",
    "outputs/plots/final_test_correct_samples_grid.png",
    "outputs/plots/final_test_misclassified_samples_grid.png",
    "outputs/plots/multilabel_train_batch_grid.png",
    "outputs/plots/transformed_train_batch_grid.png",
    "outputs/*.log",
    "outputs/*.err",
    "_submission_cleanup_archive/**",
    "marine_debris_classifier/**",
    "Project_files/**",
]


def check_required_files(project_root: Path) -> None:
    missing_paths = [
        relative_path
        for relative_path in REQUIRED_PATHS
        if not (project_root / relative_path).exists()
    ]

    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "These required deployment files are missing:\n"
            f"{missing_text}\n\n"
            "Fix the missing files before uploading."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload the Marine Debris Streamlit Space to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="UmairWaseem05/marine-debris-classifier",
        help="Hugging Face Space repo id, for example username/space-name.",
    )
    parser.add_argument(
        "--message",
        default="Deploy marine debris classifier app",
        help="Commit message shown on Hugging Face.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Prefer logging in with: hf auth login",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Run this first:\n"
            "python -m pip install -U huggingface_hub"
        ) from exc

    project_root = Path(__file__).resolve().parent
    check_required_files(project_root)

    print(f"Uploading deployment files from: {project_root}")
    print(f"Target Space: {args.repo_id}")

    api = HfApi(token=args.token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(project_root),
        path_in_repo=".",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=args.message,
    )

    print("Upload finished. Open the Hugging Face Space logs to watch the build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
