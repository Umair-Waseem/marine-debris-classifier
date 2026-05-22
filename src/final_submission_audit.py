from pathlib import Path
import json


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PROCESSED_RECORDS_DIR = PROJECT_ROOT / "data" / "processed" / "records"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_REPORT_PATH = REPORTS_DIR / "final_submission_audit_report.md"
AUDIT_SUMMARY_PATH = REPORTS_DIR / "final_submission_audit_summary.json"


# =========================================================
# Required Files
# =========================================================

REQUIRED_FILES = {
    "final_model_checkpoint": MODELS_DIR / "resnet18_finetuned_layer4_best.pth",
    "class_mapping_json": REPORTS_DIR / "class_mapping.json",
    "final_split_csv": PROCESSED_RECORDS_DIR / "final_train_val_test_split.csv",
    "final_test_summary_json": REPORTS_DIR / "final_test_evaluation_summary.json",
    "multilabel_thresholds_json": REPORTS_DIR / "multilabel_layer4_tuned_thresholds.json",
    "multilabel_threshold_summary_json": REPORTS_DIR / "multilabel_threshold_tuning_summary.json",
    "final_test_confusion_matrix_plot": PLOTS_DIR / "final_test_confusion_matrix.png",
    "final_test_normalized_confusion_matrix_plot": (
        PLOTS_DIR / "final_test_confusion_matrix_normalized.png"
    ),
    "readme": PROJECT_ROOT / "README.md",
    "run_instructions": PROJECT_ROOT / "run_instructions.md",
    "project_report": PROJECT_ROOT / "project_report.md",
    "streamlit_app": PROJECT_ROOT / "app" / "streamlit_app.py",
    "predict_py": PROJECT_ROOT / "src" / "predict.py",
    "requirements": PROJECT_ROOT / "requirements.txt",
}


# =========================================================
# Helper Functions
# =========================================================

def rel(path: Path) -> str:
    """Return relative path if possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main():
    print("=" * 80)
    print("FINAL SUBMISSION AUDIT STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)

    checks = []
    missing_files = []
    passed_files = []

    for name, path in REQUIRED_FILES.items():
        exists = path.exists()

        check = {
            "name": name,
            "path": str(path),
            "relative_path": rel(path),
            "exists": bool(exists),
        }

        checks.append(check)

        if exists:
            passed_files.append(rel(path))
        else:
            missing_files.append(rel(path))

    final_readiness_status = len(missing_files) == 0

    summary = {
        "project_root": str(PROJECT_ROOT),
        "total_required_files": int(len(REQUIRED_FILES)),
        "passed_count": int(len(passed_files)),
        "missing_count": int(len(missing_files)),
        "final_readiness_status": bool(final_readiness_status),
        "passed_files": passed_files,
        "missing_files": missing_files,
        "checks": checks,
    }

    with open(AUDIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    markdown_lines = []

    markdown_lines.append("# Final Submission Audit Report")
    markdown_lines.append("")
    markdown_lines.append("## 1. Purpose")
    markdown_lines.append("")
    markdown_lines.append(
        "This audit checks whether the important final project files exist "
        "before submission."
    )

    markdown_lines.append("")
    markdown_lines.append("## 2. Final Readiness Status")
    markdown_lines.append("")
    markdown_lines.append(f"`{final_readiness_status}`")

    markdown_lines.append("")
    markdown_lines.append("## 3. File Checks")
    markdown_lines.append("")
    markdown_lines.append("| Name | Path | Exists |")
    markdown_lines.append("|---|---|---|")

    for check in checks:
        markdown_lines.append(
            f"| {check['name']} | `{check['relative_path']}` | {check['exists']} |"
        )

    markdown_lines.append("")
    markdown_lines.append("## 4. Missing Files")
    markdown_lines.append("")

    if missing_files:
        for path in missing_files:
            markdown_lines.append(f"- `{path}`")
    else:
        markdown_lines.append("- No missing files.")

    markdown_lines.append("")
    markdown_lines.append("## 5. Passed Files")
    markdown_lines.append("")

    for path in passed_files:
        markdown_lines.append(f"- `{path}`")

    markdown_lines.append("")
    markdown_lines.append("## 6. Recommendation")
    markdown_lines.append("")

    if final_readiness_status:
        markdown_lines.append(
            "All required final files exist. The project is ready for final packaging/submission review."
        )
    else:
        markdown_lines.append(
            "Some required files are missing. Create the missing files and rerun this audit."
        )

    AUDIT_REPORT_PATH.write_text("\n".join(markdown_lines), encoding="utf-8")

    print("\nPassed checks:")
    for path in passed_files:
        print("-", path)

    print("\nMissing files:")
    if missing_files:
        for path in missing_files:
            print("-", path)
    else:
        print("- None")

    print("\nFinal readiness status:", final_readiness_status)

    print("\nAudit output files:")
    print("-", AUDIT_REPORT_PATH)
    print("-", AUDIT_SUMMARY_PATH)

    print("\nFINAL SUBMISSION AUDIT COMPLETED.")


if __name__ == "__main__":
    main()
