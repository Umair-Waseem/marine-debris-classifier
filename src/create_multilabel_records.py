from pathlib import Path
import json
import ast

import pandas as pd


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_RECORDS_DIR = DATA_DIR / "processed" / "records"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

PROCESSED_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Input / Output Files
# =========================================================

FINAL_SPLIT_CSV = PROCESSED_RECORDS_DIR / "final_train_val_test_split.csv"

MULTILABEL_ALL_SPLITS_CSV = PROCESSED_RECORDS_DIR / "multilabel_records_all_splits.csv"
MULTILABEL_TRAIN_VAL_CSV = PROCESSED_RECORDS_DIR / "multilabel_train_val_records.csv"

SUMMARY_JSON = REPORTS_DIR / "multilabel_records_summary.json"
REPORT_MD = REPORTS_DIR / "multilabel_records_report.md"
CLASS_DISTRIBUTION_CSV = REPORTS_DIR / "multilabel_class_distribution.csv"


# =========================================================
# Fixed Multi-Label Class Order
# =========================================================

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

MULTILABEL_COLUMNS = {
    "plastic": "multilabel_plastic",
    "foam": "multilabel_foam",
    "metal": "multilabel_metal",
    "other_debris": "multilabel_other_debris",
}


# =========================================================
# Helper Functions
# =========================================================

def normalize_text(value) -> str:
    """Convert value to clean string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_label(value) -> str:
    """Normalize label text."""
    return normalize_text(value).lower()


def to_bool(value) -> bool:
    """Convert common boolean-like values to bool."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def safe_parse_json_like(value):
    """
    Safely parse JSON-like values.

    Handles:
    - JSON list/dict strings
    - Python literal list/dict strings
    - already parsed list/dict
    - empty / missing values
    """
    if pd.isna(value):
        return None

    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()

    if text == "":
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def extract_labels_from_parsed_value(parsed_value):
    """
    Extract project labels from parsed JSON-like object.

    Supported parsed formats:
    - list: ["plastic", "metal"]
    - dict: {"plastic": 3, "metal": 1}
    - string: "plastic"
    """
    labels = set()

    if parsed_value is None:
        return labels

    if isinstance(parsed_value, list):
        for item in parsed_value:
            label = normalize_label(item)

            if label in TARGET_CLASSES:
                labels.add(label)

    elif isinstance(parsed_value, dict):
        for key in parsed_value.keys():
            label = normalize_label(key)

            if label in TARGET_CLASSES:
                labels.add(label)

    elif isinstance(parsed_value, str):
        label = normalize_label(parsed_value)

        if label in TARGET_CLASSES:
            labels.add(label)

    return labels


def get_multilabel_set_from_row(row: pd.Series):
    """
    Create multi-label set for one record.

    Priority:
    1. mapped_unique_labels
    2. mapped_label_counts keys
    3. mapped_bbox_area_by_label keys
    4. selected_label fallback

    The fallback ensures every usable image has at least one label when mapped
    columns are missing.
    """
    label_set = set()

    candidate_columns = [
        "mapped_unique_labels",
        "mapped_label_counts",
        "mapped_bbox_area_by_label",
    ]

    for column in candidate_columns:
        if column not in row.index:
            continue

        parsed_value = safe_parse_json_like(row[column])
        extracted_labels = extract_labels_from_parsed_value(parsed_value)

        if extracted_labels:
            label_set.update(extracted_labels)

    if not label_set:
        selected_label = normalize_label(row.get("selected_label", ""))

        if selected_label in TARGET_CLASSES:
            label_set.add(selected_label)

    return label_set


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to Markdown table without extra dependencies."""
    if df is None or df.empty:
        return "_No records found._"

    small_df = df.head(max_rows).copy()
    columns = list(small_df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in small_df.iterrows():
        values = []

        for col in columns:
            text = str(row[col]).replace("\n", " ").replace("|", "/")
            values.append(text)

        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def validate_required_columns(df: pd.DataFrame):
    """Validate minimum required columns from final split CSV."""
    required_columns = [
        "filename",
        "selected_label",
        "final_split",
        "usable_for_multiclass",
        "caution_flag",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "final_train_val_test_split.csv is missing required columns:\n"
            + "\n".join(f"- {col}" for col in missing_columns)
        )


def create_distribution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-label positive-count distribution by split."""
    rows = []

    valid_splits = ["train", "validation", "test", "excluded_unusable"]

    for split_name in valid_splits:
        split_df = df[df["final_split"] == split_name].copy()

        for class_name in TARGET_CLASSES:
            col = MULTILABEL_COLUMNS[class_name]

            rows.append({
                "split": split_name,
                "class_name": class_name,
                "positive_count": int(split_df[col].sum()) if col in split_df.columns else 0,
                "split_records": int(len(split_df)),
            })

    return pd.DataFrame(rows)


def check_binary_columns(df: pd.DataFrame):
    """Check that all multi-label target columns contain only 0 or 1."""
    warnings = []

    for class_name, col in MULTILABEL_COLUMNS.items():
        if col not in df.columns:
            warnings.append(f"Missing multi-label column: {col}")
            continue

        unique_values = sorted(df[col].dropna().unique().tolist())

        invalid_values = [
            value for value in unique_values
            if value not in [0, 1]
        ]

        if invalid_values:
            warnings.append(
                f"Column {col} contains invalid values: {invalid_values}"
            )

    return warnings


def create_multilabel_records(df: pd.DataFrame):
    """Create multi-label columns and metadata columns."""
    df = df.copy()

    validate_required_columns(df)

    df["filename"] = df["filename"].apply(normalize_text)
    df["selected_label"] = df["selected_label"].apply(normalize_label)
    df["final_split"] = df["final_split"].apply(normalize_text).str.lower()
    df["caution_flag"] = df["caution_flag"].apply(normalize_text).str.lower()
    df["usable_for_multiclass"] = df["usable_for_multiclass"].apply(to_bool)

    if "is_ambiguous" in df.columns:
        df["is_ambiguous"] = df["is_ambiguous"].apply(to_bool)
    else:
        df["is_ambiguous"] = df["caution_flag"].eq("ambiguous_multilabel")

    multilabel_sets = []

    for _, row in df.iterrows():
        if to_bool(row.get("usable_for_multiclass", False)):
            label_set = get_multilabel_set_from_row(row)
        else:
            label_set = set()

        multilabel_sets.append(label_set)

    for class_name in TARGET_CLASSES:
        col = MULTILABEL_COLUMNS[class_name]

        df[col] = [
            1 if class_name in label_set else 0
            for label_set in multilabel_sets
        ]

    vectors = []
    label_names_list = []
    positive_counts = []

    for label_set in multilabel_sets:
        vector = [
            1 if class_name in label_set else 0
            for class_name in TARGET_CLASSES
        ]

        label_names = [
            class_name
            for class_name in TARGET_CLASSES
            if class_name in label_set
        ]

        vectors.append(json.dumps(vector))
        label_names_list.append(json.dumps(label_names))
        positive_counts.append(int(sum(vector)))

    df["multilabel_vector"] = vectors
    df["multilabel_label_names"] = label_names_list
    df["multilabel_num_positive_classes"] = positive_counts
    df["multilabel_is_single_positive"] = df["multilabel_num_positive_classes"].eq(1)
    df["multilabel_is_multi_positive"] = df["multilabel_num_positive_classes"].gt(1)

    return df


def build_report(summary: dict, distribution_df: pd.DataFrame):
    """Create Markdown report."""
    lines = []

    lines.append("# Multi-Label Records Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents creation of multi-label target records for an "
        "experimental multi-label branch of the Marine Debris Classifier project."
    )

    lines.append("")
    lines.append("## 2. Important Notes")
    lines.append("")
    lines.append("- This step does not train a model.")
    lines.append("- This step does not create DataLoaders.")
    lines.append("- This step does not evaluate the test set.")
    lines.append("- This step does not modify `data/raw/`.")
    lines.append("- This step does not overwrite the official single-label model.")
    lines.append("- Multi-label vector order is fixed as `[plastic, foam, metal, other_debris]`.")

    lines.append("")
    lines.append("## 3. Input File")
    lines.append("")
    lines.append(f"`{FINAL_SPLIT_CSV}`")

    lines.append("")
    lines.append("## 4. Summary")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total records | {summary['total_records']} |")
    lines.append(f"| Usable records | {summary['usable_records']} |")
    lines.append(f"| Unusable records | {summary['unusable_records']} |")
    lines.append(f"| Single-positive usable records | {summary['single_positive_usable_records']} |")
    lines.append(f"| Multi-positive usable records | {summary['multi_positive_usable_records']} |")
    lines.append(f"| Ambiguous records | {summary['ambiguous_records']} |")
    lines.append(f"| Usable records with zero positive labels | {summary['usable_zero_positive_records']} |")

    lines.append("")
    lines.append("## 5. Split Distribution")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["split_distribution"], indent=4))
    lines.append("```")

    lines.append("")
    lines.append("## 6. Multi-Label Class Distribution")
    lines.append("")
    lines.append(dataframe_to_markdown(distribution_df))

    lines.append("")
    lines.append("## 7. Validation Results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["validation_checks"], indent=4))
    lines.append("```")

    lines.append("")
    lines.append("## 8. Warnings")
    lines.append("")

    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No warnings generated.")

    lines.append("")
    lines.append("## 9. Output Files")
    lines.append("")

    for name, path in summary["output_files"].items():
        lines.append(f"- {name}: `{path}`")

    lines.append("")
    lines.append("## 10. Recommended Next Step")
    lines.append("")
    lines.append(summary["recommended_next_step"])

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main Script
# =========================================================

def main():
    print("=" * 80)
    print("CREATE MULTI-LABEL RECORDS")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Input final split CSV:", FINAL_SPLIT_CSV)

    if not FINAL_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {FINAL_SPLIT_CSV}")

    original_df = pd.read_csv(FINAL_SPLIT_CSV)

    multilabel_df = create_multilabel_records(original_df)

    # -----------------------------------------------------
    # Validation Checks
    # -----------------------------------------------------

    warnings = []

    total_records = int(len(multilabel_df))
    usable_records = int(multilabel_df["usable_for_multiclass"].sum())
    unusable_records = int(total_records - usable_records)

    usable_df = multilabel_df[multilabel_df["usable_for_multiclass"] == True].copy()

    usable_zero_positive_records = int(
        (usable_df["multilabel_num_positive_classes"] == 0).sum()
    )

    if usable_zero_positive_records > 0:
        warnings.append(
            f"{usable_zero_positive_records} usable records have zero positive multi-label targets."
        )

    binary_warnings = check_binary_columns(multilabel_df)
    warnings.extend(binary_warnings)

    train_df = multilabel_df[
        (multilabel_df["final_split"] == "train")
        & (multilabel_df["usable_for_multiclass"] == True)
    ].copy()

    validation_df = multilabel_df[
        (multilabel_df["final_split"] == "validation")
        & (multilabel_df["usable_for_multiclass"] == True)
    ].copy()

    test_df = multilabel_df[
        (multilabel_df["final_split"] == "test")
        & (multilabel_df["usable_for_multiclass"] == True)
    ].copy()

    train_class_presence = {}
    validation_class_presence = {}

    for class_name in TARGET_CLASSES:
        col = MULTILABEL_COLUMNS[class_name]

        train_count = int(train_df[col].sum())
        validation_count = int(validation_df[col].sum())

        train_class_presence[class_name] = train_count > 0
        validation_class_presence[class_name] = validation_count > 0

        if train_count == 0:
            warnings.append(f"Class '{class_name}' has zero positives in train split.")

        if validation_count == 0:
            warnings.append(f"Class '{class_name}' has zero positives in validation split.")

        if train_count < 10:
            warnings.append(
                f"Class '{class_name}' has fewer than 10 positives in train split: {train_count}"
            )

    all_classes_train = all(train_class_presence.values())
    all_classes_validation = all(validation_class_presence.values())

    # -----------------------------------------------------
    # Save CSV Files
    # -----------------------------------------------------

    multilabel_df.to_csv(MULTILABEL_ALL_SPLITS_CSV, index=False)

    multilabel_train_val_df = multilabel_df[
        multilabel_df["final_split"].isin(["train", "validation"])
        & (multilabel_df["usable_for_multiclass"] == True)
    ].copy()

    multilabel_train_val_df.to_csv(MULTILABEL_TRAIN_VAL_CSV, index=False)

    distribution_df = create_distribution_dataframe(multilabel_df)
    distribution_df.to_csv(CLASS_DISTRIBUTION_CSV, index=False)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    split_distribution = {
        str(split): int(count)
        for split, count in multilabel_df["final_split"].value_counts().to_dict().items()
    }

    single_positive_usable_records = int(
        (usable_df["multilabel_num_positive_classes"] == 1).sum()
    )

    multi_positive_usable_records = int(
        (usable_df["multilabel_num_positive_classes"] > 1).sum()
    )

    ambiguous_records = int(multilabel_df["is_ambiguous"].sum())

    per_split_positive_counts = {}

    for split_name in ["train", "validation", "test", "excluded_unusable"]:
        split_df = multilabel_df[multilabel_df["final_split"] == split_name].copy()

        per_split_positive_counts[split_name] = {
            class_name: int(split_df[MULTILABEL_COLUMNS[class_name]].sum())
            for class_name in TARGET_CLASSES
        }

    validation_checks = {
        "usable_records_have_at_least_one_positive_label": usable_zero_positive_records == 0,
        "target_columns_are_binary": len(binary_warnings) == 0,
        "all_classes_present_in_train": bool(all_classes_train),
        "all_classes_present_in_validation": bool(all_classes_validation),
        "train_class_presence": train_class_presence,
        "validation_class_presence": validation_class_presence,
    }

    recommended_next_step = (
        "Proceed to the next multi-label step: create a PyTorch Dataset/DataLoader "
        "pipeline for multilabel_train_val_records.csv using BCEWithLogitsLoss-compatible "
        "multi-hot targets. Do not overwrite the official single-label model."
    )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "input_file": str(FINAL_SPLIT_CSV),
        "target_classes": TARGET_CLASSES,
        "multilabel_vector_order": TARGET_CLASSES,
        "total_records": total_records,
        "usable_records": usable_records,
        "unusable_records": unusable_records,
        "split_distribution": split_distribution,
        "single_positive_usable_records": single_positive_usable_records,
        "multi_positive_usable_records": multi_positive_usable_records,
        "ambiguous_records": ambiguous_records,
        "usable_zero_positive_records": usable_zero_positive_records,
        "per_split_positive_counts": per_split_positive_counts,
        "validation_checks": validation_checks,
        "warnings": warnings,
        "official_single_label_model_untouched": True,
        "training_performed": False,
        "test_evaluation_performed": False,
        "data_raw_modified": False,
        "output_files": {
            "multilabel_records_all_splits_csv": str(MULTILABEL_ALL_SPLITS_CSV),
            "multilabel_train_val_records_csv": str(MULTILABEL_TRAIN_VAL_CSV),
            "summary_json": str(SUMMARY_JSON),
            "report_md": str(REPORT_MD),
            "class_distribution_csv": str(CLASS_DISTRIBUTION_CSV),
        },
        "recommended_next_step": recommended_next_step,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    build_report(summary, distribution_df)

    # -----------------------------------------------------
    # Console Output
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-LABEL RECORDS SUMMARY")
    print("=" * 80)

    print("Total records:", total_records)
    print("Usable records:", usable_records)
    print("Unusable records:", unusable_records)

    print("\nSplit distribution:")
    for split_name, count in split_distribution.items():
        print(f"- {split_name}: {count}")

    print("\nSingle-positive usable records:", single_positive_usable_records)
    print("Multi-positive usable records:", multi_positive_usable_records)
    print("Ambiguous records:", ambiguous_records)
    print("Usable zero-positive records:", usable_zero_positive_records)

    print("\nPer-split positive counts:")
    print(json.dumps(per_split_positive_counts, indent=4))

    print("\nValidation checks:")
    print(json.dumps(validation_checks, indent=4))

    print("\nWarnings:")
    if warnings:
        for warning in warnings:
            print("-", warning)
    else:
        print("- No warnings generated.")

    print("\nSaved output files:")
    print("-", MULTILABEL_ALL_SPLITS_CSV)
    print("-", MULTILABEL_TRAIN_VAL_CSV)
    print("-", SUMMARY_JSON)
    print("-", REPORT_MD)
    print("-", CLASS_DISTRIBUTION_CSV)

    print("\nRecommended next step:")
    print(recommended_next_step)

    print("\nCREATE MULTI-LABEL RECORDS COMPLETED.")


if __name__ == "__main__":
    main()