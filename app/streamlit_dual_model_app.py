from pathlib import Path
import datetime
import json
import sys
import time
import uuid

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st


# =========================================================
# Project Path Setup
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from predict_dual_model import predict_dual, TARGET_CLASSES
from report_helpers import (
    load_multilabel_confusion_report,
    load_single_label_confusion_report,
)
import streamlit_ui as project_ui


apply_project_style = project_ui.apply_project_style
render_footer = project_ui.render_footer
render_hero = project_ui.render_hero
render_note = project_ui.render_note
render_section_heading = project_ui.render_section_heading


def render_hint(text: str) -> None:
    if hasattr(project_ui, "render_hint"):
        project_ui.render_hint(text)
    else:
        st.caption(text)


def render_result_banner(label: str, value: str) -> None:
    if hasattr(project_ui, "render_result_banner"):
        project_ui.render_result_banner(label, value)
    else:
        st.success(f"{label}: {value}")


# =========================================================
# Streamlit Page Config
# =========================================================

st.set_page_config(
    page_title="Marine Debris Dual-Model Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_project_style()


# =========================================================
# Paths and Session State
# =========================================================

TEMP_UPLOAD_DIR = PROJECT_ROOT / "outputs" / "temp_dual_model_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "sample_images"
    / "plastic_bottles_dominant.jpg"
)

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "sample_image_path" not in st.session_state:
    st.session_state.sample_image_path = None


# =========================================================
# Cached Report Loading
# =========================================================

@st.cache_data
def get_cached_single_label_confusion_report():
    return load_single_label_confusion_report()


@st.cache_data
def get_cached_multilabel_confusion_report():
    return load_multilabel_confusion_report()


# =========================================================
# Helper Functions
# =========================================================

def class_display_name(class_name: str) -> str:
    return str(class_name).replace("_", " ").title()


def format_probability(value):
    return f"{float(value) * 100:.2f}%"


def make_single_label_dataframe(result):
    probabilities = result["single_label_prediction"]["probabilities_by_class"]

    rows = []
    for class_name in TARGET_CLASSES:
        rows.append({
            "Class": class_display_name(class_name),
            "Probability": float(probabilities[class_name]),
            "Probability (%)": format_probability(probabilities[class_name]),
        })

    return pd.DataFrame(rows).sort_values(
        by="Probability",
        ascending=False,
    ).reset_index(drop=True)


def make_multilabel_dataframe(result):
    details = result["multilabel_prediction"]["per_class_details"]

    rows = []
    for item in details:
        rows.append({
            "Class": class_display_name(item["class_name"]),
            "Probability": float(item["probability"]),
            "Probability (%)": format_probability(item["probability"]),
            "Threshold": float(item["threshold"]),
            "Threshold (%)": format_probability(item["threshold"]),
            "Predicted Positive": bool(item["predicted_positive"]),
        })

    return pd.DataFrame(rows).sort_values(
        by="Probability",
        ascending=False,
    ).reset_index(drop=True)


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    temp_path = TEMP_UPLOAD_DIR / f"dual_model_upload_{uuid.uuid4().hex}{suffix}"

    with open(temp_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return temp_path


def is_uploaded_file(value) -> bool:
    return hasattr(value, "getbuffer")


def show_single_label_confusion_matrix():
    report = get_cached_single_label_confusion_report()
    target_classes = report["target_classes"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Test Accuracy", f"{report['test_accuracy']:.2%}")
    with metric_col2:
        st.metric("Macro-F1", f"{report['macro_f1']:.2%}")
    with metric_col3:
        st.metric("Weighted-F1", f"{report['weighted_f1']:.2%}")

    matrix_col, table_col = st.columns([1.08, 0.92], gap="large")

    with matrix_col:
        if report["plot_path"].exists():
            st.image(
                str(report["plot_path"]),
                caption="Single-label final held-out test confusion matrix",
                width="stretch",
            )

    with table_col:
        st.markdown("#### Matrix Values")
        if report["matrix"]:
            matrix_df = pd.DataFrame(
                report["matrix"],
                index=[f"Actual {class_display_name(name)}" for name in target_classes],
                columns=[
                    f"Predicted {class_display_name(name)}"
                    for name in target_classes
                ],
            )
            st.dataframe(matrix_df, width="stretch")

        st.caption(
            "Rows are true labels and columns are predicted labels. "
            "This is the official held-out test matrix."
        )

    with st.expander("Normalized single-label confusion matrix"):
        if report["normalized_plot_path"].exists():
            st.image(
                str(report["normalized_plot_path"]),
                width="stretch",
            )

        if report["normalized_matrix"]:
            normalized_df = pd.DataFrame(
                report["normalized_matrix"],
                index=[f"Actual {class_display_name(name)}" for name in target_classes],
                columns=[
                    f"Predicted {class_display_name(name)}"
                    for name in target_classes
                ],
            )
            st.dataframe(normalized_df, width="stretch")


def show_multilabel_confusion_matrix():
    report = get_cached_multilabel_confusion_report()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Validation Subset Accuracy", f"{report['subset_accuracy']:.2%}")
    with metric_col2:
        st.metric("Validation Hamming Accuracy", f"{report['hamming_accuracy']:.2%}")
    with metric_col3:
        st.metric("Validation Macro-F1", f"{report['macro_f1']:.2%}")

    matrix_col, summary_col = st.columns([0.9, 1.1], gap="large")

    with matrix_col:
        st.markdown("#### Binary Matrices By Class")
        class_tabs = st.tabs([
            class_display_name(name)
            for name in report["target_classes"]
        ])

        for tab, class_name in zip(class_tabs, report["target_classes"]):
            with tab:
                matrix = report["matrices"][class_name]
                matrix_df = pd.DataFrame(
                    matrix,
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Predicted Negative", "Predicted Positive"],
                )
                st.dataframe(matrix_df, width="stretch")
                threshold = report["thresholds_by_class"].get(class_name)
                if threshold is not None:
                    st.caption(
                        f"Decision threshold for {class_display_name(class_name)}: "
                        f"{float(threshold):.2f}"
                    )

    with summary_col:
        st.markdown("#### Per-Class Summary")
        summary_df = pd.DataFrame(report["rows"])
        if not summary_df.empty:
            summary_df["class"] = summary_df["class"].apply(class_display_name)
            st.dataframe(summary_df, width="stretch", hide_index=True)

        render_note(
            "These multi-label matrices use validation data. They are not official test results."
        )


def show_confusion_matrices():
    render_section_heading(
        "Evaluation Summary",
        "Review the official single-label matrix and the experimental multi-label validation matrices.",
    )

    with st.container(border=True):
        single_tab, multilabel_tab = st.tabs([
            "Single-Label Model",
            "Multi-Label Model",
        ])

        with single_tab:
            show_single_label_confusion_matrix()

        with multilabel_tab:
            show_multilabel_confusion_matrix()


def render_prediction_tabs(result, prediction_time: float):
    primary_class = result["single_label_prediction"]["predicted_class"]
    primary_confidence = result["single_label_prediction"]["confidence"]
    positive_labels = result["multilabel_prediction"]["positive_labels"]
    fallback_used = result["multilabel_prediction"]["fallback_used"]

    single_df = make_single_label_dataframe(result)
    multi_df = make_multilabel_dataframe(result)

    tab1, tab2, tab3 = st.tabs([
        "Primary Prediction",
        "Additional Debris",
        "Interpretation",
    ])

    with tab1:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            render_result_banner("Primary Class", class_display_name(primary_class))
        with metric_col2:
            st.metric("Confidence", format_probability(primary_confidence))
            render_hint("Confidence is the model probability for the primary class.")

        st.markdown("#### Class Probabilities")
        st.dataframe(single_df, width="stretch", hide_index=True)
        st.bar_chart(
            single_df.set_index("Class")["Probability"],
            width="stretch",
        )

        st.download_button(
            "Download single-label results",
            data=single_df.to_csv(index=False).encode("utf-8"),
            file_name="dual_model_singlelabel_results.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab2:
        if fallback_used:
            st.warning(
                "No class passed its threshold, so the highest-probability class was shown."
            )

        if positive_labels:
            st.success(
                "Possible debris types: "
                + ", ".join(class_display_name(label) for label in positive_labels)
            )
        else:
            st.warning("No additional debris type was detected.")

        st.markdown("#### Probabilities and Thresholds")
        st.dataframe(multi_df, width="stretch", hide_index=True)
        st.bar_chart(
            multi_df.set_index("Class")["Probability"],
            width="stretch",
        )

        st.download_button(
            "Download multi-label results",
            data=multi_df.to_csv(index=False).encode("utf-8"),
            file_name="dual_model_multilabel_results.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab3:
        st.metric("Prediction Time", f"{prediction_time:.2f}s")
        st.write(
            f"Primary result: **{class_display_name(primary_class)}** "
            f"with **{format_probability(primary_confidence)}** confidence."
        )
        st.write(
            "Possible debris types: "
            f"**{', '.join(class_display_name(label) for label in positive_labels) or 'None'}**."
        )
        render_note(
            "Academic prototype. Use these predictions for demonstration, not certified monitoring."
        )

        with st.expander("Show raw prediction JSON"):
            st.json(result)
            st.download_button(
                "Download full results as JSON",
                data=json.dumps(result, indent=2).encode("utf-8"),
                file_name="dual_model_full_results.json",
                mime="application/json",
                width="stretch",
            )

        if not st.session_state.prediction_done:
            st.balloons()
            st.session_state.prediction_done = True


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("Dual-Model Classifier")
    st.caption("Main class plus possible debris types")

    st.divider()
    st.subheader("Model Files")
    st.write("Official single-label model")
    st.code("resnet18_finetuned_layer4_best.pth")
    st.write("Experimental multi-label model")
    st.code("multilabel_resnet18_layer4_best.pth")

    st.divider()
    st.subheader("Thresholds")
    threshold_df = pd.DataFrame([
        {"Class": "Plastic", "Threshold": 0.25},
        {"Class": "Foam", "Threshold": 0.25},
        {"Class": "Metal", "Threshold": 0.75},
        {"Class": "Other Debris", "Threshold": 0.15},
    ])
    st.dataframe(threshold_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Official Single-Label Results")
    side_col1, side_col2 = st.columns(2)
    with side_col1:
        st.metric("Accuracy", "81.61%")
        st.metric("Macro-F1", "69.29%")
    with side_col2:
        st.metric("Weighted-F1", "82.24%")

    st.divider()
    with st.expander("Important Scope", expanded=False):
        st.write("Official result: single-label model.")
        st.write("Experimental result: multi-label model.")

    with st.expander("Feedback", expanded=False):
        feedback = st.text_area("Optional feedback:", height=100)
        if st.button("Submit Feedback", width="stretch"):
            if feedback.strip():
                with open(PROJECT_ROOT / "feedback.txt", "a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now()}: {feedback.strip()}\n")
                st.success("Feedback saved.")
            else:
                st.warning("Please enter feedback before submitting.")


# =========================================================
# Main Page
# =========================================================

render_hero(
    title="Marine Debris Dual-Model Classifier",
    subtitle=(
        "Upload one image. The app predicts the main debris class and checks "
        "for other visible debris types."
    ),
    eyebrow="University ML Project Demonstration",
    tags=["ResNet18", "Single-label", "Multi-label", "Threshold tuning"],
)

render_note(
    "The single-label model is the official evaluated model. The multi-label output is experimental."
)

show_confusion_matrices()

render_section_heading(
    "Analyze an Image",
    "Upload an image or use the sample image, then compare both model outputs.",
)

with st.container(border=True):
    upload_col, preview_col, action_col = st.columns([0.95, 0.95, 0.85], gap="large")

    with upload_col:
        st.markdown("#### Image Source")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, PNG, and WebP.",
        )

        if uploaded_file is not None:
            st.session_state.sample_image_path = None

        sample_clicked = st.button(
            "Try Sample Image",
            help="Use the built-in plastic bottle sample image.",
            width="stretch",
        )
        if sample_clicked:
            if SAMPLE_IMAGE_PATH.exists():
                st.session_state.sample_image_path = str(SAMPLE_IMAGE_PATH)
                st.success("Plastic bottle sample selected.")
            else:
                st.error("Sample image asset not found.")

        reset_clicked = st.button(
            "Reset App",
            help="Clear all results and start fresh.",
            width="stretch",
        )
        if reset_clicked:
            st.session_state.prediction_done = False
            st.session_state.sample_image_path = None
            st.rerun()

    image_source = uploaded_file
    if image_source is None and st.session_state.sample_image_path:
        sample_path = Path(st.session_state.sample_image_path)
        if sample_path.exists():
            image_source = sample_path

    with preview_col:
        st.markdown("#### Preview")
        if image_source is None:
            st.info("Upload an image or select the sample.")
            image = None
            image_path_for_prediction = None
        else:
            try:
                if is_uploaded_file(image_source):
                    image = ImageOps.exif_transpose(Image.open(image_source)).convert("RGB")
                    image_path_for_prediction = save_uploaded_file(image_source)
                else:
                    image_path_for_prediction = Path(image_source)
                    image = ImageOps.exif_transpose(Image.open(image_path_for_prediction)).convert("RGB")

                st.image(
                    image,
                    width="stretch",
                    caption=f"Selected image ({image.width}x{image.height})",
                )
            except Exception as exc:
                image = None
                image_path_for_prediction = None
                st.error("The selected image could not be loaded.")
                with st.expander("Error Details"):
                    st.exception(exc)

    with action_col:
        st.markdown("#### Action")
        if image_source is None:
            st.info("Select an image to enable prediction.")
            predict_button = False
        else:
            st.write("Run both models on the selected image.")
            predict_button = st.button(
                "Analyze Image",
                type="primary",
                width="stretch",
            )

if image_source is not None and predict_button:
    if image_path_for_prediction is None:
        st.error("Prediction failed because the image could not be prepared.")
    else:
        try:
            with st.status("Analyzing image...", expanded=True) as status:
                start_time = time.time()
                result = predict_dual(image_path_for_prediction)
                prediction_time = time.time() - start_time
                status.update(
                    label=f"Analysis complete in {prediction_time:.2f}s.",
                    state="complete",
                    expanded=False,
                )

            render_section_heading(
                "Prediction Results",
                "Primary class, possible debris types, exports, and raw output.",
            )
            with st.container(border=True):
                render_prediction_tabs(result, prediction_time)

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

elif image_source is not None:
    st.caption("Ready to analyze the selected image.")


st.divider()
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
render_footer(
    "Marine Debris Dual-Model Classifier | Academic ML Prototype | "
    f"Streamlit | Updated {current_time}"
)
