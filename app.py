from pathlib import Path
import sys
import time
import uuid

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from predict import load_trained_model, predict_pil_image
from predict_dual_model import predict_dual
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


TEMP_UPLOAD_DIR = PROJECT_ROOT / "outputs" / "temp_hf_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "sample_images"
    / "plastic_bottles_dominant.jpg"
)


st.set_page_config(
    page_title="Marine Debris Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_project_style()

if "dual_sample_image_path" not in st.session_state:
    st.session_state.dual_sample_image_path = None


@st.cache_resource
def get_single_label_model():
    model, device, checkpoint = load_trained_model()
    return model, device, checkpoint


@st.cache_data
def get_single_confusion_report():
    return load_single_label_confusion_report()


@st.cache_data
def get_multilabel_confusion_report():
    return load_multilabel_confusion_report()


def class_display_name(class_name: str) -> str:
    return str(class_name).replace("_", " ").title()


def format_probability(value) -> str:
    return f"{float(value) * 100:.2f}%"


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    output_path = TEMP_UPLOAD_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    with open(output_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return output_path


def is_uploaded_file(value) -> bool:
    return hasattr(value, "getbuffer") and hasattr(value, "name")


def show_single_label_confusion_matrix():
    report = get_single_confusion_report()
    target_classes = report["target_classes"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Accuracy", f"{report['test_accuracy']:.2%}")
    with metric_col2:
        st.metric("Macro-F1", f"{report['macro_f1']:.2%}")
    with metric_col3:
        st.metric("Weighted-F1", f"{report['weighted_f1']:.2%}")

    matrix_col, table_col = st.columns([1.08, 0.92], gap="large")
    with matrix_col:
        if report["plot_path"].exists():
            st.image(
                str(report["plot_path"]),
                caption="Final held-out test confusion matrix",
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

        st.caption("Rows are true labels and columns are predicted labels.")


def show_multilabel_confusion_matrix():
    report = get_multilabel_confusion_report()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Subset Accuracy", f"{report['subset_accuracy']:.2%}")
    with metric_col2:
        st.metric("Hamming Accuracy", f"{report['hamming_accuracy']:.2%}")
    with metric_col3:
        st.metric("Macro-F1", f"{report['macro_f1']:.2%}")

    matrix_col, summary_col = st.columns([0.9, 1.1], gap="large")

    with matrix_col:
        st.markdown("#### Binary Matrices By Class")
        tabs = st.tabs([class_display_name(name) for name in report["target_classes"]])
        for tab, class_name in zip(tabs, report["target_classes"]):
            with tab:
                matrix_df = pd.DataFrame(
                    report["matrices"][class_name],
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Predicted Negative", "Predicted Positive"],
                )
                st.dataframe(matrix_df, width="stretch")

    with summary_col:
        st.markdown("#### Per-Class Summary")
        rows_df = pd.DataFrame(report["rows"])
        if not rows_df.empty:
            rows_df["class"] = rows_df["class"].apply(class_display_name)
            st.dataframe(rows_df, width="stretch", hide_index=True)

        render_note(
            "These multi-label matrices use validation data. They are not official test results."
        )


def render_confusion_matrices(show_multilabel: bool):
    render_section_heading(
        "Model Evaluation Matrices",
        "Review model performance before running a prediction.",
    )

    with st.container(border=True):
        if not show_multilabel:
            show_single_label_confusion_matrix()
            return

        single_tab, multi_tab = st.tabs(["Single-Label Model", "Multi-Label Model"])
        with single_tab:
            show_single_label_confusion_matrix()
        with multi_tab:
            show_multilabel_confusion_matrix()


def render_single_label_app():
    render_hero(
        title="Single-Label Marine Debris Classifier",
        subtitle=(
            "Upload one image. The official model predicts the main debris type."
        ),
        eyebrow="Deployment Demo",
        tags=["Official model", "Four classes", "CSV export"],
    )

    render_confusion_matrices(show_multilabel=False)

    render_section_heading(
        "Classify an Image",
        "Upload one image to view the predicted class, confidence, and probabilities.",
    )

    with st.container(border=True):
        upload_col, result_col = st.columns([0.95, 1.05], gap="large")

        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="single_upload",
            )

            if uploaded_file is None:
                st.info("Upload an image to begin.")

        with result_col:
            if uploaded_file is None:
                st.info("Prediction results will appear here.")
                return

            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            preview_col, action_col = st.columns([0.9, 1.1], gap="medium")

            with preview_col:
                st.image(image, width="stretch", caption="Uploaded image")

            with action_col:
                if st.button(
                    "Classify Image",
                    type="primary",
                    width="stretch",
                ):
                    with st.status("Classifying image...", expanded=True) as status:
                        model, device, checkpoint = get_single_label_model()
                        result = predict_pil_image(
                            image=image,
                            model=model,
                            device=device,
                            top_k=4,
                        )
                        status.update(label="Classification complete.", state="complete")

                    render_result_banner(
                        "Predicted Class",
                        class_display_name(result["predicted_label"]),
                    )
                    st.metric("Confidence", format_probability(result["confidence"]))
                    render_hint(
                        "Confidence is the model probability for the predicted class."
                    )

                    probabilities_df = pd.DataFrame([
                        {
                            "Class": class_display_name(item["label"]),
                            "Probability": item["probability"],
                            "Probability (%)": format_probability(item["probability"]),
                        }
                        for item in result["top_k_predictions"]
                    ])
                    table_tab, chart_tab = st.tabs(["Table", "Chart"])
                    with table_tab:
                        st.dataframe(
                            probabilities_df,
                            width="stretch",
                            hide_index=True,
                        )
                    with chart_tab:
                        st.bar_chart(
                            probabilities_df.set_index("Class")["Probability"],
                            width="stretch",
                        )


def render_dual_model_app():
    render_hero(
        title="Dual / Multi-Label Marine Debris Classifier",
        subtitle=(
            "Upload one image. The app predicts the main class and checks for other visible debris types."
        ),
        eyebrow="Deployment Demo",
        tags=["Single-label", "Multi-label", "Thresholds", "Raw JSON"],
    )

    render_note(
        "The single-label result is official. The multi-label result is experimental."
    )

    render_confusion_matrices(show_multilabel=True)

    render_section_heading(
        "Analyze an Image",
        "Upload one image and compare both model outputs.",
    )

    with st.container(border=True):
        upload_col, preview_col, action_col = st.columns([0.9, 0.95, 0.95], gap="large")

        with upload_col:
            st.markdown("#### Image Source")
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
                key="dual_upload",
            )

            if uploaded_file is not None:
                st.session_state.dual_sample_image_path = None

            sample_clicked = st.button(
                "Try Sample Image",
                help="Use the built-in plastic bottle sample image.",
                key="dual_sample_button",
                width="stretch",
            )
            if sample_clicked:
                if SAMPLE_IMAGE_PATH.exists():
                    st.session_state.dual_sample_image_path = str(SAMPLE_IMAGE_PATH)
                    st.success("Plastic bottle sample selected.")
                else:
                    st.error("Sample image asset not found.")

            if uploaded_file is None and not st.session_state.dual_sample_image_path:
                st.info("Upload an image or use the sample image.")

        image_source = uploaded_file
        if image_source is None and st.session_state.dual_sample_image_path:
            sample_path = Path(st.session_state.dual_sample_image_path)
            if sample_path.exists():
                image_source = sample_path

        if image_source is None:
            return

        if is_uploaded_file(image_source):
            image = ImageOps.exif_transpose(Image.open(image_source)).convert("RGB")
            image_path = save_uploaded_file(image_source)
            preview_caption = "Uploaded image"
        else:
            image_path = Path(image_source)
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            preview_caption = "Sample image"

        with preview_col:
            st.image(image, width="stretch", caption=preview_caption)

        with action_col:
            run_clicked = st.button(
                "Analyze Image",
                type="primary",
                width="stretch",
            )
            if not run_clicked:
                st.caption("Ready to analyze the selected image.")
                return

            with st.status("Analyzing image...", expanded=True) as status:
                start_time = time.time()
                result = predict_dual(image_path)
                prediction_time = time.time() - start_time
                status.update(
                    label=f"Analysis complete in {prediction_time:.2f}s.",
                    state="complete",
                )

    single = result["single_label_prediction"]
    multi = result["multilabel_prediction"]

    render_section_heading(
        "Prediction Results",
        "Primary class, possible debris types, score tables, and raw output.",
    )

    with st.container(border=True):
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            render_result_banner("Primary Class", class_display_name(single["predicted_class"]))
        with summary_col2:
            st.metric("Primary Confidence", format_probability(single["confidence"]))
            render_hint("Confidence is the model probability for the primary class.")
        with summary_col3:
            st.metric("Prediction Time", f"{prediction_time:.2f}s")

        visible_debris = ", ".join(
            class_display_name(label)
            for label in multi["positive_labels"]
        )
        st.success(f"Possible debris types: {visible_debris}")
        if multi["fallback_used"]:
            st.warning("No class passed its threshold, so the highest-probability class was shown.")

        single_df = pd.DataFrame([
            {
                "Class": class_display_name(name),
                "Probability": probability,
                "Probability (%)": format_probability(probability),
            }
            for name, probability in single["probabilities_by_class"].items()
        ]).sort_values(by="Probability", ascending=False)

        multi_df = pd.DataFrame([
            {
                "Class": class_display_name(item["class_name"]),
                "Probability": item["probability"],
                "Threshold": item["threshold"],
                "Predicted Positive": item["predicted_positive"],
            }
            for item in multi["per_class_details"]
        ]).sort_values(by="Probability", ascending=False)

        tab1, tab2, tab3 = st.tabs([
            "Single-Label Scores",
            "Multi-Label Scores",
            "Raw JSON",
        ])
        with tab1:
            st.dataframe(single_df, width="stretch", hide_index=True)
            st.bar_chart(single_df.set_index("Class")["Probability"])
        with tab2:
            st.dataframe(multi_df, width="stretch", hide_index=True)
            st.bar_chart(multi_df.set_index("Class")["Probability"])
        with tab3:
            st.json(result)


with st.sidebar:
    st.title("Marine Debris Demo")
    app_mode = st.radio(
        "Choose application",
        ["Single-label model", "Dual / multi-label models"],
    )

    st.divider()
    st.subheader("Scope")
    st.write("Official: single-label ResNet18")
    st.write("Experimental: multi-label ResNet18")

    st.divider()
    render_note("Academic prototype. Predictions are for demonstration only.")


if app_mode == "Single-label model":
    render_single_label_app()
else:
    render_dual_model_app()


st.divider()
render_footer("Marine Debris Classifier | Academic ML Prototype | Streamlit | 2026")
