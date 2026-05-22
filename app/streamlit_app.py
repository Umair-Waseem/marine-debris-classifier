from pathlib import Path
import datetime
import sys

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# =========================================================
# Make src/ importable
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from predict import load_trained_model, predict_pil_image, TARGET_CLASSES
from report_helpers import load_single_label_confusion_report
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
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Marine Debris Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_project_style()


# =========================================================
# Cached Loading
# =========================================================

@st.cache_resource
def get_cached_model():
    model, device, checkpoint = load_trained_model()
    return model, device, checkpoint


@st.cache_data
def get_cached_single_label_confusion_report():
    return load_single_label_confusion_report()


if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


# =========================================================
# Display Helpers
# =========================================================

def class_display_name(class_name: str) -> str:
    return str(class_name).replace("_", " ").title()


def show_single_label_confusion_matrix():
    report = get_cached_single_label_confusion_report()
    target_classes = report["target_classes"]

    render_section_heading(
        "Evaluation Summary",
        "Official test results for the single-label ResNet18 model.",
    )

    with st.container(border=True):
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Test Accuracy", f"{report['test_accuracy']:.2%}")
        with metric_col2:
            st.metric("Macro-F1", f"{report['macro_f1']:.2%}")
        with metric_col3:
            st.metric("Weighted-F1", f"{report['weighted_f1']:.2%}")

        matrix_col, table_col = st.columns([1.08, 0.92], gap="large")

        with matrix_col:
            plot_path = report["plot_path"]
            if plot_path.exists():
                st.image(
                    str(plot_path),
                    caption="Final held-out test confusion matrix",
                    width="stretch",
                )

        with table_col:
            st.markdown("#### Matrix Values")
            matrix = report["matrix"]
            if matrix:
                matrix_df = pd.DataFrame(
                    matrix,
                    index=[
                        f"Actual {class_display_name(name)}"
                        for name in target_classes
                    ],
                    columns=[
                        f"Predicted {class_display_name(name)}"
                        for name in target_classes
                    ],
                )
                st.dataframe(matrix_df, width="stretch")

            st.caption(
                "Rows show the true class. Columns show the predicted class."
            )

        with st.expander("View normalized matrix"):
            normalized_col, normalized_table_col = st.columns([1.08, 0.92], gap="large")
            with normalized_col:
                normalized_plot_path = report["normalized_plot_path"]
                if normalized_plot_path.exists():
                    st.image(
                        str(normalized_plot_path),
                        caption="Row-normalized final test confusion matrix",
                        width="stretch",
                    )

            with normalized_table_col:
                normalized_matrix = report["normalized_matrix"]
                if normalized_matrix:
                    normalized_df = pd.DataFrame(
                        normalized_matrix,
                        index=[
                            f"Actual {class_display_name(name)}"
                            for name in target_classes
                        ],
                        columns=[
                            f"Predicted {class_display_name(name)}"
                            for name in target_classes
                        ],
                    )
                    st.dataframe(normalized_df, width="stretch")


def show_prediction_results(prediction: dict):
    render_result_banner(
        "Predicted Class",
        class_display_name(prediction["predicted_label"]),
    )
    st.metric("Confidence", f"{prediction['confidence']:.2%}")
    render_hint(
        "Confidence is the model's estimated probability for the predicted class."
    )

    prob_df = pd.DataFrame([
        {
            "Class": class_display_name(item["label"]),
            "Probability": item["probability"],
        }
        for item in prediction["top_k_predictions"]
    ])

    table_tab, chart_tab = st.tabs(["Table", "Chart"])

    with table_tab:
        st.dataframe(prob_df, width="stretch", hide_index=True)

    with chart_tab:
        st.bar_chart(prob_df.set_index("Class"), width="stretch")

    st.download_button(
        label="Download results as CSV",
        data=prob_df.to_csv(index=False),
        file_name="marine_debris_prediction_results.csv",
        mime="text/csv",
        help="Save the probability table as a CSV file.",
        width="stretch",
    )

    render_note(
        "Academic prototype. Use this result for demonstration, not certified monitoring."
    )

    if not st.session_state.prediction_done:
        st.balloons()
        st.session_state.prediction_done = True


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("Marine Debris Classifier")
    st.caption("Single-label image classification")

    st.divider()
    st.subheader("Model")
    st.write("Architecture: ResNet18")
    st.write("Task: predict one main class")
    st.write("Fine-tuned layers: layer4 and classifier")

    st.subheader("Classes")
    for class_name in TARGET_CLASSES:
        st.write(f"- {class_display_name(class_name)}")

    st.divider()
    st.subheader("Final Test Results")
    side_col1, side_col2 = st.columns(2)
    with side_col1:
        st.metric("Accuracy", "81.61%")
        st.metric("Macro-F1", "69.29%")
    with side_col2:
        st.metric("Weighted-F1", "82.24%")

    st.divider()
    with st.expander("Limitations", expanded=False):
        st.write("- Small dataset")
        st.write("- Class imbalance")
        st.write("- Few metal examples")
        st.write("- One label per image")

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
# Main UI
# =========================================================

render_hero(
    title="Marine Debris Image Classifier",
    subtitle=(
        "Upload a marine debris image. The model predicts the main debris type: "
        "plastic, foam, metal, or other debris."
    ),
    eyebrow="University ML Project Demonstration",
    tags=["ResNet18", "Single-label inference", "Held-out test evaluation"],
)

render_note(
    "Academic prototype. Results are for project demonstration and should not be used as certified environmental monitoring."
)

show_single_label_confusion_matrix()

render_section_heading(
    "Classify an Image",
    "Upload one image to view the predicted class, confidence, and class probabilities.",
)

with st.container(border=True):
    upload_col, result_col = st.columns([0.95, 1.05], gap="large")

    with upload_col:
        st.markdown("#### Image")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, PNG, BMP, and WebP.",
        )

        reset_clicked = st.button(
            "Reset App",
            help="Clear all results and start fresh.",
            width="stretch",
        )
        if reset_clicked:
            st.session_state.prediction_done = False
            st.rerun()

        if uploaded_file is None:
            st.info("Upload an image to begin.")

    with result_col:
        st.markdown("#### Prediction")

        if uploaded_file is None:
            st.info("Prediction results will appear here.")
        else:
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
                image_col, action_col = st.columns([0.9, 1.1], gap="medium")

                with image_col:
                    st.image(
                        image,
                        width="stretch",
                        caption="Uploaded image",
                    )

                with action_col:
                    run_clicked = st.button(
                        "Classify Image",
                        type="primary",
                        width="stretch",
                    )

                    if run_clicked:
                        with st.status("Classifying image...", expanded=True) as status:
                            model, device, checkpoint = get_cached_model()
                            prediction = predict_pil_image(
                                image=image,
                                model=model,
                                device=device,
                                top_k=4,
                            )
                            status.update(
                                label="Classification complete.",
                                state="complete",
                                expanded=False,
                            )

                        show_prediction_results(prediction)
                    else:
                        st.caption("Ready to classify this image.")

            except Exception as exc:
                st.error("The image could not be processed. Try another file.")
                with st.expander("Error Details"):
                    st.exception(exc)


st.divider()
render_footer(
    "Marine Debris Classifier | Academic ML Prototype | Streamlit | 2026"
)
