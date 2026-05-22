# Run Instructions

## 1. Environment Setup

Create and activate a Python environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run The Official Single-Label App

```bash
streamlit run app/streamlit_app.py
```

This app loads:

```text
outputs/models/resnet18_finetuned_layer4_best.pth
```

It predicts one dominant debris class and shows the official single-label
confusion matrix from the held-out test set.

## 3. Run The Dual / Multi-Label App

```bash
streamlit run app/streamlit_dual_model_app.py
```

This app loads:

```text
outputs/models/resnet18_finetuned_layer4_best.pth
outputs/models/multilabel_resnet18_layer4_best.pth
outputs/reports/multilabel_layer4_tuned_thresholds.json
```

It shows the official single-label confusion matrix and the validation
multi-label binary confusion matrices.

## 4. Run The Hugging Face Combined App Locally

```bash
streamlit run app.py
```

The combined app is the entry point intended for Hugging Face Spaces.

## 5. Command-Line Inference

Single-label prediction:

```bash
python src/predict.py --image "path/to/image.jpg"
```

Optional single-label JSON export:

```bash
python src/predict.py --image "path/to/image.jpg" --output "outputs/reports/single_model_prediction_example.json"
```

Dual-model prediction:

```bash
python src/predict_dual_model.py --image "path/to/image.jpg"
```

## 6. Reproduce The Main Pipeline

```bash
python src/check_dataset_pipeline.py
python src/train_baseline.py
python src/train_finetune.py
python src/evaluate_final_test.py
python src/final_submission_audit.py
```

## 7. Reproduce The Experimental Multi-Label Pipeline

```bash
python src/create_multilabel_records.py
python src/check_multilabel_dataset_pipeline.py
python src/train_multilabel_resnet18.py
python src/train_multilabel_resnet18_finetune_layer4.py
python src/tune_multilabel_thresholds.py
```

## 8. Hugging Face Deployment

Use the Docker Space setup described in `huggingface_deployment.md`.

The Space should run:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

## 9. Important Caution

This is a student project prototype. The single-label model has official
held-out test results. The multi-label model is an experimental validation-tuned
extension and should not be treated as a production monitoring system.
