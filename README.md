---

title: Marine Debris Classifier
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
------------

# Marine Debris Classifier Using Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A ResNet18-based deep learning project for classifying marine debris images into `plastic`, `foam`, `metal`, and `other_debris`.

This repository includes dataset preprocessing, model training, fine-tuning, final evaluation, command-line inference, Streamlit apps, Docker deployment, and an experimental multi-label extension.

---

## Overview

Marine debris affects coastal and underwater ecosystems. This project uses computer vision to identify the dominant debris type in an image and support future environmental monitoring, cleanup planning, and automated waste-detection systems.

The official system is a **single-label multi-class classifier**. An experimental dual-model version is also included for predicting possible additional visible debris types.

---

## Key Features

* Complete marine debris image classification pipeline
* Dataset inspection, cleaning, and preprocessing
* Train, validation, and test split preparation
* ResNet18 baseline training and layer4 fine-tuning
* Final held-out test evaluation
* Streamlit inference applications
* Command-line prediction scripts
* Docker-ready Hugging Face deployment
* Experimental multi-label and dual-model support

---

## Target Classes

| Class          | Description                                         |
| -------------- | --------------------------------------------------- |
| `plastic`      | Plastic bags, bottles, packaging, and related waste |
| `foam`         | Polystyrene foam and foam-like debris               |
| `metal`        | Cans, metallic objects, and metal debris            |
| `other_debris` | Glass, rubber, mixed, or unidentified debris        |

---

## Official Model

The official model is a ResNet18-based single-label classifier fine-tuned for four marine debris classes.

```text
outputs/models/resnet18_finetuned_layer4_best.pth
```

### Final Test Results

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 81.61% |
| Macro-F1    | 69.29% |
| Weighted-F1 | 82.24% |

Regenerate the final evaluation:

```bash
python src/evaluate_final_test.py
```

---

## Experimental Multi-Label Extension

The experimental workflow combines the official single-label model with a separate multi-label model.

| Model              | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Single-label model | Predicts the dominant debris class       |
| Multi-label model  | Predicts additional visible debris types |

```text
outputs/models/multilabel_resnet18_layer4_best.pth
outputs/reports/multilabel_layer4_tuned_thresholds.json
```

| Class          | Threshold |
| -------------- | --------: |
| `plastic`      |      0.25 |
| `foam`         |      0.25 |
| `metal`        |      0.75 |
| `other_debris` |      0.15 |

This branch is experimental and does not replace the official single-label test result.

---

## Dataset Summary

The processed dataset contains **575 records**. One unusable record was excluded, leaving **574 usable images**.

| Split      | Images |
| ---------- | -----: |
| Train      |    400 |
| Validation |     87 |
| Test       |     87 |
| Excluded   |      1 |

### Class Distribution

| Class          | Train | Validation | Test |
| -------------- | ----: | ---------: | ---: |
| `plastic`      |   264 |         57 |   57 |
| `foam`         |    46 |         10 |   10 |
| `metal`        |    13 |          3 |    3 |
| `other_debris` |    77 |         17 |   17 |

The dataset is small and imbalanced, especially for the `metal` class.

---

## Labeling Policy

* Original annotations were treated as ground truth.
* The official model uses one dominant label per image.
* The experimental branch uses multi-hot labels.
* No manual relabeling was performed for the official version.

---

## Project Structure

```text
marine-debris-classifier/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- Dockerfile
|-- app.py
|-- app/
|   |-- streamlit_app.py
|   `-- streamlit_dual_model_app.py
|-- data/processed/records/
|   |-- final_train_val_test_split.csv
|   |-- multilabel_records_all_splits.csv
|   `-- multilabel_train_val_records.csv
|-- outputs/
|   |-- models/
|   |-- reports/
|   `-- plots/
|-- src/
|   |-- train_baseline.py
|   |-- train_finetune.py
|   |-- evaluate_final_test.py
|   |-- predict.py
|   |-- predict_dual_model.py
|   |-- train_multilabel_resnet18.py
|   |-- train_multilabel_resnet18_finetune_layer4.py
|   `-- tune_multilabel_thresholds.py
`-- project_report.md
```

Raw datasets, virtual environments, temporary files, and bulky regenerated outputs are excluded. Final model weights are tracked with Git LFS because they exceed GitHub’s standard file size limit.

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows PowerShell
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Main libraries include Python 3.9+, PyTorch, Torchvision, Pandas, NumPy, Pillow, Matplotlib, Streamlit, tqdm, python-docx, ReportLab, and huggingface-hub.

---

## Running the Apps

```bash
# Combined app
streamlit run app.py

# Official single-label app
streamlit run app/streamlit_app.py

# Experimental dual-model app
streamlit run app/streamlit_dual_model_app.py
```

---

## Command-Line Prediction

```bash
# Single-label prediction
python src/predict.py --image "path/to/image.jpg"

# Save prediction as JSON
python src/predict.py --image "path/to/image.jpg" --output "outputs/reports/single_model_prediction_example.json"

# Dual-model prediction
python src/predict_dual_model.py --image "path/to/image.jpg"
```

---

## Reproducing the Pipeline

### Official Workflow

```bash
python src/check_dataset_pipeline.py
python src/train_baseline.py
python src/train_finetune.py
python src/evaluate_final_test.py
python src/final_submission_audit.py
```

### Experimental Multi-Label Workflow

```bash
python src/create_multilabel_records.py
python src/check_multilabel_dataset_pipeline.py
python src/train_multilabel_resnet18.py
python src/train_multilabel_resnet18_finetune_layer4.py
python src/tune_multilabel_thresholds.py
```

---

## Results Interpretation

The model achieved **81.61% accuracy** and **82.24% Weighted-F1** on the held-out test set. Macro-F1 is lower at **69.29%** because the dataset is imbalanced and minority classes, especially `metal`, have very few samples.

---

## Hugging Face Deployment

This project is prepared for Docker-based Hugging Face Spaces deployment.

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

Deployment details are available in:

```text
huggingface_deployment.md
```

---

## Limitations

* Small and imbalanced dataset
* Very few `metal` class samples
* Many images contain multiple debris types
* Official model predicts only one dominant class
* Multi-label branch is experimental and validation-tuned
* Streamlit apps are academic prototypes, not production systems

---

## Future Work

* Collect more images for underrepresented classes
* Add expert label verification
* Train an object detection model for localized debris detection
* Evaluate the multi-label model on a separate unseen test set
* Add batch prediction, logging, and model explainability

---

## Conclusion

This project provides a complete marine debris classification prototype using ResNet18 transfer learning. The official model achieved **81.61% test accuracy** with a reproducible workflow for preprocessing, training, evaluation, inference, Streamlit demonstration, and Docker deployment.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

* Dataset metadata references Kaggle and Roboflow exports.
* Built using PyTorch and Streamlit.
