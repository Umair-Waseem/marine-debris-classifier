---
title: Marine Debris Classifier
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
---

# Marine Debris Classifier Using Deep Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning system for classifying marine debris images into `plastic`,
`foam`, `metal`, and `other_debris` using ResNet18 transfer learning.

## Project Overview

This project implements an image-classification workflow for marine debris:

- Dataset inspection and quality checks
- Annotation conversion and dominant-label policy
- Train/validation/test splitting
- ResNet18 baseline training and layer4 fine-tuning
- Final held-out test evaluation
- Streamlit inference apps
- Optional experimental multi-label / dual-model extension

The official submitted model is a single-label classifier. The experimental
dual-model workflow keeps the official single-label model for the primary class
and adds a multi-label model for possible additional visible debris types.

## Project Status

- Full pipeline completed
- Official single-label model trained and evaluated
- Experimental multi-label system implemented
- Streamlit apps ready for local or Docker deployment

## Target Classes

| Class | Description |
|---|---|
| `plastic` | Plastic bags, bottles, packaging |
| `foam` | Polystyrene foam fragments |
| `metal` | Cans and metallic objects |
| `other_debris` | Glass, rubber, mixed, or unidentified debris |

## Official Final Model

Model file:

```text
outputs/models/resnet18_finetuned_layer4_best.pth
```

Official held-out test results:

| Metric | Score |
|---|---:|
| Accuracy | 81.61% |
| Macro-F1 | 69.29% |
| Weighted-F1 | 82.24% |

Detailed evaluation artifacts can be regenerated with `python src/evaluate_final_test.py`.

## Experimental Dual-Model Extension

| Model | Purpose |
|---|---|
| Single-label model | Predicts the main / dominant debris class |
| Multi-label model | Predicts possible additional visible debris types |

Multi-label model:

```text
outputs/models/multilabel_resnet18_layer4_best.pth
```

Tuned thresholds:

```text
outputs/reports/multilabel_layer4_tuned_thresholds.json
```

| Class | Threshold |
|---|---:|
| `plastic` | 0.25 |
| `foam` | 0.25 |
| `metal` | 0.75 |
| `other_debris` | 0.15 |

This branch is experimental and does not replace the official single-label
test result.

## Dataset Summary

The processed dataset contains 575 records. One record is excluded as unusable,
leaving 574 usable images.

| Split | Images |
|---|---:|
| Train | 400 |
| Validation | 87 |
| Test | 87 |
| Excluded unusable | 1 |

Dominant-label distribution:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| `plastic` | 264 | 57 | 57 |
| `foam` | 46 | 10 | 10 |
| `metal` | 13 | 3 | 3 |
| `other_debris` | 77 | 17 | 17 |

The dataset is small and imbalanced. The `metal` class has the fewest samples.

## Label Policy

- Original dataset annotations are treated as ground truth.
- The official model uses one dominant label per image.
- The experimental multi-label branch uses multi-hot target vectors.
- No manual relabeling was performed for the official version.

## Project Structure

```text
marine-debris-classifier/
|-- README.md
|-- run_instructions.md
|-- requirements.txt
|-- Dockerfile
|-- app.py
|-- app/
|   |-- streamlit_app.py
|   `-- streamlit_dual_model_app.py
|-- assets/
|   `-- sample_images/
|-- data/
|   `-- processed/
|       `-- records/
|           |-- final_train_val_test_split.csv
|           |-- multilabel_records_all_splits.csv
|           `-- multilabel_train_val_records.csv
|-- outputs/
|   |-- models/
|   |   |-- resnet18_finetuned_layer4_best.pth
|   |   `-- multilabel_resnet18_layer4_best.pth
|   |-- reports/   # curated app/evaluation JSON and Markdown only
|   `-- plots/     # confusion matrix plots used by the app
|-- src/
|   |-- config.py
|   |-- transforms.py
|   |-- dataset.py
|   |-- model.py
|   |-- train_utils.py
|   |-- train_baseline.py
|   |-- train_finetune.py
|   |-- evaluate_final_test.py
|   |-- predict.py
|   |-- predict_dual_model.py
|   |-- multilabel_dataset.py
|   |-- multilabel_model.py
|   |-- multilabel_train_utils.py
|   |-- train_multilabel_resnet18.py
|   |-- train_multilabel_resnet18_finetune_layer4.py
|   `-- tune_multilabel_thresholds.py
`-- project_report.md
```

The GitHub repository intentionally excludes raw dataset files, local virtual
environments, temporary uploads, bulky generated report assets, draft report
builders, and regenerated output clutter. Final model weights are tracked with
Git LFS because each checkpoint is larger than GitHub's normal file limit.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Main libraries: Python 3.9+, PyTorch, Torchvision, Pandas, NumPy, Pillow,
Matplotlib, Streamlit, tqdm, python-docx, ReportLab, and huggingface-hub.

## Running the Apps

Combined app for Hugging Face / local demo:

```bash
streamlit run app.py
```

Official single-label app:

```bash
streamlit run app/streamlit_app.py
```

Experimental dual-model app:

```bash
streamlit run app/streamlit_dual_model_app.py
```

## Command-Line Predictions

Single-label prediction:

```bash
python src/predict.py --image "path/to/image.jpg"
```

Optional JSON export:

```bash
python src/predict.py --image "path/to/image.jpg" --output "outputs/reports/single_model_prediction_example.json"
```

Dual-model prediction:

```bash
python src/predict_dual_model.py --image "path/to/image.jpg"
```

The dual-model CLI writes:

```text
outputs/reports/dual_model_prediction_example.json
```

## Reproducing the Pipeline

Official single-label workflow:

```bash
python src/check_dataset_pipeline.py
python src/train_baseline.py
python src/train_finetune.py
python src/evaluate_final_test.py
python src/final_submission_audit.py
```

Experimental multi-label workflow:

```bash
python src/create_multilabel_records.py
python src/check_multilabel_dataset_pipeline.py
python src/train_multilabel_resnet18.py
python src/train_multilabel_resnet18_finetune_layer4.py
python src/tune_multilabel_thresholds.py
```

## Results Interpretation

| Metric | Value | Interpretation |
|---|---:|---|
| Accuracy | 81.61% | Good overall performance |
| Weighted-F1 | 82.24% | High because larger classes dominate |
| Macro-F1 | 69.29% | Lower because minority classes have few examples |

The macro-F1 drop is expected because the dataset is small, imbalanced, and
contains many multi-object or visually ambiguous images.

## Hugging Face Deployment

This project is prepared for a Docker-based Hugging Face Space. The Docker
entry point runs:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

See `huggingface_deployment.md` for deployment details.

## Limitations

- Small dataset size
- Severe class imbalance, especially for `metal`
- Many images contain multiple debris types
- Official model predicts one dominant class per image
- Multi-label branch is experimental and validation-tuned
- Streamlit apps are academic prototypes, not production monitoring systems

## Future Work

- Collect more images, especially for metal and foam
- Add expert label verification
- Train an object detection model for localized debris detection
- Improve multi-label validation with a separate unseen test set
- Add batch prediction and monitoring for deployment scenarios

## Conclusion

This project delivers a complete marine debris classification prototype using
deep learning. The official ResNet18 model achieves 81.61% held-out test
accuracy and includes a reproducible workflow from processed records to
training, evaluation, inference, Streamlit demos, and Docker deployment.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Dataset metadata in the project references Kaggle and Roboflow exports.
- Built with PyTorch and Streamlit.
