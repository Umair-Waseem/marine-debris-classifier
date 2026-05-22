---
title: Marine Debris Classifier
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
---

# Marine Debris Classifier Using Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project that classifies marine debris images into four categories: `plastic`, `foam`, `metal`, and `other_debris`.

The project uses ResNet18 transfer learning and includes data preprocessing, model training, fine-tuning, evaluation, prediction scripts, Streamlit apps, Docker deployment, and an experimental multi-label extension.

---

## Overview

Marine debris harms coastal and underwater environments. This project uses computer vision to identify the main type of debris shown in an image. It can support future work in environmental monitoring, cleanup planning, and automated waste detection.

The official model is a **single-label multi-class classifier**. It predicts one main debris class for each image. An experimental dual-model version is also included to predict additional debris types that may be visible in the same image.

---

## Key Features

* Complete image classification pipeline for marine debris
* Dataset checking, cleaning, preprocessing, and splitting
* ResNet18 baseline training and layer4 fine-tuning
* Final evaluation on a held-out test set
* Streamlit apps for interactive image prediction
* Command-line scripts for local prediction
* Docker-ready setup for Hugging Face Spaces
* Experimental multi-label and dual-model prediction workflow

---

## Target Classes

| Class          | Meaning                                             |
| -------------- | --------------------------------------------------- |
| `plastic`      | Plastic bags, bottles, packaging, and related waste |
| `foam`         | Polystyrene foam and foam-like debris               |
| `metal`        | Cans, metallic objects, and other metal debris      |
| `other_debris` | Glass, rubber, mixed waste, or unidentified debris  |

---

## Official Model

The official model is a ResNet18-based classifier fine-tuned to predict one of the four marine debris classes.

```text
outputs/models/resnet18_finetuned_layer4_best.pth
```

### Final Test Results

| Metric      |  Score |
| ----------- | -----: |
| Accuracy    | 81.61% |
| Macro-F1    | 69.29% |
| Weighted-F1 | 82.24% |

### Multiclass Model Details

The official multiclass model was evaluated on the held-out test set of **87 images**. It correctly classified **71 out of 87** images, giving an overall test accuracy of **81.61%**.

| Class          | Correct / Total | Class Accuracy |
| -------------- | --------------: | -------------: |
| `plastic`      |         49 / 57 |         85.96% |
| `foam`         |          6 / 10 |         60.00% |
| `metal`        |           2 / 3 |         66.67% |
| `other_debris` |         14 / 17 |         82.35% |

The model performs best on classes with more training examples, especially `plastic`. The `metal` class has very few samples, so its class-wise performance is less stable.

Run the final evaluation with:

```bash
python src/evaluate_final_test.py
```

---

## Experimental Multi-Label Extension

The experimental workflow uses two models together.

| Model              | Role                                                      |
| ------------------ | --------------------------------------------------------- |
| Single-label model | Predicts the main debris class                            |
| Multi-label model  | Predicts other possible debris types visible in the image |

```text
outputs/models/multilabel_resnet18_layer4_best.pth
outputs/reports/multilabel_layer4_tuned_thresholds.json
```

### Multi-Label Thresholds

| Class          | Threshold |
| -------------- | --------: |
| `plastic`      |      0.25 |
| `foam`         |      0.25 |
| `metal`        |      0.75 |
| `other_debris` |      0.15 |

This extension is experimental. It does not replace the official single-label model or its final test results.

---

## Dataset Summary

The processed dataset contains **575 records**. One unusable record was removed, leaving **574 usable images**.

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

The dataset is small and imbalanced. The `metal` class has the fewest images, so it is harder for the model to learn this class reliably.

---

## Labeling Policy

* Original dataset annotations were used as ground truth.
* The official model uses one main label for each image.
* The experimental multi-label branch supports multiple labels when more than one debris type may be visible.
* No manual relabeling was done for the official version.

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

Raw datasets, virtual environments, temporary files, and large regenerated outputs are not included in the repository. Final model weights are tracked with Git LFS because they are larger than GitHub’s standard file size limit.

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows PowerShell
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install the required packages:

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
# Predict the main debris class
python src/predict.py --image "path/to/image.jpg"

# Save the prediction as a JSON file
python src/predict.py --image "path/to/image.jpg" --output "outputs/reports/single_model_prediction_example.json"

# Run dual-model prediction
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

The official model achieved **81.61% accuracy** and **82.24% Weighted-F1** on the held-out test set. The Macro-F1 score is lower at **69.29%** because the dataset is imbalanced and some classes, especially `metal`, have very few examples.

The multiclass evaluation shows that the model correctly predicted **49/57 plastic**, **6/10 foam**, **2/3 metal**, and **14/17 other_debris** test images. These results show that the model performs strongly overall, while minority classes remain more challenging because of limited training data.

---

## Hugging Face Deployment

This project is ready for Docker-based deployment on Hugging Face Spaces.

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

Deployment details are available in:

```text
huggingface_deployment.md
```

---

## Limitations

* The dataset is small and imbalanced.
* The `metal` class has very few samples.
* Some images contain more than one debris type.
* The official model predicts only one main class per image.
* The multi-label branch is experimental and tuned on validation data.
* The Streamlit apps are academic prototypes, not production systems.

---

## Future Work

* Collect more images for underrepresented classes
* Add expert label verification
* Train an object detection model to locate debris in images
* Evaluate the multi-label model on a separate unseen test set
* Add batch prediction, logging, and model explainability

---

## Conclusion

This project developed a complete marine debris classification system using deep learning. A fine-tuned ResNet18 model classified images into four classes: `plastic`, `foam`, `metal`, and `other_debris`.

The official single-label model achieved **81.61% test accuracy**, **69.29% Macro-F1**, and **82.24% Weighted-F1** on the held-out test set. These results are strong considering the small dataset, class imbalance, and difficult underwater image conditions.

An experimental multi-label model was also developed for images with more than one debris type. After validation-based threshold tuning, it achieved **70.11% subset accuracy**, **91.67% Hamming accuracy**, **94.09% Micro-F1**, **93.15% Macro-F1**, and **94.11% Weighted-F1** on the validation set.

Overall, the project demonstrates a full machine learning workflow, including dataset preparation, preprocessing, model training, evaluation, deployment, and experimental improvement.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

* Dataset metadata references Kaggle and Roboflow exports.
* Built using PyTorch and Streamlit.
