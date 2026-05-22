# Marine Debris Classifier Project Report

## Project Overview

This project builds a deep learning prototype for classifying underwater marine
debris images into four target classes:

- `plastic`
- `foam`
- `metal`
- `other_debris`

The official model is a single-label classifier that predicts the dominant
debris class in an image. The project also includes an experimental dual-model
extension that combines the official single-label model with a multi-label model
for possible additional debris types.

## Dataset

The processed dataset contains 575 records. One record is excluded as unusable,
leaving 574 usable images.

Final usable split:

| Split | Images |
|---|---:|
| Train | 400 |
| Validation | 87 |
| Test | 87 |

Dominant-label distribution:

| Class | Total |
|---|---:|
| plastic | 378 |
| foam | 66 |
| metal | 19 |
| other_debris | 111 |

The dataset is small and imbalanced. The metal class is especially limited, and
many images contain multiple debris types.

## Label Policy

The official model uses a dominant-label policy. When an image contains multiple
mapped debris labels, one selected label is used for the single-label task. The
multi-label branch uses the mapped label set to create multi-hot target vectors.

## Model Architecture

The official model uses ResNet18 transfer learning:

- Image size: 224 x 224
- ImageNet normalization
- Final classifier replaced with a four-class linear head
- Baseline training with frozen backbone
- Fine-tuning with layer4 and the final classifier unfrozen

The experimental multi-label model also uses ResNet18, but its final layer
outputs four independent logits trained with `BCEWithLogitsLoss`.

## Official Single-Label Results

Official held-out test results for
`outputs/models/resnet18_finetuned_layer4_best.pth`:

| Metric | Score |
|---|---:|
| Accuracy | 81.61% |
| Macro-F1 | 69.29% |
| Weighted-F1 | 82.24% |

Per-class F1:

| Class | F1 | Support |
|---|---:|---:|
| plastic | 0.8829 | 57 |
| foam | 0.6667 | 10 |
| metal | 0.4444 | 3 |
| other_debris | 0.7778 | 17 |

The official confusion matrix is available at:

```text
outputs/plots/final_test_confusion_matrix.png
```

## Experimental Dual-Model Extension

The dual-model app combines:

- Single-label model: dominant class prediction
- Multi-label model: possible visible debris classes

The multi-label branch uses tuned validation thresholds:

| Class | Threshold |
|---|---:|
| plastic | 0.25 |
| foam | 0.25 |
| metal | 0.75 |
| other_debris | 0.15 |

The multi-label confusion matrices shown in the app are validation
threshold-tuning matrices, not official held-out test results.

## Applications

The project includes three Streamlit entry points:

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | Official single-label app |
| `app/streamlit_dual_model_app.py` | Experimental dual/multi-label app |
| `app.py` | Combined Hugging Face Space app |

## Hugging Face Deployment

The project is prepared for a Docker-based Hugging Face Space. The Space entry
point is `app.py`, which exposes both the single-label and dual-model workflows.
Deployment details are documented in:

```text
huggingface_deployment.md
```

## Limitations

- Small dataset size
- Severe class imbalance, especially for metal
- Many images contain multiple debris types
- The official single-label task reduces multi-object scenes to one dominant
  label
- The multi-label branch is experimental and validation-tuned
- This prototype is not suitable for operational marine pollution monitoring

## Conclusion

The project delivers a complete prototype pipeline from audited records to
training, evaluation, inference, Streamlit apps, and Hugging Face deployment
files. The official single-label model achieves strong overall accuracy for a
small dataset, but macro-F1 and minority-class performance show clear room for
future improvement.
