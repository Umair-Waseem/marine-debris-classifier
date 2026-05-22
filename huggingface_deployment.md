# Hugging Face Deployment Guide

This project is prepared for a Docker-based Hugging Face Space. The root
`app.py` exposes both applications:

- Single-label marine debris classifier
- Dual / multi-label marine debris classifier

## Why Docker

Hugging Face's current guidance for new Streamlit Spaces is to use a Docker
Space. The included `Dockerfile` runs Streamlit on port `7860`, which matches
the Space configuration in the README metadata.

## Files Required In The Space

Upload or push these paths to the Hugging Face Space repository:

- `app.py`
- `Dockerfile`
- `.dockerignore`
- `.gitignore`
- `.streamlit/config.toml`
- `requirements.txt`
- `src/`
- `outputs/models/resnet18_finetuned_layer4_best.pth`
- `outputs/models/multilabel_resnet18_layer4_best.pth`
- `outputs/reports/`
- `outputs/plots/`
- `README.md`
- `LICENSE`

The model files are large, so store them with Git LFS when using Git.

## Create The Space

1. Open https://huggingface.co/new-space.
2. Choose **Docker** as the SDK.
3. Name the Space, for example `marine-debris-classifier`.
4. Keep it public or private according to your submission needs.
5. Upload/push the required files above.

## Expected Runtime

The Space runs:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

## Notes

- The single-label model's confusion matrix is the official held-out test
  result.
- The multi-label confusion matrices are validation threshold-tuning results,
  not official held-out test results.
- Raw training images are not required for inference deployment.
