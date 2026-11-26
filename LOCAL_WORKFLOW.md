# Full notebook, local training and Streamlit

## What is preserved

`private_archive/Smart_Retinal_Camera_AI_documented.ipynb` is the full downloaded notebook, unchanged, with original figures and outputs. Open it in VS Code using the Python/Jupyter extensions. It is private and Git-ignored. `notebooks/full_research_code.ipynb` has exactly the same source cells with outputs, attachments and metadata removed; it can be shared. Historical hard-coded `/content` paths, Google Drive mounting, setup diagnostics and the single-epoch training cells remain in both notebooks. Use the modules below for portable new experiments.

The original model is `models/resnet18_epoch1.pth`; the preserved split CSVs are `splits/train.csv`, `splits/val.csv`, `splits/test.csv`. These are local-only assets. They were retrieved from the existing Drive release. They are not available to someone who only clones GitHub.

## Environment

Use Python 3.12 and an isolated virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
python -m streamlit run app.py
```

`requirements-local.txt` records the dependencies used for this local implementation. The historical `requirements.txt` and `environment.json` retain the original Colab environment; they have not been replaced. For a CUDA training machine choose the appropriate official PyTorch build. CPU is the default fallback; MPS can be explicitly selected for training, but has not been validated for this project. Seeded runs improve reproducibility, but hardware/library differences and operations reported by PyTorch as nondeterministic can still change results.

## Supply the dataset

Acquire APTOS 2019 through your authorised Kaggle account. Place its labelled `train.csv` at `data/aptos/train.csv` and photographs at `data/aptos/train_images/<id_code>.png`. No dataset images or credentials are committed. The data loader validates labels, image ID uniqueness, split disjointness, manifest coverage and image existence before training. Exact duplicate/near-duplicate or patient-level leakage checking is still outstanding.

## Train a new model

```bash
python -m src.train \
  --csv data/aptos/train.csv \
  --images data/aptos/train_images \
  --splits splits \
  --epochs 5 --batch-size 32 --lr 0.001 --seed 42
```

This starts with ImageNet ResNet18 weights and a new binary head. The frozen backbone stays in evaluation mode so its BatchNorm statistics are unchanged. The first use may download official ImageNet weights. Each invocation creates a unique `runs/<UTC timestamp>/` directory and never replaces the original model or historical results. It saves `config.json`, `history.csv`, split manifests and `best.pth`, selected only by validation loss. The test images are not evaluated during training.

Optional changes for explicitly new experiments:

- `--fine-tune --lr 0.00001` trains the backbone as well as the head; this differs from the historical frozen-feature experiment.
- `--initial-checkpoint models/resnet18_epoch1.pth` explicitly warm-starts from the old baseline with a fresh optimiser; this is not exact continuation of the original optimiser state.
- Omit `--splits` to generate a stratified 70/15/15 image-level split using `--seed`. Prefer the preserved manifests when comparing with the original run.
- `--random-init` exists for software smoke tests. It is not the pretrained research baseline and must not be reported as such.

## Evaluate after model selection

Replace the run timestamp with your completed run:

```bash
python -m src.evaluate \
  --checkpoint runs/<timestamp>/best.pth \
  --manifest splits/test.csv \
  --images data/aptos/train_images
```

This writes a new evaluation directory with metrics, checkpoint identity and per-image predictions. Keep these private. Do not repeatedly use held-out results to select models or thresholds. A fresh model's metrics are new results; never attribute the historical 94% test accuracy to it without evaluation.

## Demonstrate in Streamlit

Start `python -m streamlit run app.py`, then visit http://localhost:8501. Choose a local checkpoint, upload an authorised image and optionally enable Grad-CAM. The app processes images in memory and does not save uploads. The JSON download contains the score, flags and checkpoint hash. It does not contain the image. No patient identifiers should be placed in demonstration uploads.

The original notebook classified native RGB images using torchvision's 224-pixel resize and ImageNet normalisation. The app uses the same preprocessing and native-resolution quality measurements. Predictions use a fixed score threshold of 0.5. Quality thresholds and the 0.4–0.6 review interval remain heuristic. A missing flag does not mean the photograph is adequate or that disease is absent. Grad-CAM explains the positive DR logit even when the predicted class is No DR; it is not clinical lesion localisation.

The server binds to localhost and usage telemetry is disabled. This is a local research demo, not a deployed clinical service. Remote deployment would require separate privacy, access and operational decisions.

## Software checks versus research results

Run `python -m pytest -q`. Tests cover metric counts, review-rule boundaries, overlapping manifests, checkpoint loading/Grad-CAM cleanup, notebook source preservation and Streamlit startup. Small synthetic images may be used to test the training/evaluation plumbing. Their metrics are software-test outputs only. No new full-dataset retinal training has been completed by adding this workflow.

API references: [Streamlit uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest), [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
