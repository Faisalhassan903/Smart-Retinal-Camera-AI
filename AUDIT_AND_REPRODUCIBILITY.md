# Audit and execution guide

Preserved original cells 0–38 without modifying their source, outputs or execution counts. The documented copy adds headings and explanatory markdown. Cells 39–40 were pre-existing placeholders; cell 40 now contains the snapshot operation. The snapshot was taken before packaging.

Resolved in this release: missing standalone exports, model/split backup, missing documentation, ambiguous sections, absent environment inventory, absent saved test predictions, and incomplete demonstration packaging. Added metric and split checks, a portable inference entry point and a file-integrity manifest.

Outstanding: full clean-runtime execution, fresh deterministic retraining, confidence intervals, patient/duplicate audit, external validation, calibrated scores, validated quality thresholds, explanation validation, camera hardware integration and dataset redistribution review.

To replicate inference: obtain APTOS through your authorised Kaggle account and comply with its terms. Place images in /content/aptos/train_images. Use preserved splits and checkpoint; do not recreate the test split to optimise results. Run notebook definitions in cells 0, 10, 13 and 14, then restore the saved checkpoint; evaluate with cells 33–34, or run src/demo.py for a single image. Cell 14 initialises a fresh head; do not mistake it for a trained model. Run cells 30–32 only after loading the checkpoint for Grad-CAM. Training cells 19–20 are not needed for inference and can overwrite the checkpoint if rerun. Historical cells 15–18 diagnose an earlier incomplete state and are not the final experiment.

For a new training experiment: make a separate output directory; seed Python, NumPy and PyTorch before model/head construction and loader creation; log deterministic settings and library versions; preserve the existing checkpoint. The original result is not guaranteed to recur. Colab credentials must be entered using getpass/Secrets, never embedded in source. Do not commit .kaggle, tokens, .env, datasets, private prediction manifests or retinal panels.
