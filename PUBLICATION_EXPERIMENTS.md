# Retrospective feasibility experiments — execution record

**26 November 2026. Research software, not a clinical device.** This record separates completed analyses from optional training that has not run. The historical one-epoch checkpoint, test partition and result in `RESEARCH_REPORT.md` are unchanged. New analyses from the preserved checkpoint are identified below.

## Current execution status

| Priority | Experiment | Status on 26 November 2026 | Evidence |
|---|---|---|---|
| 1 | Split integrity | **Completed for the three saved manifests.** Unique image IDs, valid grade/binary labels, disjoint partitions and 3,662-row union checked. | Private `runs/publication/20261126T111513059140Z/manifest_audit.csv` and `status.json` |
| 1 | Image exact/perceptual duplicates | **Exact-file screen completed.** Perceptual candidates screened with two hashes; human review remains. | Private `runs/publication/20261126T160952168783Z/exact_cross_partition.csv` and `perceptual_candidates_refined.csv` |
| 2 | Baseline checkpoint | **Fresh image inference completed; historical summary reproduced.** | Private `condition_metrics.csv`, `historical_comparison.csv`, `score_reproduction_summary.csv` in the same run |
| 3 | Bootstrap and grade results | **Completed from saved scores and fresh inference; results agree.** | Private `bootstrap_intervals.csv`, `grade_sensitivity.csv` |
| 4 | Paired robustness | **Completed on all 550 original test IDs.** | Private `paired_predictions_PRIVATE.csv`, `condition_metrics.csv`, `confusion_matrices.csv` |
| 5 | Review flag coverage | **Completed using fixed existing thresholds on original test images.** | Private `review_coverage.csv` |
| 6 | Three-seed retraining | **Optional, not run on retinal images.** | `src/train.py`, optional Colab cell |

The private Drive backup has the notebook, checkpoint, split manifests and historical per-image test predictions, but no APTOS `train_images` directory or archive. The original dataset was downloaded into a Colab runtime using Kaggle API access. On 26 November, the authorised Kaggle API also downloaded the original archive privately to the ignored local `data/aptos` directory. The historical test predictions were retrieved into the ignored `private_archive/results/test_predictions_private.csv`. The local 50-image random-noise run in `work/smoke/publication/` only tested software execution; its metrics are not study results.

The saved manifest audit found train **2,563** (grade counts 1263/259/699/135/207), validation **549** (271/55/150/29/44), and test **550** (271/56/150/29/44). The saved test manifest SHA-256 is `322bb102697072a244df0b421d5cc814e8dc1d74a5ded2a2b51da89444860a61`. The saved manifests match Kaggle `train.csv` by image ID and grade. The partitions remain unchanged.

### Material data-integrity finding

SHA-256 found **66 cross-partition pairs** involving **64 distinct exact-file groups**. **35 distinct test images** have a byte-identical image in training or validation. The affected test images are all DR-positive (grades 1/2/3/4: 7/21/2/5). Among the cross-partition exact-file groups, **14 have conflicting severity grades** and **one has conflicting binary labels** (a train grade-1 image and byte-identical validation grade-0 image). These are direct evidence of duplicate leakage and at least one annotation conflict. Unique image IDs alone failed to detect them.

The saved predictions classify 32/35 affected test images as DR and miss 3. A **post hoc descriptive** calculation on the other 515 test images gives 485 correct, 94.17% accuracy, 17 false negatives, 93.03% DR sensitivity and 0.9764 AUC. This is **not** a repaired independent test estimate: the model had already been trained using the original split; near duplicates, patient overlap and other biases remain unresolved. The original 550-image result remains intact.

The broad 64-bit difference-hash screen generated 155,793 cross-partition candidate pairs at Hamming distance ≤5, showing inadequate specificity for these retinal images. A second DCT-based perceptual hash at distance ≤6 retained **2,791 candidate pairs**. A stringent aligned grayscale thumbnail screen (correlation ≥0.99 and mean absolute intensity difference ≤15) shortlisted **457 pairs**. A private 12-pair contact sheet was inspected: the highest-ranked pairs include visually near-identical retinal photographs with different bytes, but the full shortlist has not been adjudicated. Hash and thumbnail matches are candidates, not established patient identity. No claim that near duplicates have been exhaustively resolved is warranted.

### Reanalysis of saved test scores

The private 550-row prediction table was checked against every saved test ID and grade. Its scores recreate the historical TN/FP/FN/TP = **258/13/20/259**, accuracy **0.9400**, sensitivity **0.9283**, specificity **0.9520**, precision **0.9522**, F1 **0.9401**, and AUC **0.9754**. This confirms consistency of the saved predictions with the historical summary; it is **not** a fresh checkpoint inference run. The prediction file SHA-256 is `e0ad669184a1c105b1fc2403c48c89a86af07b8bb37c8589876176acf8878c4c`.

Fresh CPU inference from the preserved checkpoint on the original Kaggle images independently reproduced the **same 550-image confusion matrix and summary metrics**. All 550 class decisions matched the saved prediction table. Mean absolute score difference was **5.00×10⁻⁷**, maximum **3.28×10⁻⁶**, consistent with minor runtime numerical variation. The checkpoint SHA-256 is `b968f81ebe724b45cf55081200e276ec7aa49606bc2faaa873e241e896d561c2`. This reproduces inference, not the original training trajectory.

Two thousand stratified image bootstrap resamples (seed 20261126) give the following 95% percentile intervals. They are exploratory image-level intervals because patient clustering is unknown.

| Metric | Point | Lower 95% | Upper 95% |
|---|---:|---:|---:|
| Accuracy | 0.9400 | 0.9200 | 0.9582 |
| Sensitivity | 0.9283 | 0.8961 | 0.9570 |
| Specificity | 0.9520 | 0.9262 | 0.9742 |
| Precision | 0.9522 | 0.9276 | 0.9740 |
| F1 | 0.9401 | 0.9201 | 0.9586 |
| ROC AUC | 0.9754 | 0.9633 | 0.9862 |

Test grade-specific positive sensitivity from the same saved scores is grade 1 **49/56 (0.8750; 7 false negatives)**, grade 2 **140/150 (0.9333; 10 false negatives)**, grade 3 **28/29 (0.9655; 1 false negative)**, and grade 4 **42/44 (0.9545; 2 false negatives)**. These are descriptive counts on small subgroups, not separately validated endpoints.

### Matched degradation results on the original test IDs

All six conditions used the same 550 images and the same PIL bilinear 224×224 resizing before any synthetic degradation. The table below gives fresh checkpoint inference at threshold 0.5. Each row's full confusion matrix and paired per-image scores are saved privately.

| Condition | Accuracy | Sensitivity | Specificity | ROC AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|
| Original | 0.9400 | 0.9283 | 0.9520 | 0.9754 | 258 / 13 / 20 / 259 |
| Mild blur, 3×3, σ1 | 0.9273 | 0.9104 | 0.9446 | 0.9731 | 256 / 15 / 25 / 254 |
| Moderate blur, 9×9, σ3 | 0.9127 | 0.9319 | 0.8930 | 0.9598 | 242 / 29 / 19 / 260 |
| Severe blur, 21×21, σ7 | 0.8618 | 0.9498 | 0.7712 | 0.9472 | 209 / 62 / 14 / 265 |
| Brightness ×0.65 | 0.9291 | 0.9391 | 0.9188 | 0.9765 | 249 / 22 / 17 / 262 |
| Contrast ×0.65 | 0.9236 | 0.9391 | 0.9077 | 0.9748 | 246 / 25 / 17 / 262 |

Severe blur reduced accuracy and specificity substantially, while sensitivity increased as many more negatives were labelled DR. The small AUC rise under reduced brightness is an observed point estimate, not evidence of improvement; no paired uncertainty test was performed. These are fixed synthetic transformations of one dataset, not a camera deployment study.

### Fixed automated review rules

On the **original** 550 test images, the unchanged heuristic rules flagged **237/550 (43.09%)**. They captured **21/33 classification errors (63.64%)** and **12/20 false negatives (60.00%)**. Among **313 unflagged** images, **12 errors remained (3.83%)**, including **8 false negatives**. Thus the rules do not reliably reject all problematic images or all missed DR cases. These percentages measure the existing rules; no threshold was tuned on test data.

## Run in Google Colab

Open `notebooks/publication_colab.ipynb`. It mounts the private `Smart-Retinal-Camera-AI/research_release_20261121T123249Z` Drive backup, clones the public repository, prompts for the Kaggle API token without displaying it, downloads APTOS through the competition API, copies the preserved manifests and checkpoint to the temporary runtime, and runs experiments 1–5. Results go to the private Drive release under `publication_runs/<UTC timestamp>/`. The token file is deleted after extraction. Do not put a token in the notebook or repository.

Equivalent local command after obtaining the authorised images and original labels:

```bash
python -m src.publication_experiments \
  --splits splits --labels data/aptos/train.csv \
  --images data/aptos/train_images \
  --checkpoint models/resnet18_epoch1.pth \
  --output runs/publication --bootstrap 2000
```

`--images` omitted runs the manifest audit. Adding `--saved-predictions private_archive/results/test_predictions_private.csv` also performs the documented historical-score reanalysis without photographs. Private per-image files, duplicate candidates, checkpoint and dataset remain excluded from Git. Do not copy the whole run directory into a public release.

## Prespecified analysis methods

**Duplicate screen.** SHA-256 checks byte-identical photographs across partitions. A 64-bit difference hash on 9×8 grayscale resized images screens cross-partition candidates at Hamming distance ≤5. A 64-bit DCT perceptual hash at distance ≤6 refines them, and aligned grayscale thumbnail correlation ≥0.99 with mean absolute difference ≤15 ranks a shortlist. A hash candidate needs manual image review and does not prove patient identity; a negative screen does not prove no duplicate or patient leakage. The source data has no established patient IDs. The code never reassigns a test image.

**Baseline comparison.** The locked checkpoint is evaluated on the saved test IDs at sigmoid threshold 0.5. `historical_comparison.csv` compares accuracy, sensitivity, specificity, F1, AUC and TN/FP/FN/TP with `results/test_metrics.csv`. Counts must match exactly; floating metrics use tolerance 0.0005 to allow historical rounding. `status.json` reports `historical_metrics_all_match`; investigate any false value before calling the run reproduced. `paired_predictions_PRIVATE.csv` stores IDs, grades, labels, scores, predictions and quality indicators.

**Uncertainty.** For the original condition, 2,000 percentile bootstrap resamples sample the No DR and DR test images separately with replacement. The reported 2.5th and 97.5th percentiles cover accuracy, sensitivity, specificity, precision, F1 and ROC AUC. The seed is 20261126. These are **image-level** intervals. They cannot account for within-patient correlation if both eyes or repeated acquisitions occur. Grade-specific positive sensitivity and false-negative counts are descriptive, with denominators saved.

**Matched degradations.** Every condition begins with native RGB → one PIL bilinear resize to 224×224 → degradation → tensor conversion and ImageNet normalization. Baseline has no degradation. Gaussian blur is OpenCV 3×3/σ1, 9×9/σ3, and 21×21/σ7. Brightness multiplies 8-bit RGB intensities by 0.65 with rounding/clipping. Contrast moves RGB channels toward each image's per-channel spatial mean by factor 0.65, with rounding/clipping. No randomness or test-set parameter optimisation is used. All six conditions use the identical test IDs and order. `confusion_matrices.csv` gives every 2×2 matrix; `paired_predictions_PRIVATE.csv` enables paired analyses. These synthetic changes are not validated simulations of real camera faults.

**Automated review.** Fixed prior rules: native-original Laplacian variance <10; grayscale mean <40 or >100; grayscale standard deviation <25; or model score in [0.4, 0.6]. On the original test images, `review_coverage.csv` reports fraction flagged, errors captured, false negatives captured, unflagged denominator, residual errors and residual false negatives. The quality score is measured at original resolution for the baseline. The thresholds were not fitted on test data, but they also have not been validated against expert gradability labels. A flag does not reject an image or withhold a prediction.

## Output inventory after a full image run

All numerical study tables are CSV: `manifest_audit.csv`, `image_hashes_PRIVATE.csv`, `exact_cross_partition.csv`, `perceptual_candidates.csv`, `perceptual_candidates_refined.csv`, `perceptual_candidates_ranked_PRIVATE.csv`, `perceptual_shortlist_PRIVATE.csv`, `paired_predictions_PRIVATE.csv`, `condition_metrics.csv`, `confusion_matrices.csv`, `bootstrap_intervals.csv`, `grade_sensitivity.csv`, `review_coverage.csv`, `exact_duplicate_sensitivity.csv`, `historical_comparison.csv`, `score_reproduction_PRIVATE.csv` and `score_reproduction_summary.csv`. The actual local run names its post hoc exact-duplicate table `historical_exact_duplicate_sensitivity.csv`; both represent the same code-defined calculation. The saved-score-only run additionally writes `historical_scores_metrics.csv`, `historical_scores_confusion.csv` and `historical_scores_comparison.csv`. `status.json` records methods, hashes, software versions and run completion. The 300-dpi PNG figures are `baseline_intervals.png`, `robustness.png` and `grade_sensitivity.png`. Inspect the figures and all candidate duplicates before manuscript use.

## Optional repeated training

The final Colab cell is disabled by default. Enabling it runs frozen-backbone ResNet18 for five epochs each with seeds **11, 23 and 37**, using the original training/validation partitions. `src/train.py` writes configuration, validation history, epoch seconds, total training seconds and validation-selected `best.pth` into unique private run directories. These would be new results. Do not attribute the historical 94% test result to them; select using validation data before any held-out test evaluation.

## Interpretation limits

This remains a single-dataset, image-level retrospective feasibility analysis. Patient-level leakage remains unresolved; the perceptual candidate shortlist still requires full manual adjudication. Binary any-DR evaluation may conceal severity differences. Confidence intervals and degradation experiments cannot establish clinical safety, real acquisition robustness, suitable image rejection, calibrated probabilities or diagnostic suitability. The original checkpoint was trained for one completed epoch with unseeded head initialisation and shuffle, so exact retraining is not expected.
