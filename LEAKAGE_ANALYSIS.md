# APTOS 2019 duplicate leakage analysis

**Retrospective research record, 26 November 2026.** This is an image-content grouping experiment, not patient-level validation or a clinical device evaluation. The original `splits/` manifests, historical one-epoch checkpoint and historical results remain unchanged.

## Methods and confirmed data-integrity observations

The labelled APTOS 2019 `train.csv` contains **3,662** images. All 3,662 PNG files were SHA-256 hashed. The files form **3,534 distinct content groups**; **123** groups contain two or more files, comprising **251 images**. Across the entire dataset, **30 exact-content groups have conflicting five-grade disease labels** and **4 have conflicting binary no-DR/any-DR labels**. These conflicts require annotation review. Exact group membership, labels, and original and new partition assignments are stored privately in `runs/leakage/splits_confirmed_seed42/exact_group_members_PRIVATE.csv`; the image identifiers and hashes are excluded from Git.

Of the 123 multi-image groups, **64 cross an original partition boundary**: 30 train/test, 29 train/validation and 5 validation/test. The other 59 are within one original partition (57 training, 2 validation). The original test partition contains **35 distinct images** with an exact-content counterpart elsewhere, as documented in `PUBLICATION_EXPERIMENTS.md`. The original 550-image test metrics are preserved as historical evidence, but cross-partition duplicates limit their interpretation.

The prior 64-bit difference-hash screen produced 155,793 broad cross-partition candidates at Hamming distance ≤5. A DCT perceptual hash at distance ≤6 retained 2,791; aligned thumbnail correlation ≥0.99 and grayscale mean absolute difference ≤15 shortlisted 457. One reviewer visually inspected the private 12-pair contact sheet. **Six high-confidence same-image pairs with different bytes** were recorded in `runs/leakage/verified_near_pairs_PRIVATE.csv` and merged into the confirmed grouping. This was a limited unblinded review, not an independent adjudication. The other candidates have not been fully adjudicated. Image similarity does not establish patient identity; different eyes or visits from a patient can also evade these screens.

## New duplicate-controlled split

`src/leakage_splits.py` validates the original manifests against Kaggle labels, merges the exact-content and six visually verified groups, then searches 300 deterministic group-stratified candidate splits seeded from 42. Selection minimizes grade-distribution imbalance against a 70/15/15 target. It uses labels for stratification, never model predictions or test outcomes. The resulting **3,528 confirmed groups** do not cross the new train/validation/test partitions. New manifests are private under `runs/leakage/splits_confirmed_seed42/`; the original manifests were not edited.

| Partition | Images | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
|---|---:|---:|---:|---:|---:|---:|
| Training | 2,563 | 1,264 | 259 | 696 | 136 | 208 |
| Validation | 549 | 270 | 56 | 151 | 28 | 44 |
| Test | 550 | 271 | 55 | 152 | 29 | 43 |

The new test set shares **89 image IDs** with the original test set before excluding any original-training/validation duplicate groups for the common evaluation subset. The new test manifest SHA-256 is `2c600969fe4d3ebcb41edfaf8179fb2a5232dfbdc32ae584b8345a4aabd2b45d`; the original test manifest SHA-256 remains `322bb102697072a244df0b421d5cc814e8dc1d74a5ded2a2b51da89444860a61`. The grouping is only as complete as the verified image pairs and cannot establish patient-level separation.

## Matched training and analysis protocol

`src/leakage_train.py` computes 512-dimensional features once using **torchvision ResNet18 ImageNet1K V1 weights** and the same 224×224 ImageNet-normalized input transform as the historical baseline. The backbone is frozen in evaluation mode. Each new binary linear head is trained from its seed-specific random initialization with Adam, learning rate 0.001, batch size 32, binary cross-entropy with logits, for five epochs. Seeds **11, 23 and 37** are used for both original and controlled splits. The lowest validation-loss epoch is selected. The historical checkpoint is never loaded for this experiment. This shared feature-cache implementation is mathematically equivalent to frozen-backbone training without stochastic augmentation; it is not full-network fine tuning.

The official ImageNet weight file SHA-256 was `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; the Kaggle label CSV SHA-256 was `b2a3479695922e62c83c94a1a9ef669b4b6d68cc26a1d7090ebedecdd7dd79a5`. The local feature-cache SHA-256 and all six new checkpoint hashes are in private run configuration files. The train script enforces identical epochs, batch size, learning rate, preprocessing, optimizer, loss and selection rule for both split conditions.

`src/leakage_analysis.py` evaluates each selected model on its own test partition at the fixed 0.5 threshold, and on images in both test partitions whose confirmed group is absent from both training/validation sets. This common subset reduces test-composition differences but may be small and cannot remove unknown patient overlap. The script reports accuracy, sensitivity, specificity, precision, F1 and ROC AUC with 2,000-resample percentile intervals that resample **confirmed duplicate groups** within binary-class strata. The intervals still cannot account for unknown patient clusters.

For each controlled-split seed, fixed mild/moderate/severe Gaussian blur and brightness/contrast reductions are applied to the same controlled test images through one bilinear 224×224 resizing pipeline. Confusion matrices and paired predictions are stored privately. The existing blur, brightness, contrast and uncertain-score review thresholds are reused without test-set tuning. Review coverage reports flagged fraction, errors and false negatives captured, and residual error on unflagged images.

## Results and execution status

The data-integrity analysis, visual review of six near pairs, duplicate-controlled split and **six matched five-epoch training runs** are completed. All six selected validation-loss minima occurred at epoch 5. The shared feature extraction and all head runs took **354.6 seconds** locally; the per-head `training_seconds` in configuration files excludes the shared feature extraction. Saved model checkpoints, validation histories, configuration files and per-image scores are private in `runs/leakage/matched_train_seed11_23_37/`.

| Split | Seed | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 11 | .9509 | .9462 | .9557 | .9565 | .9514 | .9823 | 259 / 12 / 15 / 264 |
| Controlled | 11 | .9455 | .9176 | .9742 | .9734 | .9446 | .9779 | 264 / 7 / 23 / 256 |
| Original | 23 | .9545 | .9427 | .9668 | .9669 | .9546 | .9827 | 262 / 9 / 16 / 263 |
| Controlled | 23 | .9345 | .8925 | .9779 | .9765 | .9326 | .9764 | 265 / 6 / 30 / 249 |
| Original | 37 | .9455 | .9534 | .9373 | .9399 | .9466 | .9841 | 254 / 17 / 13 / 266 |
| Controlled | 37 | .9455 | .9140 | .9779 | .9770 | .9444 | .9784 | 265 / 6 / 24 / 255 |

Across seeds, own-test mean accuracy was **.9503** on original splits and **.9418** on controlled splits; mean ROC AUC was **.9830** and **.9776**, respectively. These are separate test populations, so the difference is not a leakage effect estimate. The **85 common uncontaminated images** (85 confirmed groups) gave mean accuracy **.9333** for original-trained versus **.9255** for controlled-trained models, and mean AUC **.9740** versus **.9723**. `paired_common_differences.csv` contains per-seed paired, group-resampled 95% intervals for controlled minus original. The small subset and three seeds do not support a definitive superiority claim.

For illustration, group-bootstrap 95% accuracy intervals were **.9309–.9691** for original seed 11 and **.9252–.9636** for controlled seed 11. The other seed-specific intervals for all six metrics are in `leakage_results/model_intervals.csv`. These are conditional on the chosen split and confirmed grouping, not population-level patient intervals. Twelve fresh image predictions from the saved full controlled seed-11 checkpoint matched the cached-head scores within **1.90×10⁻⁶**; all 550 original-condition cached features matched stored scores within **2.06×10⁻⁶**.

### Duplicate-controlled test: fixed image degradations

The following entries are **means across three seeds** on the same 550 controlled-test images. Per-seed confusion matrices and per-image paired scores are private; per-seed aggregate values are in `leakage_results/controlled_degradation_metrics.csv`.

| Condition | Accuracy | Sensitivity | Specificity | ROC AUC |
|---|---:|---:|---:|---:|
| Original | .9418 | .9080 | .9766 | .9776 |
| Mild Gaussian blur, 3×3 σ1 | .9303 | .8937 | .9680 | .9760 |
| Moderate Gaussian blur, 9×9 σ3 | .9030 | .8483 | .9594 | .9682 |
| Severe Gaussian blur, 21×21 σ7 | .8776 | .8805 | .8745 | .9502 |
| Brightness ×0.65 | .9406 | .9140 | .9680 | .9787 |
| Contrast ×0.65 | .9455 | .9200 | .9717 | .9773 |

All changes are synthetic. Apparent point-estimate gains under brightness or contrast changes are not validated improvements and were not used to tune a model. Severe blur caused the largest mean accuracy loss; its effect on sensitivity and specificity varied by seed.

### Fixed automated review rules

The unchanged rules flagged **37.45%–37.64%** of the controlled-test images across seeds (mean **37.58%**). They captured **43.33%–50.00%** of classification errors and **43.48%–53.33%** of false negatives. Residual error among unflagged images was **4.94%–5.25%** (mean **5.05%**). These results are in `leakage_results/controlled_review_coverage.csv`. Since many errors and false negatives remain unflagged, the flags cannot be called a reliable automated rejection mechanism.

## Reproduction and output inventory

Use the authorised Kaggle `train.csv` and `train_images/` files locally or in a private Colab runtime. Run the full hash screen in `src.publication_experiments` first as documented in `PUBLICATION_EXPERIMENTS.md`; its `image_hashes_PRIVATE.csv` supplies the content grouping. Manual review decisions must be recorded in a private CSV with `id_a,id_b,review_basis`. The six reviewed decisions used here remain in `runs/leakage/verified_near_pairs_PRIVATE.csv`. From the repository root, the subsequent commands were:

```bash
python -m src.leakage_splits --labels data/aptos/train.csv \
  --hashes runs/publication/20261126T160952168783Z/image_hashes_PRIVATE.csv \
  --verified-near-pairs runs/leakage/verified_near_pairs_PRIVATE.csv \
  --original-splits splits --output runs/leakage/splits_confirmed_seed42 \
  --seed 42 --trials 300
python -m src.leakage_train --labels data/aptos/train.csv \
  --images data/aptos/train_images --original-splits splits \
  --controlled-splits runs/leakage/splits_confirmed_seed42 \
  --output runs/leakage/matched_train_seed11_23_37 \
  --epochs 5 --batch-size 32 --lr 0.001 --seeds 11 23 37
python -m src.leakage_analysis --original-splits splits \
  --controlled-splits runs/leakage/splits_confirmed_seed42 \
  --groups runs/leakage/splits_confirmed_seed42/group_keys_PRIVATE.csv \
  --training runs/leakage/matched_train_seed11_23_37 \
  --images data/aptos/train_images --output runs/leakage/final_analysis \
  --seeds 11 23 37 --bootstrap 2000
python -m src.leakage_paired --original-splits splits \
  --controlled-splits runs/leakage/splits_confirmed_seed42 \
  --groups runs/leakage/splits_confirmed_seed42/group_keys_PRIVATE.csv \
  --training runs/leakage/matched_train_seed11_23_37 \
  --output runs/leakage/final_analysis --seeds 11 23 37 --bootstrap 2000
```

Use a new output directory for a new run; the split generator refuses to overwrite an existing nonempty directory. The private directory contains all group membership, score, checkpoint, cache and paired prediction evidence. `leakage_results/` contains only checked aggregate CSV tables and figures. Do not copy the private directory into a public repository or manuscript supplement without a separate dataset and privacy review.

## Interpretation and manuscript limits

The original and duplicate-controlled models have different training and test compositions; differences in their own-test metrics cannot isolate a pure causal effect of leakage. The common uncontaminated subset is a useful paired sensitivity analysis, not an independent external validation. Exact and six verified near groups are controlled, but other unreviewed near duplicates and patient overlap may remain. A single dataset, image-level labels, uncertain annotation quality, synthetic degradations and heuristic quality thresholds preclude clinical or diagnostic claims. Original test scores must not guide model selection, thresholds or the controlled split.
