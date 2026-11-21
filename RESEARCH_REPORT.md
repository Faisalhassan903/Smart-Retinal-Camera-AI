# Smart Retinal Camera AI: research report

Prepared for a University of Exeter PhD application, 21 November 2026. This is a retrospective, single-dataset research prototype. It is not clinically validated and is not suitable for diagnosis. The name describes an intended camera-support workflow; no physical camera integration, prospective study or deployment performance was demonstrated.

## Research question and contribution
Can a lightweight transfer-learning baseline combine retinal-image classification with image-quality review indicators, and what do controlled degradations reveal about its behaviour? The contribution is the integration and critical evaluation of classification, quality measurements and explanatory visualisations, rather than a claim of a novel network architecture or clinical benefit.

## Data and experimental design
The APTOS 2019 labelled set contains 3,662 photographs. Grades 0–4 contain 1,805, 370, 999, 193 and 295 images respectively. The binary target is grade 0 versus any grade greater than 0: 1,805 No-DR and 1,857 DR images. This is an any-DR label, not a validated referable-DR endpoint.

Two stratified splits using diagnosis grade and random_state=42 produce 2,563 training, 549 validation and 550 test images. Image-ID sets are disjoint. This verifies image-ID separation only; patient-level separation and near-duplicate exclusion have not been established. The labelled internal test split is distinct from Kaggle's unlabelled competition test set.

Images are converted to RGB, resized to 224 by 224 using torchvision/PIL, converted to tensors and normalised using mean [0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. A pretrained ResNet18 has frozen backbone parameters and a new 512-to-1 linear head (513 trainable parameters). The backbone remains in evaluation mode during head training, preserving BatchNorm statistics. Training uses BCEWithLogitsLoss, Adam with learning rate 0.001 and batch size 32. Only one completed epoch is evidenced. A separate epochs=5 placeholder does not implement five epochs.

The recorded training loss is 0.4144, training accuracy 0.8697 and elapsed time 277.6 seconds. Validation loss is 0.2969 and accuracy 0.9053. The saved checkpoint is resnet18_epoch1.pth. The binary decision rule is sigmoid score >=0.5; scores have not been calibrated as clinical probabilities.

## Held-out results
The saved test evaluation and exported prediction arrays give 94.00% accuracy, 92.83% sensitivity, 95.20% specificity, ROC AUC 0.9754 and positive-class F1 0.9401. The confusion matrix is TN=258, FP=13, FN=20, TP=259 (271 negatives, 279 positives). Thus 20 positive images were missed. These are point estimates on one internal split, not evidence of clinical safety or superiority to other systems. No confidence intervals, repeated-seed estimates or external validation were originally reported.

Validation results must not be substituted for test results: the initial validation confusion matrix was TN=248, FP=23, FN=29, TP=249, with AUC 0.9613. The 29 missed positive validation images had grade counts 13, 12, 2 and 2 for grades 1–4 respectively. These are exploratory validation findings, not test subgroup performance.

## Controlled blur and quality findings
The first blur experiment applies Gaussian kernels 3, 7 and 15 at native resolution before the standard transform. Original/mild/moderate/strong accuracies were 0.9053/0.9107/0.9071/0.9071 and AUCs 0.9613/0.9612/0.9608/0.9598. Small native-resolution kernels followed by resizing produced little change in this experiment; this does not establish blur robustness.

The second experiment first resizes with OpenCV to 224 by 224, then applies kernels 3, 9 and 21 with sigma=kernel/3. Original/mild/moderate/strong accuracies were 0.9126/0.9071/0.8889/0.8251; sensitivities 0.9065/0.8957/0.8741/0.9173; AUCs 0.9619/0.9579/0.9377/0.9121; false negatives 26/29/35/23. Strong blur had TN=198, FP=73, FN=23, TP=255. Its higher sensitivity therefore accompanies many false positives and lower accuracy/AUC; it is not an improvement in overall robustness. Its specificity, derived from those counts, is 198/271=73.06%.

The unblurred baseline differs between experiments because the resizing pipelines differ (PIL/torchvision versus OpenCV). Compare conditions within each experiment; a matched preprocessing ablation is still needed.

Quality measures are grayscale Laplacian variance, mean intensity and intensity standard deviation. In the 100-image training sample, their means were 25.10, 64.69 and 40.29. Missed positive validation images had higher mean Laplacian variance (26.45) than detected positives (11.64), so this evidence does not support a simple claim that lower edge detail explains the misses. Severity, retinal structure, resolution and borders may confound these measurements; no causal analysis was performed.

## Demonstration and interpretability
The demonstration integrates classification and review flags. Thresholds are edge-detail score <10, brightness <40 or >100, contrast <25, or sigmoid score in [0.4,0.6]. These are heuristic flags, not validated gradability or calibrated uncertainty estimates. A flag does not suppress the model output, and no flag does not imply safety.

For validation image f4c7ae514c54, original, blurred (31 by 31, sigma 8) and darkened (intensity multiplier 0.35) variants all retained the experimental No-DR class, with scores 0.1325, 0.1158 and 0.2121. Review flags were false, true and true respectively. Blur produced the low-edge-detail flag; darkening produced low-edge-detail, low-brightness and low-contrast flags. This is one illustrative image, not a quantitative evaluation of review-rule sensitivity.

Grad-CAM targets the final ResNet18 convolutional block and the positive DR logit, including when the predicted class is negative. The recorded false-negative example e25ccfe38e44 has score 0.2734; the selected confident correct grade-4 example cc3d2e961768 has score 0.9870. These selected illustrations do not validate lesion localisation, causality or explanation reliability. Original embedded panels are preserved privately.

## Reproducibility audit and limitations
Original code and outputs are retained. Empty history plots and checkpoint-existence diagnostics are historical development artefacts, not successful multi-epoch training. Generic variables such as y_true/y_prob and results are overwritten across experiments; export uses explicitly named test arrays and checks their alignment and confusion matrix. Saved output is evidence of a historical run; a clean-runtime execution remains outstanding.

Split seeds are fixed, but initial head weights and training shuffle are not explicitly seeded in the original run. The original random trajectory cannot be recovered retrospectively. A checkpoint and split manifests enable inference replication; a fresh training run may differ. Full restart-and-run-all replication has not been performed during packaging, and historical outputs were not replaced. Requirements record the active runtime, which is not a portable CUDA lockfile. Credentials and dataset archives are excluded from the public scaffold. Private notebooks and figure exports can contain restricted retinal images and must not be uploaded wholesale to GitHub.

Further limitations include single-dataset selection bias, image-level rather than patient-level splitting, possible duplicate leakage, binary reduction of severity, synthetic degradations that do not reproduce all optical artefacts, unvalidated quality thresholds, selected explanation examples, absent calibration and uncertainty intervals, and absent physical-device, latency, fairness or prospective clinical evaluation.

## Future research
First, establish patient/site grouping where available, audit near-duplicates and preregister a locked evaluation protocol. Repeat training with explicit seeds and immutable configuration, retain raw predictions and calculate appropriate uncertainty intervals. Use validation-only selection for thresholds and calibration, then evaluate once on an untouched external dataset. Match resize/interpolation across robustness baselines and investigate real acquisition degradation, exposure and motion. Validate quality measures against expert gradability labels and assess review coverage versus residual error. Evaluate explanations against suitable annotations and sanity checks. Only after appropriate governance should a physical-camera study examine usability, latency, acquisition quality and prospective performance. Exeter-specific supervisor or laboratory fit should be personalised separately rather than invented.

## Evidence provenance
Primary evidence is the supplied Colab notebook (original zero-based cells 0–38), its saved outputs, the existing checkpoint and runtime arrays. Cells 7–8 define splits, 13–21 define the model/training/validation, 25 and 27–29 define blur experiments, 30–32 define Grad-CAM, 33–34 define test evaluation and 36–38 define the demonstration. Exported CSVs retain numerical precision where available; report values follow displayed rounding. No new model training or clinical experiments were performed in preparing this report.

## Method references

- He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. arXiv:1512.03385 (2015). https://arxiv.org/abs/1512.03385
- Selvaraju, R. R., et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. arXiv:1610.02391; ICCV 2017 / revised IJCV version. https://arxiv.org/abs/1610.02391
- APTOS 2019 Blindness Detection, Kaggle competition source: https://www.kaggle.com/competitions/aptos2019-blindness-detection

These identify upstream methods and data; all project performance claims derive from the supplied notebook, not these references.
