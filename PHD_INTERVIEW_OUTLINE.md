# Five-minute PhD interview outline

0:00–0:40 | Motivation and question. Explain the goal of combining image classification with acquisition-quality review in a research camera workflow. State that this is a retrospective prototype, not a diagnostic system. Visual: integrated_demo_summary.png.

0:40–1:30 | Method. APTOS 2019: 3,662 labelled images, grade 0 versus any DR; stratified 2,563/549/550 image split. Pretrained ResNet18, frozen backbone, 513-parameter head, one completed epoch, 224-pixel input. Explain why a small baseline enables a focused investigation of image quality.

1:30–2:15 | Held-out evidence. Show test_confusion_matrix.png and test_roc.png. State 94.00% accuracy, 92.83% sensitivity, 95.20% specificity, AUC 0.9754, F1 0.9401. Explicitly discuss the 20 missed positive images. Keep validation and test evidence separate.

2:15–3:10 | What the experiments taught me. Show blur_robustness.png. Strong blur reduced accuracy from 91.26% to 82.51%; sensitivity increased alongside 73 false positives. Explain the preprocessing difference between the two blur experiments. Demonstrate how review flags can change while the predicted class remains unchanged. Grad-CAM is exploratory, not proof of correct reasoning.

3:10–4:10 | Critical appraisal. Discuss image-level splitting, one dataset, unseeded original training, synthetic blur, heuristic quality thresholds and uncalibrated scores. Explain why no patient-level or clinical conclusion follows. Describe preservation of evidence and separation of private assets from the GitHub scaffold.

4:10–5:00 | PhD direction. Propose grouped/external evaluation, real-camera degradation and expert gradability labels, validated review policies, calibration and explanation sanity checks. Define success in terms of reliable performance and review trade-offs, not accuracy alone. Add a personally verified connection to the intended Exeter supervisor's work.

Likely questions: Why only one epoch? It is the completed baseline, not a claimed optimum. Does 94% imply diagnostic suitability? No. Why did sensitivity rise under strong blur? The model predicted more positives and specificity fell. Is Grad-CAM lesion validation? No. What is reproducible? Saved-checkpoint inference and split membership; the original random training trajectory is not fully specified.
