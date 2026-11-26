"""Analyse matched retraining, common untouched test images, and controlled-test robustness."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models

from .model import quality_metrics, review_flags, sha256
from .publication_experiments import (CONDITIONS, MEASURES, TENSOR, condition_metrics,
                                      grade_sensitivity, prepared, review_coverage)


def controlled_condition_features(test, images, output, batch_size=32):
    path = output / 'controlled_condition_features_PRIVATE.npz'
    quality_path = output / 'controlled_quality_PRIVATE.csv'
    if path.exists() and quality_path.exists():
        return path, quality_path
    extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    extractor.fc = torch.nn.Identity()
    extractor.eval()
    blocks, rows = {}, []
    for condition in CONDITIONS:
        vectors, pending = [], []
        def flush():
            if pending:
                with torch.inference_mode():
                    vectors.append(extractor(torch.stack(pending)).numpy())
                pending.clear()
        for row in test.itertuples(index=False):
            with Image.open(images / f'{row.id_code}.png') as image:
                image.load()
                native = quality_metrics(image)
                array = prepared(image, condition)
            quality = native if condition == 'original' else quality_metrics(Image.fromarray(array))
            flags, _ = review_flags(quality, -1)
            rows.append({'condition': condition, 'id_code': row.id_code,
                         'blur_score': quality['blur_score'], 'brightness': quality['brightness'],
                         'contrast': quality['contrast'], 'quality_flags': '|'.join(flags),
                         'quality_flagged': bool(flags)})
            pending.append(TENSOR(Image.fromarray(array)))
            if len(pending) == batch_size:
                flush()
        flush()
        blocks[condition] = np.concatenate(vectors)
        print(f'Controlled test features: {condition}, {len(test)} images', flush=True)
    np.savez_compressed(path, ids=test.id_code.to_numpy(dtype=str), **blocks)
    pd.DataFrame(rows).to_csv(quality_path, index=False)
    return path, quality_path


def scored_frame(all_scores, manifest):
    frame = manifest[['id_code', 'diagnosis', 'binary_label']].merge(
        all_scores[['id_code', 'score']], on='id_code', validate='one_to_one')
    frame['prediction'] = (frame.score >= .5).astype(int)
    frame['condition'] = 'original'
    return frame


def group_bootstrap(frame, group_keys, resamples=2000, seed=20261126):
    """Percentile intervals resampling confirmed duplicate groups, with class strata."""
    indexed = frame.copy()
    indexed['group_key'] = indexed.id_code.map(group_keys)
    if indexed.group_key.isna().any():
        raise ValueError('Missing duplicate group for bootstrap.')
    groups = [part.index.to_numpy() for _, part in indexed.groupby('group_key', sort=True)]
    group_class = np.array([int(indexed.loc[indices, 'binary_label'].mean() >= .5) for indices in groups])
    strata = [np.flatnonzero(group_class == value) for value in (0, 1)]
    if any(len(s) == 0 for s in strata):
        raise ValueError('Both group classes are required for stratified bootstrap.')
    y, scores = indexed.binary_label, indexed.score
    rng = np.random.default_rng(seed)
    values = {key: [] for key in MEASURES}
    for _ in range(resamples):
        selected = np.concatenate([rng.choice(stratum, len(stratum), replace=True) for stratum in strata])
        positions = np.concatenate([groups[i] for i in selected])
        yy, ss = y.loc[positions].to_numpy(), scores.loc[positions].to_numpy()
        pred = ss >= .5
        tn = int(((yy == 0) & ~pred).sum()); fp = int(((yy == 0) & pred).sum())
        fn = int(((yy == 1) & ~pred).sum()); tp = int(((yy == 1) & pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.
        observed = {'accuracy': (tn + tp) / len(yy), 'sensitivity': tp / (tp + fn),
                    'specificity': tn / (tn + fp), 'precision': precision,
                    'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                    'roc_auc': roc_auc_score(yy, ss)}
        for key in MEASURES:
            values[key].append(observed[key])
    point = condition_metrics(frame)[0].iloc[0]
    return pd.DataFrame([{'metric': key, 'estimate': point[key],
                          'lower_95': np.quantile(values[key], .025),
                          'upper_95': np.quantile(values[key], .975),
                          'resamples': resamples, 'seed': seed,
                          'method': 'stratified confirmed-group bootstrap percentile'}
                         for key in MEASURES])


def render_figures(comparison, degradation, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    for subset, filename in [('own_test', 'split_comparison.png'), ('common_uncontaminated', 'common_subset_comparison.png')]:
        part = comparison[comparison.subset == subset]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        for ax, key in zip(axes.flat, ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc')):
            for name, color in [('original', '#a45a2a'), ('controlled', '#176b72')]:
                row = part[part.experiment == name].sort_values('seed')
                ax.plot(row.seed.astype(str), row[key], marker='o', label=name, color=color)
            ax.set(title=key.replace('_', ' ').title(), ylim=(max(.75, float(part[key].min()) - .06), 1))
            ax.grid(alpha=.2)
        axes.flat[0].legend()
        fig.suptitle(f'Matched ImageNet ResNet-18 head training: {subset.replace("_", " ")}')
        fig.tight_layout(); fig.savefig(output / filename, dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, color in [('accuracy', '#176b72'), ('sensitivity', '#a45a2a'),
                          ('specificity', '#3a5b96'), ('roc_auc', '#764796')]:
        mean = degradation.groupby('condition')[metric].mean().reindex(CONDITIONS)
        ax.plot(range(len(mean)), mean, marker='o', color=color, label=metric)
    ax.set_xticks(range(len(CONDITIONS)), ['Original', 'Mild blur', 'Moderate blur', 'Severe blur', 'Low brightness', 'Low contrast'], rotation=20, ha='right')
    ax.set(ylabel='Mean across three seeds', ylim=(.7, 1), title='Duplicate-controlled test: fixed synthetic degradations')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'controlled_degradation.png', dpi=300); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-splits', type=Path, required=True)
    parser.add_argument('--controlled-splits', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if not (args.training / 'completed.json').is_file():
        raise ValueError('Matched training must finish before analysis.')
    args.output.mkdir(parents=True, exist_ok=True)
    original = {name: pd.read_csv(args.original_splits / f'{name}.csv') for name in ('train', 'val', 'test')}
    controlled = {name: pd.read_csv(args.controlled_splits / f'{name}.csv') for name in ('train', 'val', 'test')}
    hashes = pd.read_csv(args.groups).set_index('id_code').group_key
    overlap = set(original['test'].id_code) & set(controlled['test'].id_code)
    original_training_hashes = set(hashes.loc[pd.concat([original['train'], original['val']]).id_code])
    controlled_training_hashes = set(hashes.loc[pd.concat([controlled['train'], controlled['val']]).id_code])
    common = {image_id for image_id in overlap if hashes.loc[image_id] not in original_training_hashes
              and hashes.loc[image_id] not in controlled_training_hashes}
    analysis_plan = {'common_test_intersection': len(overlap), 'common_uncontaminated_images': len(common),
                     'definition': 'Test in both splits; confirmed duplicate group absent from both train/validation sets',
                     'patient_level_separation_verified': False}
    (args.output / 'analysis_plan.json').write_text(json.dumps(analysis_plan, indent=2))
    comparisons, intervals, grades = [], [], []
    for seed in args.seeds:
        for name, splits in [('original', original), ('controlled', controlled)]:
            saved = pd.read_csv(args.training / f'{name}_seed{seed}' / 'all_scores_PRIVATE.csv')
            for subset, manifest in [('own_test', splits['test']),
                                     ('common_uncontaminated', splits['test'][splits['test'].id_code.isin(common)])]:
                frame = scored_frame(saved, manifest)
                if frame.binary_label.nunique() != 2:
                    continue
                metric = condition_metrics(frame)[0].iloc[0].to_dict()
                metric.update({'experiment': name, 'seed': seed, 'subset': subset})
                comparisons.append(metric)
                ci = group_bootstrap(frame, hashes, args.bootstrap, 20261126 + seed)
                ci['experiment'] = name; ci['seed'] = seed; ci['subset'] = subset
                intervals.append(ci)
                if subset == 'own_test':
                    grade = grade_sensitivity(frame)
                    grade['experiment'] = name; grade['seed'] = seed
                    grades.append(grade)
        pd.DataFrame(comparisons).to_csv(args.output / 'model_comparison.csv', index=False)
    pd.concat(intervals).to_csv(args.output / 'model_intervals.csv', index=False)
    pd.concat(grades).to_csv(args.output / 'grade_sensitivity.csv', index=False)
    # For all three seeds, reuse one controlled-test feature extraction pass for fixed degradations.
    cache, quality_path = controlled_condition_features(controlled['test'], args.images, args.output)
    stored = np.load(cache)
    if not np.array_equal(stored['ids'], controlled['test'].id_code.to_numpy()):
        raise ValueError('Condition features and test manifest differ.')
    quality = pd.read_csv(quality_path)
    condition_rows, coverage_rows = [], []
    for seed in args.seeds:
        checkpoint = args.training / f'controlled_seed{seed}' / 'best.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        weight = state['fc.weight'].numpy().reshape(-1)
        bias = float(state['fc.bias'].item())
        predictions = []
        for condition in CONDITIONS:
            logits = stored[condition] @ weight + bias
            scores = 1 / (1 + np.exp(-logits))
            frame = controlled['test'].copy()
            frame['condition'] = condition
            frame['score'] = scores
            frame['prediction'] = (scores >= .5).astype(int)
            q = quality[quality.condition == condition].reset_index(drop=True)
            if not np.array_equal(q.id_code, frame.id_code):
                raise ValueError('Quality feature order mismatch.')
            frame['quality_flags'] = q.quality_flags
            frame['review_required'] = q.quality_flagged.to_numpy() | ((scores >= .4) & (scores <= .6))
            predictions.append(frame)
        paired = pd.concat(predictions, ignore_index=True)
        paired.to_csv(args.output / f'controlled_seed{seed}_paired_predictions_PRIVATE.csv', index=False)
        metric_table, confusion = condition_metrics(paired)
        metric_table['seed'] = seed; confusion['seed'] = seed
        condition_rows.append(metric_table)
        confusion.to_csv(args.output / f'controlled_seed{seed}_confusion.csv', index=False)
        cov = review_coverage(paired[paired.condition == 'original'])
        cov['seed'] = seed
        coverage_rows.append(cov)
    pd.concat(condition_rows).to_csv(args.output / 'controlled_degradation_metrics.csv', index=False)
    pd.concat(coverage_rows).to_csv(args.output / 'controlled_review_coverage.csv', index=False)
    # Shareable figures show seed-level results; no retinal images or IDs leave private runs.
    render_figures(pd.DataFrame(comparisons), pd.concat(condition_rows), args.output)
    (args.output / 'completed.json').write_text(json.dumps({**analysis_plan, 'seeds': args.seeds,
                                                             'bootstrap_resamples': args.bootstrap,
                                                             'controlled_manifest_sha256': sha256(args.controlled_splits / 'test.csv')}, indent=2))
    print(json.dumps(analysis_plan, indent=2), flush=True)


if __name__ == '__main__':
    main()
