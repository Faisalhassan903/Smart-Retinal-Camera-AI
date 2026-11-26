"""Locked, retrospective APTOS analyses. Outputs contain private image identifiers."""
import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
import PIL
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from torchvision import transforms

from .model import Predictor, quality_metrics, review_flags, sha256
from .train import load_splits, metrics

CONDITIONS = {
    'original': {'operation': 'identity'},
    'blur_mild': {'operation': 'gaussian_blur', 'kernel': 3, 'sigma': 1.0},
    'blur_moderate': {'operation': 'gaussian_blur', 'kernel': 9, 'sigma': 3.0},
    'blur_severe': {'operation': 'gaussian_blur', 'kernel': 21, 'sigma': 7.0},
    'brightness_low': {'operation': 'brightness', 'factor': 0.65},
    'contrast_low': {'operation': 'contrast', 'factor': 0.65},
}
MEASURES = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc')
HISTORICAL = {'accuracy': .94, 'sensitivity': 259 / 279, 'specificity': 258 / 271,
              'f1': 0.9401088929219601, 'roc_auc': 0.97539975399754,
              'TN': 258, 'FP': 13, 'FN': 20, 'TP': 259}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TENSOR = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def manifest_audit(splits):
    frames = {name: pd.read_csv(Path(splits) / f'{name}.csv') for name in ('train', 'val', 'test')}
    combined = pd.concat(frames.values(), ignore_index=True)
    # load_splits performs the label, ID and coverage checks against the union.
    load_splits_frame(combined, frames)
    rows = []
    for name, frame in frames.items():
        counts = frame.diagnosis.value_counts().to_dict()
        rows.append({'partition': name, 'images': len(frame),
                     **{f'grade_{grade}': counts.get(grade, 0) for grade in range(5)},
                     'no_dr': counts.get(0, 0), 'dr': len(frame) - counts.get(0, 0),
                     'manifest_sha256': sha256(Path(splits) / f'{name}.csv')})
    return frames, pd.DataFrame(rows)


def load_splits_frame(combined, frames):
    from .train import validate_frame
    full = validate_frame(combined)
    validated = [validate_frame(frames[name]) for name in ('train', 'val', 'test')]
    ids = [set(frame.id_code) for frame in validated]
    if any(a & b for a, b in combinations(ids, 2)):
        raise ValueError('Image IDs overlap between partitions.')
    if len(set.union(*ids)) != len(full):
        raise ValueError('Image IDs are duplicated or omitted.')
    return validated


def dhash(image):
    gray = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def phash(image):
    """64-bit DCT hash; a candidate screen rather than an identity test."""
    gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8]
    bits = (low > np.median(low[1:, :])).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def refine_perceptual_candidates(images, hash_manifest, candidates, output, maximum_distance=6):
    manifest = pd.read_csv(hash_manifest)
    hashes = {}
    for image_id in manifest.id_code:
        with Image.open(images / f'{image_id}.png') as image:
            hashes[image_id] = phash(image)
    pd.DataFrame([{'id_code': image_id, 'phash64': value} for image_id, value in hashes.items()]).to_csv(
        output / 'phashes_PRIVATE.csv', index=False)
    kept = []
    for row in pd.read_csv(candidates).itertuples(index=False):
        distance = (hashes[row.id_a] ^ hashes[row.id_b]).bit_count()
        if distance <= maximum_distance:
            kept.append((row.partition_a, row.id_a, row.partition_b, row.id_b,
                         row.dhash_hamming_distance, distance))
    pd.DataFrame(kept, columns=['partition_a', 'id_a', 'partition_b', 'id_b',
                                'dhash_hamming_distance', 'phash_hamming_distance']).to_csv(
        output / 'perceptual_candidates_refined.csv', index=False)
    return len(kept)


def rank_perceptual_candidates(images, candidates, output):
    """Rank surviving hash candidates by aligned low-resolution image similarity."""
    frame = pd.read_csv(candidates)
    ids = set(frame.id_a) | set(frame.id_b)
    thumbnails = {}
    for image_id in ids:
        with Image.open(images / f'{image_id}.png') as image:
            thumbnails[image_id] = np.asarray(image.convert('L').resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    rows = []
    for pair in frame.itertuples(index=False):
        a, b = thumbnails[pair.id_a], thumbnails[pair.id_b]
        ac, bc = a - a.mean(), b - b.mean()
        denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
        correlation = float(np.dot(ac, bc) / denominator) if denominator else 0.0
        rows.append({'partition_a': pair.partition_a, 'id_a': pair.id_a,
                     'partition_b': pair.partition_b, 'id_b': pair.id_b,
                     'dhash_hamming_distance': pair.dhash_hamming_distance,
                     'phash_hamming_distance': pair.phash_hamming_distance,
                     'grayscale_correlation': correlation,
                     'mean_absolute_grayscale_difference': float(np.abs(a - b).mean())})
    ranked = pd.DataFrame(rows).sort_values(['grayscale_correlation', 'mean_absolute_grayscale_difference'], ascending=[False, True])
    ranked.to_csv(output / 'perceptual_candidates_ranked_PRIVATE.csv', index=False)
    shortlist = ranked[(ranked.grayscale_correlation >= .99) & (ranked.mean_absolute_grayscale_difference <= 15)]
    shortlist.to_csv(output / 'perceptual_shortlist_PRIVATE.csv', index=False)
    return len(shortlist)


def duplicate_summary(frames, output, duplicate_counts):
    pairs = pd.read_csv(output / 'exact_cross_partition.csv')
    affected = {partition: set(pairs.loc[pairs.partition_a == partition, 'id_a']) |
                set(pairs.loc[pairs.partition_b == partition, 'id_b'])
                for partition in ('train', 'val', 'test')}
    labels = pd.concat([frame.assign(partition=partition) for partition, frame in frames.items()])
    hashes = pd.read_csv(output / 'image_hashes_PRIVATE.csv')
    merged = hashes.merge(labels, on=['partition', 'id_code'], validate='one_to_one')
    groups = [group for _, group in merged.groupby('sha256') if group.partition.nunique() > 1]
    summary = {'exact_cross_partition_pairs': len(pairs), 'exact_cross_partition_groups': len(groups),
               **{f'exact_affected_{name}_images': len(ids) for name, ids in affected.items()},
               'exact_grade_discordant_groups': sum(group.diagnosis.nunique() > 1 for group in groups),
               'exact_binary_discordant_groups': sum(group.binary_label.nunique() > 1 for group in groups),
               'dhash_candidates': duplicate_counts['perceptual_candidate_pairs'],
               'phash_refined_candidates': duplicate_counts['phash_refined_candidates'],
               'high_similarity_shortlist': duplicate_counts['high_similarity_shortlist'],
               'patient_level_separation_verified': False,
               'full_visual_review_completed': False}
    pd.DataFrame([summary]).to_csv(output / 'duplicate_audit_summary.csv', index=False)
    return summary


def duplicate_audit(frames, images, output, maximum_distance=5):
    records = []
    for partition, frame in frames.items():
        for image_id in frame.id_code:
            path = images / f'{image_id}.png'
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image.load()
                fingerprint = dhash(image)
            records.append((partition, image_id, sha256(path), fingerprint))
    by_hash = {}
    for partition, image_id, exact, perceptual in records:
        by_hash.setdefault(exact, []).append((partition, image_id))
    pd.DataFrame(records, columns=['partition', 'id_code', 'sha256', 'dhash64']).to_csv(
        output / 'image_hashes_PRIVATE.csv', index=False)
    exact_rows = [{'partition_a': a[0], 'id_a': a[1], 'partition_b': b[0], 'id_b': b[1], 'sha256': digest}
                  for digest, group in by_hash.items() for a, b in combinations(group, 2) if a[0] != b[0]]
    pd.DataFrame(exact_rows, columns=['partition_a', 'id_a', 'partition_b', 'id_b', 'sha256']).to_csv(output / 'exact_cross_partition.csv', index=False)
    # Cross-partition dHash is a candidate screen, not proof of duplicate identity.
    near_rows = []
    for i, (part_a, id_a, exact_a, hash_a) in enumerate(records):
        for part_b, id_b, exact_b, hash_b in records[i + 1:]:
            if part_a == part_b or exact_a == exact_b:
                continue
            distance = (hash_a ^ hash_b).bit_count()
            if distance <= maximum_distance:
                near_rows.append((part_a, id_a, part_b, id_b, distance))
    pd.DataFrame(near_rows, columns=['partition_a', 'id_a', 'partition_b', 'id_b', 'dhash_hamming_distance']).to_csv(output / 'perceptual_candidates.csv', index=False)
    return {'images_hashed': len(records), 'exact_cross_partition_pairs': len(exact_rows),
            'perceptual_candidate_pairs': len(near_rows), 'dhash_max_hamming_distance': maximum_distance,
            'patient_level_separation_verified': False}


def prepared(image, condition):
    # One PIL bilinear resize for *every* condition; degradations follow it.
    array = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)).copy()
    spec = CONDITIONS[condition]
    if spec['operation'] == 'gaussian_blur':
        array = cv2.GaussianBlur(array, (spec['kernel'], spec['kernel']), spec['sigma'])
    elif spec['operation'] == 'brightness':
        array = np.uint8(np.clip(np.rint(array.astype(np.float32) * spec['factor']), 0, 255))
    elif spec['operation'] == 'contrast':
        mean = array.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        array = np.uint8(np.clip(np.rint(mean + spec['factor'] * (array.astype(np.float32) - mean)), 0, 255))
    return array


def score_conditions(predictor, frame, images, batch_size=32):
    rows = []
    model = predictor.model
    for condition in CONDITIONS:
        pending, metadata = [], []
        def flush():
            if not pending:
                return
            with torch.inference_mode():
                scores = torch.sigmoid(model(torch.stack(pending).to(predictor.device)).squeeze(1)).cpu().numpy()
            for (image_id, grade, label, blur, brightness, contrast, flags, uncertain), score in zip(metadata, scores):
                rows.append({'condition': condition, 'id_code': image_id, 'diagnosis': grade,
                             'binary_label': label, 'score': float(score), 'prediction': int(score >= .5),
                             'blur_score': blur, 'brightness': brightness, 'contrast': contrast,
                             'quality_flags': '|'.join(flags), 'uncertain_prediction': .4 <= score <= .6,
                             'review_required': bool(flags) or (.4 <= score <= .6)})
            pending.clear(); metadata.clear()
        for row in frame.itertuples(index=False):
            with Image.open(images / f'{row.id_code}.png') as image:
                image.load()
                # Original-resolution quality values implement the existing review rules.
                native_quality = quality_metrics(image)
                array = prepared(image, condition)
            # Quality rules on each evaluated image; baseline uses native quality.
            quality = native_quality if condition == 'original' else quality_metrics(Image.fromarray(array))
            flags, uncertain = review_flags(quality, -1)
            pending.append(TENSOR(Image.fromarray(array)))
            metadata.append((row.id_code, int(row.diagnosis), int(row.binary_label), quality['blur_score'],
                             quality['brightness'], quality['contrast'], flags, uncertain))
            if len(pending) == batch_size:
                flush()
        flush()
        print(f'Evaluated condition: {condition} ({len(frame)} images)', flush=True)
    return pd.DataFrame(rows)


def condition_metrics(predictions):
    rows, confusion = [], []
    for condition, frame in predictions.groupby('condition', sort=False):
        result = metrics(frame.binary_label, frame.score)
        result['condition'] = condition
        result['precision'] = precision_score(frame.binary_label, frame.prediction, zero_division=0)
        rows.append(result)
        confusion.extend([{'condition': condition, 'actual': actual, 'predicted': predicted,
                           'count': int(confusion_matrix(frame.binary_label, frame.prediction, labels=[0, 1])[actual, predicted])}
                          for actual in (0, 1) for predicted in (0, 1)])
    return pd.DataFrame(rows), pd.DataFrame(confusion)


def bootstrap_intervals(frame, resamples=2000, seed=20261126):
    y = frame.binary_label.to_numpy(dtype=int)
    score = frame.score.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in MEASURES}
    # Stratified sampling retains both classes, giving defined sensitivity, specificity and AUC.
    classes = [np.flatnonzero(y == label) for label in (0, 1)]
    for _ in range(resamples):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        yy, ss = y[index], score[index]
        pred = ss >= .5
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
        values = {'accuracy': accuracy_score(yy, pred), 'sensitivity': tp / (tp + fn),
                  'specificity': tn / (tn + fp), 'precision': precision_score(yy, pred, zero_division=0),
                  'f1': f1_score(yy, pred, zero_division=0), 'roc_auc': roc_auc_score(yy, ss)}
        for name in MEASURES:
            samples[name].append(values[name])
    point = condition_metrics(frame)[0].iloc[0]
    return pd.DataFrame([{'metric': name, 'estimate': point[name],
                          'lower_95': np.quantile(samples[name], .025), 'upper_95': np.quantile(samples[name], .975),
                          'resamples': resamples, 'seed': seed, 'method': 'stratified image bootstrap percentile'}
                         for name in MEASURES])


def grade_sensitivity(frame):
    return pd.DataFrame([{'grade': grade, 'positive_images': len(part),
                          'true_positives': int(part.prediction.sum()),
                          'false_negatives': int((part.prediction == 0).sum()),
                          'sensitivity': float(part.prediction.mean())}
                         for grade, part in frame[frame.diagnosis > 0].groupby('diagnosis')])


def review_coverage(frame):
    errors = frame.prediction != frame.binary_label
    false_negatives = (frame.binary_label == 1) & (frame.prediction == 0)
    flagged = frame.review_required.astype(bool)
    unflagged = ~flagged
    return pd.DataFrame([{'images': len(frame), 'flagged_images': int(flagged.sum()),
                          'flagged_pct': 100 * flagged.mean(), 'classification_errors': int(errors.sum()),
                          'errors_captured': int((errors & flagged).sum()),
                          'errors_captured_pct': 100 * (errors & flagged).sum() / errors.sum() if errors.sum() else np.nan,
                          'false_negatives': int(false_negatives.sum()),
                          'false_negatives_captured': int((false_negatives & flagged).sum()),
                          'false_negatives_captured_pct': 100 * (false_negatives & flagged).sum() / false_negatives.sum() if false_negatives.sum() else np.nan,
                          'unflagged_images': int(unflagged.sum()),
                          'residual_errors': int((errors & unflagged).sum()),
                          'residual_error_pct': 100 * (errors & unflagged).sum() / unflagged.sum() if unflagged.sum() else np.nan,
                          'residual_false_negatives': int((false_negatives & unflagged).sum())}])


def exact_duplicate_sensitivity(frame, exact_pairs):
    pairs = pd.read_csv(exact_pairs)
    affected = set(pairs.loc[pairs.partition_a == 'test', 'id_a']) | set(pairs.loc[pairs.partition_b == 'test', 'id_b'])
    outputs = []
    for name, part in [('all_original_test', frame), ('exact_duplicate_test_images', frame[frame.id_code.isin(affected)]),
                       ('excluding_exact_duplicates', frame[~frame.id_code.isin(affected)])]:
        row = {'group': name, 'n': len(part), 'no_dr': int((part.binary_label == 0).sum()),
               'dr': int((part.binary_label == 1).sum()),
               'correct': int((part.prediction == part.binary_label).sum()),
               'false_negatives': int(((part.binary_label == 1) & (part.prediction == 0)).sum()),
               'accuracy': float((part.prediction == part.binary_label).mean()),
               'sensitivity': float(part.loc[part.binary_label == 1, 'prediction'].mean())}
        if part.binary_label.nunique() == 2:
            row.update({'specificity': float((part.loc[part.binary_label == 0, 'prediction'] == 0).mean()),
                        'roc_auc': roc_auc_score(part.binary_label, part.score)})
        outputs.append(row)
    return pd.DataFrame(outputs)


def figures(metrics_frame, intervals, grades, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'ROC AUC']
    for i, row in intervals.reset_index(drop=True).iterrows():
        ax.plot([row.lower_95, row.upper_95], [i, i], color='#176b72', lw=3)
        ax.plot(row.estimate, i, 'o', color='#ad5a20', ms=7)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlim=(max(0, float(intervals.lower_95.min()) - .03), 1),
           xlabel='Estimate and 95% percentile interval',
           title='APTOS held-out baseline: image bootstrap intervals')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'baseline_intervals.png', dpi=300); plt.close(fig)
    if len(metrics_frame) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = ['Original', 'Mild blur', 'Moderate blur', 'Severe blur', 'Low brightness', 'Low contrast']
        x = range(len(metrics_frame))
        for name, label, color in [('accuracy', 'Accuracy', '#176b72'), ('sensitivity', 'Sensitivity', '#ad5a20'),
                                   ('specificity', 'Specificity', '#3a5b96'), ('roc_auc', 'ROC AUC', '#7c4694')]:
            ax.plot(x, metrics_frame[name], marker='o', label=label, color=color)
        ax.set_xticks(list(x), names, rotation=25, ha='right')
        ax.set(ylabel='Metric value', title='Paired test images under prespecified degradations', ylim=(.7, 1))
        ax.legend(ncol=2); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output / 'robustness.png', dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grades.grade.astype(str), grades.sensitivity, color='#176b72')
    ax.set(xlabel='Disease grade', ylabel='Sensitivity', ylim=(0, 1), title='Held-out DR sensitivity by recorded grade')
    fig.tight_layout(); fig.savefig(output / 'grade_sensitivity.png', dpi=300); plt.close(fig)


def compare_historical(metric_table):
    observed = metric_table.iloc[0]
    rows = []
    for metric, historical in HISTORICAL.items():
        actual = float(observed[metric])
        rows.append({'metric': metric, 'historical': historical, 'evaluated': actual,
                     'difference': actual - historical,
                     'matches': abs(actual - historical) <= (0 if metric in ('TN', 'FP', 'FN', 'TP') else 5e-4)})
    return pd.DataFrame(rows)


def analyse_saved_predictions(path, test_frame, output, resamples, seed):
    saved = pd.read_csv(path)
    required = {'id_code', 'diagnosis', 'binary_label', 'score', 'prediction'}
    if not required.issubset(saved.columns) or len(saved) != len(test_frame) or not saved.id_code.is_unique:
        raise ValueError('Saved predictions must contain one row per held-out image.')
    expected = test_frame.set_index('id_code')[['diagnosis', 'binary_label']]
    if set(saved.id_code) != set(expected.index):
        raise ValueError('Saved predictions do not match the preserved test IDs.')
    aligned = expected.loc[saved.id_code]
    if not np.array_equal(saved[['diagnosis', 'binary_label']].to_numpy(), aligned.to_numpy()):
        raise ValueError('Saved prediction labels differ from the held-out manifest.')
    if saved.score.isna().any() or not saved.score.between(0, 1).all() or not np.array_equal(saved.prediction, (saved.score >= .5).astype(int)):
        raise ValueError('Saved scores or predictions are invalid.')
    saved['condition'] = 'original'
    metric_table, confusion = condition_metrics(saved)
    comparison = compare_historical(metric_table)
    if not comparison.matches.all():
        raise ValueError('Saved predictions do not reproduce historical summary metrics.')
    intervals = bootstrap_intervals(saved, resamples, seed)
    grades = grade_sensitivity(saved)
    metric_table.to_csv(output / 'historical_scores_metrics.csv', index=False)
    confusion.to_csv(output / 'historical_scores_confusion.csv', index=False)
    comparison.to_csv(output / 'historical_scores_comparison.csv', index=False)
    intervals.to_csv(output / 'bootstrap_intervals.csv', index=False)
    grades.to_csv(output / 'grade_sensitivity.csv', index=False)
    figures(metric_table, intervals, grades, output)
    return {'saved_predictions_sha256': sha256(path), 'historical_scores_reanalysis': 'completed',
            'saved_score_metrics_match_historical': True, 'bootstrap': 'completed from saved scores',
            'grade_sensitivity': 'completed from saved scores'}


def compare_saved_scores(original, saved_path, output):
    saved = pd.read_csv(saved_path)
    paired = original[['id_code', 'diagnosis', 'binary_label', 'score', 'prediction']].merge(
        saved[['id_code', 'diagnosis', 'binary_label', 'score', 'prediction']],
        on='id_code', suffixes=('_fresh', '_saved'), validate='one_to_one')
    if len(paired) != len(original) or not np.array_equal(paired.diagnosis_fresh, paired.diagnosis_saved) or not np.array_equal(paired.binary_label_fresh, paired.binary_label_saved):
        raise ValueError('Saved and fresh predictions are not aligned.')
    paired['absolute_score_difference'] = (paired.score_fresh - paired.score_saved).abs()
    paired.to_csv(output / 'score_reproduction_PRIVATE.csv', index=False)
    summary = pd.DataFrame([{'images': len(paired),
                             'mean_absolute_score_difference': paired.absolute_score_difference.mean(),
                             'max_absolute_score_difference': paired.absolute_score_difference.max(),
                             'prediction_disagreements': int((paired.prediction_fresh != paired.prediction_saved).sum())}])
    summary.to_csv(output / 'score_reproduction_summary.csv', index=False)
    return summary.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, default=Path('splits'))
    parser.add_argument('--labels', type=Path, help='Original Kaggle train.csv for full manifest coverage check')
    parser.add_argument('--images', type=Path, help='APTOS train_images directory; omit for manifest-only audit')
    parser.add_argument('--checkpoint', type=Path, default=Path('models/resnet18_epoch1.pth'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--saved-predictions', type=Path, help='Private historical test prediction CSV for analysis without images')
    parser.add_argument('--refine-candidates', type=Path, help='Existing private duplicate-audit run to refine with DCT pHash')
    parser.add_argument('--rank-candidates', type=Path, help='Existing refined audit run to rank by thumbnail similarity')
    parser.add_argument('--output', type=Path, default=Path('runs/publication'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261126)
    args = parser.parse_args()
    if args.refine_candidates:
        if not args.images:
            parser.error('--refine-candidates requires --images')
        count = refine_perceptual_candidates(args.images,
                                             args.refine_candidates / 'image_hashes_PRIVATE.csv',
                                             args.refine_candidates / 'perceptual_candidates.csv',
                                             args.refine_candidates)
        print(f'Refined perceptual candidates: {count}')
        return
    if args.rank_candidates:
        if not args.images:
            parser.error('--rank-candidates requires --images')
        count = rank_perceptual_candidates(args.images,
                                           args.rank_candidates / 'perceptual_candidates_refined.csv',
                                           args.rank_candidates)
        print(f'High-similarity candidates for manual review: {count}')
        return
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    frames, summary = manifest_audit(args.splits)
    if args.labels:
        load_splits(args.labels, args.splits)
    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / 'manifest_audit.csv', index=False)
    status = {'manifest_audit': 'completed', 'image_duplicate_audit': 'pending',
              'test_reproduction': 'pending', 'bootstrap': 'pending', 'robustness': 'pending',
              'review_flags': 'pending', 'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint.exists() else None,
              'test_manifest_sha256': sha256(args.splits / 'test.csv'),
              'original_labels_verified': bool(args.labels),
              'original_labels_sha256': sha256(args.labels) if args.labels else None,
              'software_versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                                    'opencv': cv2.__version__, 'pillow': PIL.__version__,
                                    'scikit_learn': sklearn.__version__, 'pandas': pd.__version__,
                                    'numpy': np.__version__},
              'patient_level_leakage': 'unresolved: patient IDs absent; image-level partitions only',
              'images_available': bool(args.images and args.images.is_dir())}
    if args.saved_predictions and not status['images_available']:
        status.update(analyse_saved_predictions(args.saved_predictions, frames['test'], output,
                                                args.bootstrap, args.bootstrap_seed))
    if not status['images_available']:
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        print(f'Manifests verified; image analyses pending. Saved: {output}')
        return
    images = args.images
    status['duplicate_audit'] = duplicate_audit(frames, images, output)
    status['duplicate_audit']['phash_refined_candidates'] = refine_perceptual_candidates(
        images, output / 'image_hashes_PRIVATE.csv', output / 'perceptual_candidates.csv', output)
    status['duplicate_audit']['high_similarity_shortlist'] = rank_perceptual_candidates(
        images, output / 'perceptual_candidates_refined.csv', output)
    status['duplicate_audit_summary'] = duplicate_summary(frames, output, status['duplicate_audit'])
    status['image_duplicate_audit'] = 'completed; candidates require visual review'
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    predictor = Predictor(args.checkpoint, device=device)
    predictions = score_conditions(predictor, frames['test'], images)
    predictions.to_csv(output / 'paired_predictions_PRIVATE.csv', index=False)
    metric_table, confusion = condition_metrics(predictions)
    metric_table.to_csv(output / 'condition_metrics.csv', index=False)
    confusion.to_csv(output / 'confusion_matrices.csv', index=False)
    original = predictions[predictions.condition == 'original'].copy()
    if args.saved_predictions:
        status['score_reproduction'] = compare_saved_scores(original, args.saved_predictions, output)
    intervals = bootstrap_intervals(original, args.bootstrap, args.bootstrap_seed)
    intervals.to_csv(output / 'bootstrap_intervals.csv', index=False)
    grades = grade_sensitivity(original)
    grades.to_csv(output / 'grade_sensitivity.csv', index=False)
    exact_duplicate_sensitivity(original, output / 'exact_cross_partition.csv').to_csv(
        output / 'exact_duplicate_sensitivity.csv', index=False)
    coverage = review_coverage(original)
    coverage.to_csv(output / 'review_coverage.csv', index=False)
    comparison = compare_historical(metric_table)
    comparison.to_csv(output / 'historical_comparison.csv', index=False)
    figures(metric_table, intervals, grades, output)
    status.update({'test_reproduction': 'completed', 'bootstrap': 'completed', 'robustness': 'completed',
                   'review_flags': 'completed', 'device_used': device,
                   'historical_metrics_all_match': bool(comparison.matches.all()),
                   'method': {'resize': 'PIL RGB bilinear 224x224 once before every degradation',
                              'normalization': {'mean': MEAN, 'std': STD}, 'conditions': CONDITIONS,
                              'review_thresholds': 'original existing fixed thresholds; no test-set tuning',
                              'bootstrap': 'stratified image-level percentile; patient clustering unresolved'}})
    (output / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2)); print(f'Saved: {output}')


if __name__ == '__main__':
    main()
