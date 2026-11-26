"""Paired group-bootstrap differences on common uncontaminated test images."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .publication_experiments import MEASURES
from .train import metrics


def measured(y, scores):
    result = metrics(y, scores)
    result['precision'] = result['TP'] / (result['TP'] + result['FP']) if result['TP'] + result['FP'] else 0.
    return result


def paired_intervals(frame, resamples=2000, seed=20261126):
    groups = [part.index.to_numpy() for _, part in frame.groupby('group_key', sort=True)]
    labels = frame.binary_label.to_numpy()
    strata = [np.array([i for i, indices in enumerate(groups) if (labels[indices].mean() >= .5) == bool(value)], dtype=int)
              for value in (0, 1)]
    if any(len(s) == 0 for s in strata):
        raise ValueError('Common test subset needs both classes of groups.')
    rng = np.random.default_rng(seed)
    original = frame.score_original.to_numpy()
    controlled = frame.score_controlled.to_numpy()
    point_original = measured(labels, original)
    point_controlled = measured(labels, controlled)
    samples = {key: [] for key in MEASURES}
    for _ in range(resamples):
        selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        idx = np.concatenate([groups[i] for i in selected])
        a = measured(labels[idx], original[idx]); b = measured(labels[idx], controlled[idx])
        for key in MEASURES:
            samples[key].append(b[key] - a[key])
    return pd.DataFrame([{'metric': key, 'controlled_minus_original': point_controlled[key] - point_original[key],
                          'lower_95': np.quantile(samples[key], .025),
                          'upper_95': np.quantile(samples[key], .975),
                          'resamples': resamples, 'method': 'paired stratified confirmed-group bootstrap percentile'}
                         for key in MEASURES])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-splits', type=Path, required=True)
    parser.add_argument('--controlled-splits', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    old_test = pd.read_csv(args.original_splits / 'test.csv')
    new_test = pd.read_csv(args.controlled_splits / 'test.csv')
    group_map = pd.read_csv(args.groups).set_index('id_code').group_key
    intersection = set(old_test.id_code) & set(new_test.id_code)
    old_fit = set(group_map.loc[pd.concat([pd.read_csv(args.original_splits / f'{part}.csv')
                                            for part in ('train', 'val')]).id_code])
    new_fit = set(group_map.loc[pd.concat([pd.read_csv(args.controlled_splits / f'{part}.csv')
                                            for part in ('train', 'val')]).id_code])
    common = sorted(image_id for image_id in intersection if group_map.loc[image_id] not in old_fit | new_fit)
    base = old_test[old_test.id_code.isin(common)][['id_code', 'binary_label']].sort_values('id_code')
    base['group_key'] = base.id_code.map(group_map)
    output = []
    for seed in args.seeds:
        paired = base.copy()
        for name in ('original', 'controlled'):
            scores = pd.read_csv(args.training / f'{name}_seed{seed}' / 'all_scores_PRIVATE.csv').set_index('id_code').score
            paired[f'score_{name}'] = paired.id_code.map(scores)
        paired = paired.reset_index(drop=True)
        if paired[['score_original', 'score_controlled']].isna().any().any():
            raise ValueError('Missing common-subset score.')
        paired.to_csv(args.output / f'common_seed{seed}_paired_PRIVATE.csv', index=False)
        result = paired_intervals(paired, args.bootstrap, 20261126 + seed)
        result['seed'] = seed
        result['common_images'] = len(paired)
        result['common_groups'] = paired.group_key.nunique()
        output.append(result)
    pd.concat(output).to_csv(args.output / 'paired_common_differences.csv', index=False)


if __name__ == '__main__':
    main()
