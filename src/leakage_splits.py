"""Create grade-stratified splits that keep confirmed duplicate groups together."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .model import sha256
from .train import load_splits, validate_frame


def exact_groups(labels, hashes):
    labels = validate_frame(labels)
    hashes = hashes[['id_code', 'sha256']].copy()
    if len(hashes) != len(labels) or not hashes.id_code.is_unique or set(hashes.id_code) != set(labels.id_code):
        raise ValueError('Hashes must cover every labelled image exactly once.')
    return labels.merge(hashes, on='id_code', validate='one_to_one')


def add_verified_near_groups(joined, verified_pairs=None):
    """Union byte-identical groups with individually visually verified pairs."""
    parent = {digest: digest for digest in joined.sha256.unique()}
    def root(digest):
        while parent[digest] != digest:
            parent[digest] = parent[parent[digest]]
            digest = parent[digest]
        return digest
    accepted = 0
    if verified_pairs is not None:
        pairs = pd.read_csv(verified_pairs)
        if not {'id_a', 'id_b', 'review_basis'}.issubset(pairs.columns):
            raise ValueError('Verified near pairs require IDs and review basis.')
        by_id = joined.set_index('id_code').sha256
        for pair in pairs.itertuples(index=False):
            if pair.id_a not in by_id or pair.id_b not in by_id or not str(pair.review_basis).startswith('visual_review_'):
                raise ValueError('Near pairs must have labelled images and documented visual review.')
            a, b = root(by_id.loc[pair.id_a]), root(by_id.loc[pair.id_b])
            if a != b:
                parent[max(a, b)] = min(a, b)
            accepted += 1
    joined = joined.copy()
    joined['group_key'] = joined.sha256.map(root)
    return joined, accepted


def split_groups(joined, seed=42, trials=1000):
    groups = joined.groupby('group_key', sort=True).agg(
        grade=('diagnosis', lambda x: int(x.mode().min())), size=('id_code', 'size')).reset_index()
    target = np.array([.70, .15, .15])[:, None] * np.bincount(joined.diagnosis, minlength=5)[None, :]
    best, best_loss = None, float('inf')
    for offset in range(trials):
        state = seed + offset
        try:
            training, remaining = train_test_split(groups, test_size=.30, stratify=groups.grade, random_state=state)
            validation, testing = train_test_split(remaining, test_size=.50, stratify=remaining.grade, random_state=state)
        except ValueError:
            # The caller must not silently fall back to image-level splitting.
            raise ValueError('Too few groups of a grade for grade-stratified group splitting.') from None
        frames = [training, validation, testing]
        actual = np.stack([np.bincount(joined.loc[joined.group_key.isin(part.group_key), 'diagnosis'], minlength=5)
                           for part in frames])
        loss = float(np.square((actual - target) / np.maximum(target, 1)).mean())
        if loss < best_loss:
            best, best_loss = frames, loss
    result = {}
    for name, part in zip(('train', 'val', 'test'), best):
        result[name] = joined.loc[joined.group_key.isin(part.group_key),
                                  ['id_code', 'diagnosis', 'binary_label']].sort_values('id_code').reset_index(drop=True)
    assert sum(map(len, result.values())) == len(joined)
    ids = {name: set(frame.id_code) for name, frame in result.items()}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    hashes_by_id = joined.set_index('id_code').group_key
    group_sets = [set(hashes_by_id.loc[list(ids[name])]) for name in result]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    return result, best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--hashes', type=Path, required=True)
    parser.add_argument('--original-splits', type=Path, default=Path('splits'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--verified-near-pairs', type=Path)
    args = parser.parse_args()
    load_splits(args.labels, args.original_splits)
    joined, reviewed_pairs = add_verified_near_groups(
        exact_groups(pd.read_csv(args.labels), pd.read_csv(args.hashes)), args.verified_near_pairs)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    splits, loss = split_groups(joined, args.seed, args.trials)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(args.output / f'{name}.csv', index=False)
    joined[['id_code', 'sha256', 'group_key']].to_csv(args.output / 'group_keys_PRIVATE.csv', index=False)
    load_splits(args.labels, args.output)
    all_groups = joined.groupby('sha256').agg(images=('id_code', 'size'),
                                               grade_labels=('diagnosis', 'nunique'),
                                               binary_labels=('binary_label', 'nunique'))
    original = {name: set(pd.read_csv(args.original_splits / f'{name}.csv').id_code)
                for name in ('train', 'val', 'test')}
    original_assignment = {image_id: name for name, ids in original.items() for image_id in ids}
    exact_members = joined[joined.sha256.isin(all_groups[all_groups.images > 1].index)].copy()
    exact_members['original_partition'] = exact_members.id_code.map(original_assignment)
    exact_members['controlled_partition'] = exact_members.id_code.map(
        {image_id: name for name, frame in splits.items() for image_id in frame.id_code})
    exact_members[['sha256', 'id_code', 'diagnosis', 'binary_label', 'original_partition',
                   'controlled_partition']].to_csv(args.output / 'exact_group_members_PRIVATE.csv', index=False)
    cross = exact_members.groupby('sha256').original_partition.nunique()
    partition_pattern = exact_members.groupby('sha256').original_partition.agg(
        lambda values: '+'.join(sorted(set(values))))
    audit = {'images': len(joined), 'exact_content_groups': len(all_groups),
             'groups_with_multiple_images': int((all_groups.images > 1).sum()),
             'images_in_multiple_image_groups': int(all_groups.loc[all_groups.images > 1, 'images'].sum()),
             'grade_discordant_groups': int((all_groups.grade_labels > 1).sum()),
             'binary_discordant_groups': int((all_groups.binary_labels > 1).sum()),
             'original_cross_partition_exact_groups': int((cross > 1).sum()),
             'original_duplicate_group_partition_patterns': partition_pattern.value_counts().to_dict(),
             'partition_images': {name: len(frame) for name, frame in splits.items()},
             'partition_grade_counts': {name: frame.diagnosis.value_counts().reindex(range(5), fill_value=0).to_dict()
                                        for name, frame in splits.items()},
             'original_test_overlap': len(original['test'] & set(splits['test'].id_code)),
             'stratification_loss': loss, 'grouping': 'SHA-256 exact content plus individually visually verified near pairs',
             'visually_verified_near_pairs': reviewed_pairs, 'total_split_groups': int(joined.group_key.nunique()),
             'patient_level_separation_verified': False, 'seed': args.seed, 'trials': args.trials,
             'original_manifest_sha256': {name: sha256(args.original_splits / f'{name}.csv')
                                          for name in ('train', 'val', 'test')},
             'new_manifest_sha256': {name: sha256(args.output / f'{name}.csv')
                                     for name in ('train', 'val', 'test')}}
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
