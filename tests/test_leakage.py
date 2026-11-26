import pandas as pd

from src.leakage_splits import add_verified_near_groups, exact_groups, split_groups
from src.leakage_analysis import group_bootstrap


def test_confirmed_groups_stay_together(tmp_path):
    rows = []
    hashes = []
    for grade in range(5):
        for index in range(30):
            image_id = f'i{grade}_{index}'
            rows.append({'id_code': image_id, 'diagnosis': grade})
            hashes.append({'id_code': image_id, 'sha256': f'h{grade}_{index}'})
    # Exact match and individually reviewed near match are both grouped.
    hashes[1]['sha256'] = hashes[0]['sha256']
    pairs = tmp_path / 'verified.csv'
    pd.DataFrame([{'id_a': 'i1_0', 'id_b': 'i1_1', 'review_basis': 'visual_review_test'}]).to_csv(pairs, index=False)
    joined, accepted = add_verified_near_groups(exact_groups(pd.DataFrame(rows), pd.DataFrame(hashes)), pairs)
    assert accepted == 1
    splits, _ = split_groups(joined, seed=42, trials=5)
    assignment = {image_id: name for name, frame in splits.items() for image_id in frame.id_code}
    assert assignment['i0_0'] == assignment['i0_1']
    assert assignment['i1_0'] == assignment['i1_1']
    assert sum(map(len, splits.values())) == 150


def test_group_bootstrap_reports_all_metrics():
    frame = pd.DataFrame({'id_code': ['a', 'b', 'c', 'd'], 'binary_label': [0, 0, 1, 1],
                          'score': [.1, .2, .8, .9], 'prediction': [0, 0, 1, 1],
                          'condition': ['original'] * 4})
    groups = pd.Series({'a': 'g1', 'b': 'g1', 'c': 'g2', 'd': 'g3'})
    result = group_bootstrap(frame, groups, resamples=10, seed=3)
    assert set(result.metric) == {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc'}
    assert (result.estimate == 1).all()
