import numpy as np
import pandas as pd
from PIL import Image

from src.model import TRANSFORM
from src.publication_experiments import (bootstrap_intervals, condition_metrics,
                                         TENSOR, duplicate_audit, grade_sensitivity, phash, prepared,
                                         review_coverage)


def test_publication_statistics_and_review_coverage():
    rows = pd.DataFrame({
        'condition': ['original'] * 6,
        'binary_label': [0, 0, 0, 1, 1, 1],
        'diagnosis': [0, 0, 0, 1, 2, 2],
        'score': [.1, .7, .2, .8, .3, .9],
        'prediction': [0, 1, 0, 1, 0, 1],
        'review_required': [False, True, False, False, True, False],
    })
    summary, confusion = condition_metrics(rows)
    assert summary.iloc[0].precision == 2 / 3
    assert confusion['count'].tolist() == [2, 1, 1, 2]
    intervals = bootstrap_intervals(rows, resamples=2000, seed=19)
    assert len(intervals) == 6
    assert all(intervals.lower_95 <= intervals.estimate)
    assert all(intervals.upper_95 >= intervals.estimate)
    grades = grade_sensitivity(rows)
    assert grades.false_negatives.tolist() == [0, 1]
    coverage = review_coverage(rows).iloc[0]
    assert coverage.flagged_images == 2
    assert coverage.errors_captured == 2
    assert coverage.false_negatives_captured == 1
    assert coverage.residual_errors == 0


def test_duplicate_audit_and_fixed_degradation(tmp_path):
    images = tmp_path / 'images'; images.mkdir()
    rng = np.random.default_rng(5)
    original = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    original.save(images / 'a.png'); original.save(images / 'b.png')
    assert phash(original) == phash(Image.open(images / 'b.png'))
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(images / 'c.png')
    frames = {'train': pd.DataFrame({'id_code': ['a']}),
              'val': pd.DataFrame({'id_code': ['b']}),
              'test': pd.DataFrame({'id_code': ['c']})}
    result = duplicate_audit(frames, images, tmp_path)
    assert result['exact_cross_partition_pairs'] == 1
    assert len(pd.read_csv(tmp_path / 'exact_cross_partition.csv')) == 1
    plain = prepared(original, 'original')
    assert plain.shape == (224, 224, 3)
    assert np.array_equal(TENSOR(Image.fromarray(plain)).numpy(), TRANSFORM(original).numpy())
    assert not np.array_equal(plain, prepared(original, 'blur_severe'))
    assert prepared(original, 'brightness_low').mean() < plain.mean()
