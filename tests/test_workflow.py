import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.model import Predictor, build_model, review_flags
from src.train import load_splits, metrics


def test_metrics_match_historical_counts():
    labels = [0] * 271 + [1] * 279
    scores = [0.1] * 258 + [0.9] * 13 + [0.1] * 20 + [0.9] * 259
    result = metrics(labels, scores)
    assert result['accuracy'] == 0.94
    assert [result[k] for k in ('TN', 'FP', 'FN', 'TP')] == [258, 13, 20, 259]
    assert result['sensitivity'] == pytest.approx(259 / 279)


def test_review_flag_boundaries():
    quality = {'blur_score': 10, 'brightness': 40, 'contrast': 25}
    assert review_flags(quality, 0.39) == ([], False)
    assert review_flags(quality, 0.4) == ([], True)
    assert review_flags(quality, 0.6) == ([], True)
    assert review_flags({**quality, 'brightness': 39}, 0.8) == (['Low brightness'], False)


def test_overlap_rejected(tmp_path):
    frame = pd.DataFrame({'id_code': [f'x{i}' for i in range(20)], 'diagnosis': [0, 1] * 10})
    frame.to_csv(tmp_path / 'all.csv', index=False)
    for name in ('train', 'val', 'test'):
        frame.to_csv(tmp_path / f'{name}.csv', index=False)
    with pytest.raises(ValueError, match='Overlapping'):
        load_splits(tmp_path / 'all.csv', tmp_path)


def test_checkpoint_and_gradcam_are_repeatable(tmp_path):
    torch.manual_seed(42)
    model = build_model()
    path = tmp_path / 'fixture.pth'
    torch.save(model.state_dict(), path)
    predictor = Predictor(path)
    image = Image.fromarray(np.random.default_rng(42).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    before = predictor.predict(image)
    overlay, _ = predictor.gradcam(image)
    after = predictor.predict(image)
    assert before == after
    assert overlay.shape == (224, 224, 3)
    assert not predictor.model.layer4[-1]._forward_hooks


def test_app_startup():
    app = AppTest.from_file(Path('app.py').resolve(), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'Smart Retinal Camera AI'
    assert any('Not clinically validated' in warning.value for warning in app.warning)


def test_app_upload_and_invalid_file():
    if not Path('models/resnet18_epoch1.pth').exists():
        pytest.skip('Private baseline checkpoint is not distributed.')
    data = io.BytesIO()
    Image.fromarray(np.random.default_rng(7).integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(data, format='PNG')
    app = AppTest.from_file(Path('app.py').resolve(), default_timeout=30).run()
    app.file_uploader[0].set_value(('synthetic_software_test.png', data.getvalue(), 'image/png')).run()
    assert not app.exception and not app.error
    assert any(metric.label == 'DR score' for metric in app.metric)
    app.file_uploader[0].set_value(('invalid.png', b'not an image', 'image/png')).run()
    assert not app.exception
    assert any('Unable to analyse' in error.value for error in app.error)


def test_notebook_sources_preserved():
    path = Path('private_archive/Smart_Retinal_Camera_AI_documented.ipynb')
    if not path.exists():
        pytest.skip('Private original is intentionally not distributed.')
    original = json.loads(path.read_text())
    clean = json.loads(Path('notebooks/full_research_code.ipynb').read_text())
    assert [c['source'] for c in clean['cells']] == [c['source'] for c in original['cells']]
    assert all(not c.get('outputs') for c in clean['cells'])
