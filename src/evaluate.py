"""Evaluate a selected checkpoint; save new results without replacing historical evidence."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from .model import Predictor
from .train import RetinalDataset, evaluate, metrics, validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', default=Path('runs/evaluations'), type=Path)
    args = parser.parse_args()
    frame = validate_frame(pd.read_csv(args.manifest))
    predictor = Predictor(args.checkpoint)
    loader = DataLoader(RetinalDataset(frame, args.images), batch_size=32, shuffle=False)
    loss, labels, scores = evaluate(predictor.model, loader, predictor.device)
    result = metrics(labels, scores)
    result.update({'loss': loss, 'checkpoint_sha256': predictor.fingerprint, 'manifest': str(args.manifest)})
    out = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    frame['score'] = scores
    frame['prediction'] = (frame.score >= 0.5).astype(int)
    frame.to_csv(out / 'predictions.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); print(f'Saved: {out}')


if __name__ == '__main__':
    main()
