"""Create new research runs without modifying the historical baseline."""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .model import TRANSFORM, build_model, sha256


def validate_frame(frame):
    if not {'id_code', 'diagnosis'}.issubset(frame.columns):
        raise ValueError('CSV requires id_code and diagnosis columns.')
    if frame.empty or frame[['id_code', 'diagnosis']].isna().any().any() or not frame.id_code.is_unique:
        raise ValueError('Empty rows, missing labels/IDs or duplicate IDs are not allowed.')
    if not frame.diagnosis.isin(range(5)).all():
        raise ValueError('Diagnosis must be an integer grade from 0 to 4.')
    if not frame.id_code.astype(str).str.fullmatch(r'[A-Za-z0-9_-]+').all():
        raise ValueError('Image IDs must be filenames without path separators.')
    frame = frame.copy()
    derived = (frame.diagnosis > 0).astype(int)
    if 'binary_label' in frame and not np.array_equal(frame.binary_label, derived):
        raise ValueError('binary_label conflicts with diagnosis > 0.')
    frame['binary_label'] = derived
    return frame


def load_splits(csv_path, split_dir=None, seed=42):
    full = validate_frame(pd.read_csv(csv_path))
    if split_dir:
        frames = [validate_frame(pd.read_csv(Path(split_dir) / f'{name}.csv')) for name in ('train', 'val', 'test')]
    else:
        train, temp = train_test_split(full, test_size=0.30, stratify=full.diagnosis, random_state=seed)
        val, test = train_test_split(temp, test_size=0.50, stratify=temp.diagnosis, random_state=seed)
        frames = [train, val, test]
    ids = [set(frame.id_code) for frame in frames]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError('Overlapping image IDs across splits.')
    if set.union(*ids) != set(full.id_code):
        raise ValueError('Split manifests must cover exactly the labelled CSV.')
    labels = full.set_index('id_code').diagnosis
    for frame in frames:
        if not np.array_equal(frame.diagnosis.to_numpy(), labels.loc[frame.id_code].to_numpy()):
            raise ValueError('Split labels differ from the labelled CSV.')
    return frames


class RetinalDataset(Dataset):
    def __init__(self, frame, image_dir):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.image_dir / f'{row.id_code}.png') as image:
            x = TRANSFORM(image.convert('RGB'))
        return x, torch.tensor(row.binary_label, dtype=torch.float32)


def metrics(targets, scores):
    targets = np.asarray(targets, dtype=int)
    scores = np.asarray(scores)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {'n': len(targets), 'threshold': 0.5, 'accuracy': accuracy_score(targets, predictions),
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'roc_auc': float(roc_auc_score(targets, scores)) if len(np.unique(targets)) == 2 else None,
            'f1': float(f1_score(targets, predictions, zero_division=0)),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def evaluate(model, loader, device):
    model.eval()
    loss_sum, targets, scores = 0.0, [], []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='sum').item()
            targets.extend(y.cpu().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return loss_sum / len(targets), targets, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--splits', type=Path, help='Use preserved train/val/test manifests.')
    parser.add_argument('--output', type=Path, default=Path('runs'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--fine-tune', action='store_true', help='Train the backbone too; this is a new experiment.')
    parser.add_argument('--initial-checkpoint', type=Path, help='Explicit warm start; never modifies the source checkpoint.')
    parser.add_argument('--random-init', action='store_true', help='For software smoke tests only, not transfer-learning results.')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    if args.random_init and args.initial_checkpoint:
        parser.error('Choose either random initialisation or an initial checkpoint')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    train, val, test = load_splits(args.csv, args.splits, args.seed)
    for frame in (train, val, test):
        for image_id in frame.id_code:
            if not (args.images / f'{image_id}.png').is_file():
                raise FileNotFoundError(args.images / f'{image_id}.png')
    run = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'splits').mkdir()
    for name, frame in zip(('train', 'val', 'test'), (train, val, test)):
        frame.to_csv(run / 'splits' / f'{name}.csv', index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'device_used': str(device), 'python': sys.version, 'torch': str(torch.__version__),
                   'torchvision': version('torchvision'), 'numpy': np.__version__,
                   'csv_sha256': sha256(args.csv), 'selection': 'minimum validation BCE',
                   'split_sha256': {n: sha256(run / 'splits' / f'{n}.csv') for n in ('train', 'val', 'test')},
                   'initial_checkpoint_sha256': sha256(args.initial_checkpoint) if args.initial_checkpoint else None,
                   'test_used_for_selection': False, 'clinical_validation': False})
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(RetinalDataset(train, args.images), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(RetinalDataset(val, args.images), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(pretrained=not args.random_init and not args.initial_checkpoint, fine_tune=args.fine_tune)
    if args.initial_checkpoint:
        model.load_state_dict(torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True))
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    history, best = [], float('inf')
    run_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if args.fine_tune:
            model.train()
        else:
            model.eval(); model.fc.train()  # Preserve backbone BatchNorm statistics.
        loss_sum, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(y); count += len(y)
        val_loss, targets, scores = evaluate(model, val_loader, device)
        row = {'epoch': epoch, 'training_loss': loss_sum / count, 'validation_loss': val_loss,
               'epoch_seconds': time.perf_counter() - epoch_started,
               **{'val_' + k: v for k, v in metrics(targets, scores).items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        if val_loss < best:
            best = val_loss
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run / 'best.pth')
            config['best_epoch'] = epoch
        print(json.dumps(row), flush=True)
    config['checkpoint_sha256'] = sha256(run / 'best.pth')
    config['total_training_seconds'] = time.perf_counter() - run_started
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Completed: {run}\nTest set not evaluated. Use python -m src.evaluate after model selection is final.')


if __name__ == '__main__':
    main()
