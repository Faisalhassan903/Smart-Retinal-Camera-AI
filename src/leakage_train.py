"""Matched frozen-backbone ResNet-18 training with a shared ImageNet feature cache."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models

from .model import sha256
from .train import RetinalDataset, load_splits, metrics, validate_frame


def feature_cache(labels, images, output, batch_size=32, device='cpu'):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'features_PRIVATE.npz'
    if path.exists():
        cached = np.load(path)
        if len(cached['ids']) != len(labels) or set(cached['ids']) != set(labels.id_code):
            raise ValueError('Existing feature cache does not match the label file.')
        return path
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(RetinalDataset(labels, images), batch_size=batch_size, shuffle=False, num_workers=0)
    parts = []
    with torch.inference_mode():
        for number, (x, _) in enumerate(loader, 1):
            parts.append(model(x.to(device)).cpu().numpy())
            if number % 20 == 0:
                print(f'ImageNet features: {min(number * batch_size, len(labels))}/{len(labels)}', flush=True)
    np.savez_compressed(path, ids=labels.id_code.to_numpy(dtype=str), features=np.concatenate(parts))
    return path


def train_head(features, labels, frames, seed, epochs, batch_size, lr):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    feature_tensor = torch.from_numpy(features).float()
    target_tensor = torch.from_numpy((labels.diagnosis.to_numpy() > 0).astype(np.float32))
    locations = {image_id: i for i, image_id in enumerate(labels.id_code)}
    indices = {name: torch.tensor([locations[image_id] for image_id in frame.id_code])
               for name, frame in frames.items()}
    head = torch.nn.Linear(feature_tensor.shape[1], 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, history = float('inf'), None, None, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        head.train()
        order = indices['train'][torch.randperm(len(indices['train']), generator=generator)]
        loss_sum = 0.
        for batch in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(feature_tensor[batch]).squeeze(1), target_tensor[batch])
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(batch)
        head.eval()
        with torch.inference_mode():
            val_idx = indices['val']
            logits = head(feature_tensor[val_idx]).squeeze(1)
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor[val_idx]).item()
            scores = torch.sigmoid(logits).numpy()
        row = {'epoch': epoch, 'training_loss': loss_sum / len(order), 'validation_loss': val_loss,
               'epoch_seconds': time.perf_counter() - epoch_start,
               **{'val_' + key: value for key, value in metrics(target_tensor[val_idx].numpy(), scores).items()}}
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(head(feature_tensor).squeeze(1)).numpy()
    return head, pd.DataFrame(history), scores, best_epoch, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--original-splits', type=Path, required=True)
    parser.add_argument('--controlled-splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    if args.output.exists() and (args.output / 'completed.json').exists():
        raise FileExistsError('Completed run exists; use a new output directory.')
    labels = validate_frame(pd.read_csv(args.labels))
    labels = labels.sort_values('id_code').reset_index(drop=True)
    partitions = {'original': args.original_splits, 'controlled': args.controlled_splits}
    frames = {name: dict(zip(('train', 'val', 'test'), load_splits(args.labels, directory)))
              for name, directory in partitions.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cache = feature_cache(labels, args.images, args.output, args.batch_size)
    stored = np.load(cache)
    if not np.array_equal(stored['ids'], labels.id_code.to_numpy()):
        raise ValueError('Feature cache order mismatch.')
    features = stored['features']
    all_rows = []
    for seed in args.seeds:
        for name, split_frames in frames.items():
            run = args.output / f'{name}_seed{seed}'
            if run.exists():
                raise FileExistsError(run)
            run.mkdir()
            head, history, scores, best_epoch, duration = train_head(
                features, labels, split_frames, seed, args.epochs, args.batch_size, args.lr)
            # A full standard ResNet-18 checkpoint allows independent inference checks.
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            model.fc.load_state_dict(head.state_dict())
            torch.save(model.state_dict(), run / 'best.pth')
            history.to_csv(run / 'history.csv', index=False)
            prediction = labels[['id_code', 'diagnosis']].copy()
            prediction['binary_label'] = (prediction.diagnosis > 0).astype(int)
            prediction['score'] = scores
            prediction['prediction'] = (scores >= .5).astype(int)
            prediction.to_csv(run / 'all_scores_PRIVATE.csv', index=False)
            test = prediction[prediction.id_code.isin(split_frames['test'].id_code)]
            result = metrics(test.binary_label, test.score)
            result['precision'] = result['TP'] / (result['TP'] + result['FP']) if result['TP'] + result['FP'] else 0.
            result.update({'experiment': name, 'seed': seed, 'best_epoch': best_epoch,
                           'training_seconds': duration, 'checkpoint_sha256': sha256(run / 'best.pth'),
                           'test_manifest_sha256': sha256(partitions[name] / 'test.csv')})
            (run / 'config.json').write_text(json.dumps({
                'source_weights': 'torchvision ResNet18_Weights.IMAGENET1K_V1',
                'historical_checkpoint_used': False, 'feature_cache_sha256': sha256(cache),
                'head': 'linear 512-to-1', 'frozen_backbone': True, 'backbone_mode': 'eval',
                'optimizer': 'Adam', 'learning_rate': args.lr, 'loss': 'BCEWithLogitsLoss',
                'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': seed,
                'selection': 'minimum validation BCE', 'best_epoch': best_epoch,
                'training_seconds': duration, 'test_used_for_selection': False,
                'split_manifest_sha256': {part: sha256(partitions[name] / f'{part}.csv')
                                          for part in ('train', 'val', 'test')},
                'checkpoint_sha256': result['checkpoint_sha256']}, indent=2))
            all_rows.append(result)
            pd.DataFrame(all_rows).to_csv(args.output / 'test_metrics.csv', index=False)
            print(f'{name} seed={seed}: accuracy={result["accuracy"]:.4f}, AUC={result["roc_auc"]:.4f}', flush=True)
    (args.output / 'completed.json').write_text(json.dumps({'experiments': len(all_rows), 'elapsed_seconds': time.perf_counter() - started,
                                                             'feature_cache_sha256': sha256(cache)}, indent=2))


if __name__ == '__main__':
    main()
