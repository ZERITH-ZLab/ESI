"""Smoke-test real OpenPI batches without loading model weights or training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from data_processing.sampling.sampler import load_config
from data_processing.subtasks.io import read, require, write
from .loader import create_data_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Optional JSON report; use mounted storage')
    args = parser.parse_args()
    cfg = read(args.config)
    sampling_path = cfg.pop('sampling_config', None)
    started = time.monotonic()
    loader = create_data_loader(**cfg, sampling_config=load_config(sampling_path))
    # Deterministic probes ensure both real invalid tails and episode padding
    # are checked even if a small random batch happens not to contain them.
    probes = {}
    for i, row in enumerate(loader.dataset.rows):
        kind = ('episode_padding' if row['pad_length'] else
                'interior_masked' if row['valid_length'] < loader.dataset.horizon else 'full')
        key = f'{row["start_control"]}/{kind}'
        if key not in probes:
            sample = loader.dataset[i]
            value = loader.transformed[i]
            require(np.array_equal(sample['action_loss_mask'], value['action_loss_mask']), 'Mask changed')
            require(value['actions'].shape == (loader.dataset.horizon, loader.contract['action_dim']),
                    'Unexpected transformed action shape')
            probes[key] = dict(sample_id=int(sample['sample_id']), valid_steps=int(sample['action_loss_mask'].sum()))
    reports = []
    for i, batch in enumerate(loader):
        mask = np.asarray(batch.action_loss_mask)
        actions = np.asarray(batch.actions)
        require(mask.dtype == np.bool_ and mask.shape == actions.shape[:2], 'Invalid batch mask')
        require(mask.any(axis=1).all() and not (mask[:, 1:] & ~mask[:, :-1]).any(), 'Not a valid prefix')
        require(np.isfinite(actions).all(), 'Nonfinite batch')
        report = dict(batch=i, sample_ids=loader.last_sample_ids.tolist(),
            action_shape=list(actions.shape), valid_steps=mask.sum(axis=1).tolist(),
            state_shape=list(batch.observation.state.shape),
            images={k:list(v.shape) for k,v in batch.observation.images.items()},
            token_shape=list(batch.observation.tokenized_prompt.shape))
        reports.append(report)
        print(json.dumps(report), flush=True)
    result = dict(status='passed', windows=len(loader.dataset), contract=loader.contract,
                  probes=probes, batches=reports, elapsed_s=time.monotonic()-started,
                  loader_state=loader.state_dict(), training_performed=False)
    if args.output:
        write(args.output, result)
    print(json.dumps(dict(status='passed', batches=len(reports), windows=len(loader.dataset),
                         elapsed_s=result['elapsed_s'], output=str(args.output) if args.output else None)), flush=True)


if __name__ == '__main__':
    main()
