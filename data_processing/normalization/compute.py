"""Exact quantiles; trim only statistical observations, never training labels."""
from pathlib import Path
import json

import numpy as np

from data_processing.openpi_loader.dataset import IndexedDataset, CAMERAS
from data_processing.subtasks.io import read, require, write
from data_processing.training_index.build import checksum
from . import VERSION


def robust_statistics(values):
    """Original q01/q99 are robust bounds; mean/std use the central 98%.

    Trimming is per dimension, not per action vector. Do not trim and then
    compute another q01/q99 (which would unintentionally trim twice).
    No histogram binning across a huge min/max outlier range.
    """
    require(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] == 14,
            'Expected at least two 14-dimensional vectors')
    require(np.isfinite(values).all(), 'Nonfinite data: repair source rather than silently drop samples')
    stats = {name: [] for name in ('mean','std','q01','q99')}
    report = []
    for dim in range(14):
        v = values[:,dim].astype(np.float64)
        q001, lo, median, hi, q999 = np.quantile(v, [.001,.01,.5,.99,.999], method='linear')
        # A collapsed scale can amplify numerical noise dramatically. Require
        # an explicit follow-up choice rather than inventing a new scale.
        require(hi-lo > 1e-6, f'Dimension {dim}: q01/q99 scale collapsed; explicit scale policy required')
        keep = (v >= lo) & (v <= hi)
        central = v[keep]
        mean, std = float(central.mean()), float(central.std())
        for key, value in dict(mean=mean,std=std,q01=float(lo),q99=float(hi)).items():
            stats[key].append(value)
        report.append(dict(dimension=dim,count=len(v),min=float(v.min()),max=float(v.max()),
            q001=float(q001),q01=float(lo),median=float(median),q99=float(hi),q999=float(q999),
            excluded_low=int((v<lo).sum()),excluded_high=int((v>hi).sum()),
            retained=int(keep.sum()),raw_mean=float(v.mean()),raw_std=float(v.std()),
            trimmed_mean=mean,trimmed_std=std))
    return stats, report


def collect_vectors(dataset, transform):
    """One vote per indexed start; one vote per valid (start,offset) action.

    Each action is transformed relative to THIS window's initial state before
    masking. Masked real tails and repeated episode padding do not enter stats.
    """
    states = np.empty((len(dataset),14), dtype=np.float32)
    actions = np.empty((sum(r['valid_length'] for r in dataset.rows),14),dtype=np.float32)
    images = {f'observation.images.{c}':np.zeros((3,1,1),np.uint8) for c in CAMERAS}
    offset = 0
    for i, row in enumerate(dataset.rows):
        state, command = dataset._arrays(row['episode_key'])
        t, length = row['start_step'], row['valid_length']
        # The selected prefix is all real, supervised actions; no padding.
        require(length == min(dataset.horizon,row['valid_end_step']-t) and
                t+length <= len(command), 'Invalid indexed prefix')
        transformed = transform(dict(images, **{'observation.state':state[t].astype(np.float32,copy=True),
            'action':command[t:t+length].astype(np.float32,copy=True),
            'prompt':dataset.episodes[row['episode_key']]['task_instruction']}))
        require(transformed['state'].shape == (14,) and transformed['actions'].shape == (length,14),
                'Unexpected pre-normalization transform shape')
        states[i] = transformed['state']
        actions[offset:offset+length] = transformed['actions']
        offset += length
    return states, actions


def validate_statistics(directory, index_sha256, model_config, horizon):
    directory = Path(directory).resolve()
    manifest = read(directory/'manifest.json')
    require(manifest['version'] == VERSION and manifest['status'] == 'completed', 'Incomplete statistics')
    require(manifest['index_sha256'] == index_sha256 and manifest['model_config'] == model_config and
            manifest['horizon'] == horizon, 'Statistics belong to a different training index/model/horizon')
    for name, expected in manifest['files_sha256'].items():
        require(checksum(directory/name) == expected, f'Normalization artifact changed: {name}')
    return manifest


def compute(index_root, base_checkpoint, output):
    from openpi import transforms
    from hybrid_rollout.robodojo.pi05_server.checkpoint import data_contract
    output = Path(output).resolve()
    require(not output.exists(), 'Output already exists; use a new normalization version')
    dataset = IndexedDataset(index_root)
    cfg, dc = data_contract(base_checkpoint)
    require(dc.use_quantile_norm, 'Expected pi05 quantile normalization')
    require(cfg.model.action_horizon == dataset.horizon and
        {json.loads(e['metadata_json'])['policy_config'] for e in dataset.episodes.values()} == {cfg.name},
        'Training index/model configuration differs')
    transform = transforms.compose([*dc.repack_transforms.inputs,*dc.data_transforms.inputs])
    print(json.dumps(dict(event='normalization_start',windows=len(dataset),episodes=len(dataset.episodes))),flush=True)
    states, actions = collect_vectors(dataset,transform)
    norms, distributions = {}, {}
    for key, values in [('state',states),('actions',actions)]:
        norms[key], distributions[key] = robust_statistics(values)
    old = dc.norm_stats
    comparison = {k:dict(old_q01=old[k].q01.tolist(),old_q99=old[k].q99.tolist(),
        new_to_old_range_ratio=((np.asarray(norms[k]['q99'])-norms[k]['q01'])/
            (np.asarray(old[k].q99)-old[k].q01+1e-6)).tolist()) for k in norms}
    report = dict(episodes=len(dataset.episodes),windows=len(dataset),state_vectors=len(states),
        supervised_action_vectors=len(actions), distributions=distributions,checkpoint_comparison=comparison,
        weighting='uniform_window_starts; equal_valid_action_vectors; no_sampler_weights',
        trimming='per_dimension: below original q01 or above original q99 excluded from mean/std only',
        quantile_bounds='exact original q01/q99; no second trimming',
        training_samples_removed=0,training_actions_clipped=False,
        masked_tail_in_statistics=False,padding_in_statistics=False,
        split='all indexed training episodes; no separate validation/test split supplied')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'norm_stats.json',dict(norm_stats=norms))
    write(output/'report.json',report)
    manifest = dict(version=VERSION,status='completed',index_root=str(dataset.index_root),
        index_sha256=dataset.manifest['files_sha256']['index.sqlite'],model_config=cfg.name,horizon=dataset.horizon,
        base_checkpoint=str(Path(base_checkpoint).resolve()),asset_id=dc.asset_id,
        bounds=[.01,.99],algorithm='exact_linear_quantiles_and_per_dimension_trimmed_mean_std',
        files_sha256={name:checksum(output/name) for name in ('norm_stats.json','report.json')})
    write(output/'manifest.json',manifest)
    validate_statistics(output,manifest['index_sha256'],cfg.name,dataset.horizon)
    return dict(status='completed',output=str(output),windows=len(dataset),
                state_vectors=len(states),supervised_action_vectors=len(actions))
