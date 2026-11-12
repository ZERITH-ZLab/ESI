"""Read indexed observations and dispatched commands without rewriting the dataset."""
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np

from data_processing.subtasks.io import read, require
from data_processing.training_index.build import checksum, validate
from data_processing.training_index.windows import window_positions

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


class IndexedDataset:
    """Zero-based row order is exactly ORDER BY sample_id, as in HierarchicalSampler.

    Caches only small state/action arrays; reads the three images at the start
    of each window. File hashes are checked before use. No SQLite handles or
    NPZ readers survive a call, so the dataset can be spawned into workers.
    """

    def __init__(self, index_root, *, cache_episodes=4, instruction_mode='task'):
        require(type(cache_episodes) is int and cache_episodes > 0, 'cache_episodes must be positive')
        self.index_root = Path(index_root).resolve()
        validate(self.index_root, check_sources=False)
        self.manifest = read(self.index_root/'manifest.json')
        cfg = self.manifest['config']
        require(cfg['supervision'] == 'continuous_valid_prefix' and
                cfg['interior_tail'] == 'real_actions' and cfg['episode_padding'] == 'repeat_last' and
                cfg['instruction'] == 'episode_task' and cfg['action_label'] == 'executed_command',
                'Unsupported training index contract')
        self.horizon = cfg['horizon']
        with closing(sqlite3.connect(f'{(self.index_root/"index.sqlite").as_uri()}?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            self.rows = [dict(r) for r in db.execute('SELECT * FROM training_samples ORDER BY sample_id')]
            self.episodes = {r['episode_key']: dict(r) for r in db.execute('SELECT * FROM episodes')}
            self.hashes = dict(db.execute('SELECT path,sha256 FROM sources'))
        require(self.rows, 'No training windows')
        self.sample_ids = np.asarray([r['sample_id'] for r in self.rows], dtype=np.int64)
        self.cache_episodes = cache_episodes
        self._cache = OrderedDict()
        from .conditioning import Conditioning
        self.conditioning = Conditioning(self, instruction_mode)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return dict(self.__dict__, _cache=OrderedDict())

    def _arrays(self, key):
        if key not in self._cache:
            ep = self.episodes[key]
            require(ep['action_array'] == 'command' and ep['state_array'] == 'state',
                    f'{key}: only pre-action state and executed command labels are supported')
            path = ep['steps_npz']
            require(path in self.hashes and checksum(path) == self.hashes[path], f'Source changed: {path}')
            with np.load(path, allow_pickle=False) as data:
                state, action = data['state'].copy(), data['command'].copy()
            require(state.shape == action.shape == (ep['total_steps'], 14), f'{key}: invalid array shape')
            require(np.isfinite(state).all() and np.isfinite(action).all(), f'{key}: nonfinite state/action')
            self._cache[key] = (state, action)
            if len(self._cache) > self.cache_episodes:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def __getitem__(self, index):
        require(isinstance(index, (int, np.integer)) and 0 <= index < len(self), 'Invalid dataset row index')
        row = self.rows[index]
        ep = self.episodes[row['episode_key']]
        t, n = row['start_step'], ep['total_steps']
        positions, mask = window_positions(t, n, row['valid_length'], self.horizon)
        require(row['valid_length'] == min(self.horizon, row['valid_end_step']-t) and
                row['real_length'] == min(self.horizon, n-t) and
                row['pad_length'] == max(0, t+self.horizon-n), 'Inconsistent indexed window')
        states, commands = self._arrays(row['episode_key'])
        archive = Path(ep['source_archive']).resolve()
        path = (archive/row['observation_relpath']).resolve()
        require(path.is_relative_to(archive), 'Observation must be inside source archive')
        require(checksum(path) == row['observation_sha256'], f'Observation changed: {path}')
        with np.load(path, allow_pickle=False) as obs:
            require(obs['states'].shape == (14,) and np.allclose(obs['states'], states[t], atol=1e-6, rtol=0),
                    f'{path}: observation and pre-action state disagree')
            require(str(obs['instruction'].item()) == ep['task_instruction'], f'{path}: task instruction changed')
            sample = {'observation.state': states[t].astype(np.float32, copy=True),
                      'action': commands[positions].astype(np.float32, copy=True),
                      'prompt': self.conditioning.prompt(self, index)}
            for camera in CAMERAS:
                require(camera in obs, f'{path}: missing camera {camera}')
                image = obs[camera]
                require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3,
                        f'{path}: {camera} must be uint8 HWC RGB')
                # The checkpoint's AlohaInputs expects CHW, then converts to HWC.
                sample[f'observation.images.{camera}'] = np.ascontiguousarray(image.transpose(2, 0, 1))
        sample['action_loss_mask'] = np.asarray(mask, dtype=bool)
        sample['sample_id'] = np.int64(row['sample_id'])
        return sample


class TransformedDataset:
    """Keep mask/identity outside official repack/Aloha transforms that drop extras."""

    def __init__(self, dataset, data_config):
        from openpi import transforms
        require(data_config.norm_stats is not None, 'Checkpoint normalization statistics are required')
        self.dataset = dataset
        self.transform = transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw = self.dataset[index]
        mask, sample_id = raw.pop('action_loss_mask'), raw.pop('sample_id')
        value = self.transform(raw)
        require(value['actions'].shape[0] == len(mask) and np.isfinite(value['actions']).all(),
                'Transformed actions must preserve the time axis and be finite')
        return dict(value, action_loss_mask=mask, sample_id=sample_id)

