"""Reuse OpenPI batching while retaining the temporal loss mask and sampled IDs."""
from copy import deepcopy
import dataclasses
from pathlib import Path
from typing import Any, NamedTuple
import json

import numpy as np

from data_processing.sampling.sampler import HierarchicalSampler
from data_processing.subtasks.io import digest, require
from data_processing.training_index.build import checksum
from . import VERSION
from .dataset import IndexedDataset, TransformedDataset


class TrainingBatch(NamedTuple):
    observation: Any
    actions: Any
    action_loss_mask: Any


class RowSampler:
    """Prefetch uses a private RNG, never advances the consumed sampler state."""

    def __init__(self, sampler, count):
        self.cdf = sampler._cdf.copy()
        self.rng_state = deepcopy(sampler.state_dict()['rng_state'])
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = deepcopy(self.rng_state)
        for _ in range(self.count):
            yield int(np.searchsorted(self.cdf, rng.random(), side='right'))


class ProcessRowSampler(RowSampler):
    """Draw globally, decode only this node's portion of every microbatch."""
    def __init__(self, sampler, count, batch_size, micro_batch_size, processes, rank):
        super().__init__(sampler, count)
        from data_processing.training.distributed import local_positions
        self.positions = local_positions(batch_size, micro_batch_size, processes, rank)
        self.batch_size = batch_size

    def __len__(self):
        return self.count // self.batch_size * len(self.positions)

    def __iter__(self):
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = deepcopy(self.rng_state)
        for _ in range(self.count // self.batch_size):
            rows = np.searchsorted(self.cdf, rng.random(self.batch_size), side='right')
            yield from rows[self.positions].tolist()


class IndexedOpenPILoader:
    """Finite batches; restarting iteration continues after the last yielded batch.

    state_dict represents yielded batches, not worker-prefetched rows. Save it
    alongside the optimizer checkpoint after that batch's successful update.
    This loader deliberately returns THREE fields: native two-field train_step
    must be adapted to use action_loss_mask before running training.
    """

    def __init__(self, dataset, sampler, data_config, *, batch_size, num_batches,
                 num_workers=0, sharding=None, contract=None, micro_batch_size=None):
        require(type(batch_size) is int and 0 < batch_size <= len(dataset), 'Invalid batch_size')
        require(type(num_batches) is int and num_batches > 0, 'num_batches must be positive')
        require(type(num_workers) is int and num_workers >= 0, 'Invalid num_workers')
        require(np.array_equal(dataset.sample_ids, sampler.sample_ids), 'Dataset/sampler row order differs')
        require(dataset.manifest['files_sha256']['index.sqlite'] == sampler.index_sha256,
                'Dataset/sampler index differs')
        self.dataset, self.sampler, self._data_config = dataset, sampler, data_config
        self.transformed = TransformedDataset(dataset, data_config)
        self.batch_size, self.num_batches, self.num_workers = batch_size, num_batches, num_workers
        self.sharding, self.contract = sharding, contract
        self.micro_batch_size = micro_batch_size
        self.last_sample_ids = None
        self._active = False
        code = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))}
        self.fingerprint = digest(dict(version=VERSION, sampler=sampler.fingerprint,
                                       batch_size=batch_size, micro_batch_size=micro_batch_size,
                                       contract=contract, code=code))

    def data_config(self):
        return self._data_config

    def state_dict(self):
        return dict(version=VERSION, fingerprint=self.fingerprint, sampler=self.sampler.state_dict())

    def load_state_dict(self, state):
        require(not self._active, 'Close the active iterator before restoring loader state')
        require(state['version'] == VERSION and state['fingerprint'] == self.fingerprint,
                'Loader checkpoint contract/index/config/code differs')
        self.sampler.load_state_dict(state['sampler'])

    def __iter__(self):
        import jax
        if jax.process_count() > 1:
            yield from self._distributed_iterator()
            return
        from openpi.models.model import Observation
        from openpi.training.data_loader import TorchDataLoader
        require(not self._active, 'Only one active loader iterator is supported')
        self._active = True
        try:
            rows = RowSampler(self.sampler, self.batch_size*self.num_batches)
            native = TorchDataLoader(self.transformed, self.batch_size, sampler=rows,
                num_batches=self.num_batches, num_workers=self.num_workers,
                sharding=self.sharding, seed=self.sampler.config['seed'])
            for batch in native:
                ids = np.asarray(batch['sample_id'])
                result = TrainingBatch(Observation.from_dict(batch), batch['actions'], batch['action_loss_mask'])
                # Commit RNG advancement only after successfully reading a batch.
                before = self.sampler.state_dict()
                expected = self.sampler.draw_sample_ids(self.batch_size)
                if not np.array_equal(ids, expected):
                    self.sampler.load_state_dict(before)
                    raise ValueError('OpenPI batch order differs from hierarchical sampler')
                self.last_sample_ids = ids.copy()
                yield result
        finally:
            self._active = False

    def _distributed_iterator(self):
        import jax
        import torch
        from openpi.models.model import Observation
        from openpi.training.data_loader import _collate_fn, _worker_init_fn
        require(not self._active, 'Only one active loader iterator is supported')
        require(self.micro_batch_size and self.sharding is not None,
                'Distributed loader requires accumulated sharding and global microbatch size')
        self._active = True
        iterator = None
        try:
            rows = ProcessRowSampler(self.sampler, self.batch_size*self.num_batches,
                self.batch_size, self.micro_batch_size, jax.process_count(), jax.process_index())
            local_size = self.batch_size // jax.process_count()
            accumulation = self.batch_size // self.micro_batch_size
            native = torch.utils.data.DataLoader(self.transformed, batch_size=local_size,
                sampler=rows, num_workers=self.num_workers, collate_fn=_collate_fn,
                worker_init_fn=_worker_init_fn, drop_last=True,
                multiprocessing_context='spawn' if self.num_workers else None,
                generator=torch.Generator().manual_seed(self.sampler.config['seed']))
            iterator = iter(native)
            for batch in iterator:
                require(np.asarray(batch['action_loss_mask']).any(axis=-1).all(), 'Empty supervision mask')
                before = self.sampler.state_dict()
                expected = self.sampler.draw_sample_ids(self.batch_size)
                try:
                    require(np.array_equal(batch['sample_id'], expected[rows.positions]),
                            'Distributed batch order differs from global sampler')
                    def upload(x):
                        x = np.asarray(x).reshape((accumulation, local_size//accumulation)+x.shape[1:])
                        return jax.make_array_from_process_local_data(self.sharding, x,
                            global_shape=(accumulation, self.micro_batch_size)+x.shape[2:])
                    batch = jax.tree.map(upload, batch)
                    result = TrainingBatch(Observation.from_dict(batch), batch['actions'], batch['action_loss_mask'])
                except BaseException:
                    self.sampler.load_state_dict(before)
                    raise
                self.last_sample_ids = expected.copy()
                yield result
        finally:
            if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
            self._active = False


def create_data_loader(index_root, base_checkpoint, *, sampling_config=None,
                       batch_size=8, num_batches=100, num_workers=0,
                       cache_episodes=4, sharding=None, norm_stats_dir=None, instruction_mode='task',
                       micro_batch_size=None):
    """Factory for the installed RoboDojo pi05 checkpoint's original transforms.

    Loads config/tokenizer/normalization, NOT model weights. Multi-node training
    supplies micro_batch_size and accumulated sharding. Parameters load separately.
    """
    from hybrid_rollout.robodojo.pi05_server.checkpoint import data_contract
    dataset = IndexedDataset(index_root, cache_episodes=cache_episodes, instruction_mode=instruction_mode)
    sampler = HierarchicalSampler(index_root, sampling_config)
    cfg, dc = data_contract(base_checkpoint)
    require(cfg.model.action_horizon == dataset.horizon, 'Model/index action horizon differs')
    configs = {json.loads(ep['metadata_json'])['policy_config'] for ep in dataset.episodes.values()}
    require(configs == {cfg.name}, f'Collected policy configuration differs: {configs}')
    require(cfg.model.pi05, 'This adapter requires the pi05 action model')
    norm = Path(base_checkpoint).resolve()/'assets'/dc.asset_id/'norm_stats.json'
    if norm_stats_dir is not None:
        from openpi.shared import normalize
        from data_processing.normalization.compute import validate_statistics
        validate_statistics(norm_stats_dir, dataset.manifest['files_sha256']['index.sqlite'],
                            cfg.name, dataset.horizon)
        dc = dataclasses.replace(dc, norm_stats=normalize.load(norm_stats_dir))
        norm = Path(norm_stats_dir).resolve()/'norm_stats.json'
    conditioning = dataset.conditioning.report(dataset, sampler)
    if instruction_mode == 'subtask':
        from .conditioning import prepare_text_transforms
        print(json.dumps(dict(event='checking_conditioning_text', windows=len(dataset))), flush=True)
        dc, conditioning['token_check'] = prepare_text_transforms(dataset, dc)
    print(json.dumps(dict(event='training_conditioning', **conditioning)), flush=True)
    contract = dict(model_config=cfg.name, action_dim=cfg.model.action_dim,
                    horizon=dataset.horizon, checkpoint=str(Path(base_checkpoint).resolve()),
                    norm_stats_path=str(norm),
                    norm_stats_sha256=checksum(norm), instruction=instruction_mode,
                    conditioning=conditioning,
                    supervision='continuous_valid_prefix', attention_mask='unchanged')
    return IndexedOpenPILoader(dataset, sampler, dc, batch_size=batch_size,
        num_batches=num_batches, num_workers=num_workers, sharding=sharding, contract=contract,
        micro_batch_size=micro_batch_size)
