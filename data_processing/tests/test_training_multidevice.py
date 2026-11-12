"""Run in a fresh process with XLA_FLAGS=--xla_force_host_platform_device_count=8.

CPU devices exercise real JAX partitioning/collectives, not GPU performance.
"""
import functools
import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest

pytest.importorskip('openpi')
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from openpi.training import sharding
from openpi.training.utils import TrainState

from data_processing.training import checkpoints,runner
from data_processing.training.config import load_config
from data_processing.training.parallel import setup,batch_preparer
from data_processing.training.step import train_step
from data_processing.openpi_loader.loader import TrainingBatch,IndexedOpenPILoader
from data_processing.openpi_loader.dataset import IndexedDataset
from data_processing.sampling.sampler import HierarchicalSampler
from data_processing.tests.test_training import TinyLoader
from data_processing.tests.test_openpi_loader import loader_index,openpi_config
from data_processing.training_index.build import checksum
from data_processing.subtasks.io import read,write

pytestmark=pytest.mark.skipif(jax.device_count()!=8,reason='Requires 8 JAX devices (virtual CPUs supported)')


class MatrixModel(nnx.Module):
    def __init__(self):
        # 4 MiB: native OpenPI actually shards this parameter and its Adam states.
        self.weight=nnx.Param(jnp.full((1024,1024),.001,jnp.float32))

    def compute_loss(self,rng,observation,actions,train):
        del rng,train
        observation=sharding.activation_sharding_constraint(observation)
        prediction=(observation@self.weight.value).mean(-1)[:,None,None]
        return ((prediction-actions)**2).mean(-1)


def config():
    cfg=load_config(Path(__file__).parents[1]/'training/config.8gpu_smoke.json')
    cfg['runtime']['require_gpu']=False
    return cfg


def batch():
    return TrainingBatch(jnp.arange(16,dtype=jnp.float32)[:,None]*jnp.ones((16,1024))*.01,
        jnp.arange(16*50,dtype=jnp.float32).reshape(16,50,1)*.001,
        jnp.arange(50)[None,:]<jnp.tile(jnp.array([15,25,40,50]),4)[:,None])


def assert_trees_close(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-7)


@pytest.mark.parametrize('fsdp_devices',[1,4,8])
def test_partitioned_update_matches_unsharded_and_restores(loader_index,tmp_path,fsdp_devices):
    cfg=config();cfg['parallelism']['fsdp_devices']=fsdp_devices
    mesh,rep,data,accum=setup(cfg)
    model=MatrixModel();params=nnx.state(model);tx=optax.adam(1e-3)
    state=TrainState(step=0,params=params,model_def=nnx.graphdef(model),tx=tx,
                     opt_state=tx.init(params),ema_decay=None,ema_params=None)
    state_shape=jax.eval_shape(lambda:state)
    placements=sharding.fsdp_sharding(state_shape,mesh)
    distributed=jax.device_put(state,placements)
    matrix=distributed.params['weight'].value
    assert matrix.addressable_shards[0].data.size==matrix.size//fsdp_devices
    adam_matrices=[x for x in jax.tree.leaves(distributed.opt_state) if x.shape==(1024,1024)]
    assert len(adam_matrices)==2
    assert all(x.addressable_shards[0].data.size==x.size//fsdp_devices for x in adam_matrices)
    prepare=batch_preparer(16,8,data,accum)
    batches=prepare(jax.device_put(batch(),data))
    expected=jax.tree.map(lambda x:x.reshape((2,8)+x.shape[1:]),batch())
    assert_trees_close(batches,expected)
    # Every device gets both accumulation steps and one sample in each microbatch.
    assert all(s.data.shape==(2,1,50,1) for s in batches.actions.addressable_shards)
    for shard in batches.actions.addressable_shards:
        np.testing.assert_array_equal(shard.data,np.asarray(expected.actions)[shard.index])
    step_cfg=SimpleNamespace(trainable_filter=nnx.Param);rng=jax.random.key(7)
    reference=jax.jit(functools.partial(train_step,step_cfg))
    reference_state,reference_info=reference(rng,state,expected)
    pstep=jax.jit(functools.partial(train_step,step_cfg),
        in_shardings=(rep,placements,accum),out_shardings=(placements,rep))
    with sharding.set_mesh(mesh):
        updated,info=pstep(rng,distributed,batches)
    assert_trees_close(updated,reference_state)
    assert_trees_close(info,reference_info)
    loader=TinyLoader(loader_index[0]);loader.sampler.draw_indices(16)
    with checkpoints.manager(tmp_path/'checkpoints',2) as manager:
        checkpoints.save(manager,updated,loader,fingerprint='multi',train_rng_data=[0,7])
        restored_loader=TinyLoader(loader_index[0])
        restored,_=checkpoints.restore(manager,state_shape,restored_loader,fingerprint='multi',
                                       target_sharding=placements)
        assert_trees_close(restored,updated)
        for array,placement in zip(jax.tree.leaves(restored),jax.tree.leaves(placements),strict=True):
            assert array.sharding.is_equivalent_to(placement,array.ndim)
        np.testing.assert_array_equal(loader.sampler.draw_indices(16),restored_loader.sampler.draw_indices(16))
        with sharding.set_mesh(mesh):
            continued,_=pstep(rng,updated,batches)
            resumed,_=pstep(rng,restored,batches)
        assert_trees_close(continued,resumed)


def test_eight_device_runner_stop_resume_native_initialization(loader_index,tmp_path,monkeypatch):
    import dataclasses
    import orbax.checkpoint as ocp
    from openpi.training import config as native_config
    from hybrid_rollout.robodojo.pi05_server import checkpoint as base_contract
    cfg=config();cfg['output']=str(tmp_path/'run');cfg['base_checkpoint']=str(tmp_path/'base')
    cfg['training'].update(num_steps=2,num_workers=0)
    cfg['optimizer']['warmup_steps']=0
    base=Path(cfg['base_checkpoint']);base.mkdir()
    with ocp.PyTreeCheckpointer() as cp:
        cp.save(str(base/'params'),{'params':nnx.state(MatrixModel()).to_pure_dict()})
    native_cfg=dataclasses.replace(native_config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0'),
                                  model=SimpleNamespace(create=lambda rng:MatrixModel()))
    monkeypatch.setattr(base_contract,'data_contract',lambda _: (native_cfg,None))
    monkeypatch.setattr(base_contract,'checkpoint_identity',lambda _:dict(checkpoint_sha256='matrix-fixture'))
    class BatchLoader(TinyLoader):
        fingerprint='fixture'
        def __iter__(self):
            for _ in range(2):
                self.last_sample_ids=self.sampler.draw_sample_ids(16)
                yield jax.device_put(batch(),self.sharding)
    def make_loader(*args,sharding,**kwargs):
        loader=BatchLoader(loader_index[0]);loader.sharding=sharding
        return loader
    monkeypatch.setattr(runner,'make_loader',make_loader)
    assert runner.run(cfg,stop_after=1)['step']==1
    assert runner.run(cfg,resume=True,stop_after=1)['step']==2
    records=[json.loads(line) for line in (Path(cfg['output'])/'metrics.jsonl').read_text().splitlines()]
    sessions=[r for r in records if r.get('event')=='session_start']
    assert [r['start_step'] for r in sessions]==[0,1]
    assert all(r['num_devices']==8 and r['per_device_micro_batch_size']==1 and
               r['accumulation_steps']==2 for r in sessions)
    steps=[r for r in records if 'step' in r]
    assert [r['step'] for r in steps]==[1,2]
    assert all(r['valid_steps']==32.5 and r['update_norm']>0 for r in steps)
    sampler=HierarchicalSampler(loader_index[0])
    assert [i for r in steps for i in r['sample_ids']]==sampler.draw_sample_ids(32).tolist()


def test_indexed_loader_eight_device_order_masks_and_resume(loader_index):
    # Expand the tiny index without creating extra raw files, to allow batch 16.
    index=loader_index[0]
    with sqlite3.connect(index/'index.sqlite') as db:
        rows=db.execute('SELECT * FROM samples').fetchall()
        episode=db.execute('SELECT * FROM episodes').fetchone()
        steps=db.execute('SELECT * FROM steps').fetchall()
        for i in range(5):
            key=f'copy_{i}'
            db.execute(f'INSERT INTO episodes VALUES ({",".join("?" for _ in episode)})',
                       (key,key,*episode[2:]))
            for step in steps:
                db.execute('INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(key,*step[1:]))
            for j,row in enumerate(rows):
                db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)',(100+3*i+j,key,*row[2:]))
    manifest=read(index/'manifest.json')
    manifest['files_sha256']['index.sqlite']=checksum(index/'index.sqlite')
    write(index/'manifest.json',manifest)
    from openpi import transforms as tr
    dc=openpi_config()
    dc.model_transforms=tr.Group(inputs=[tr.PadStatesAndActions(32),tr.RepackTransform(dict(
        image={k:f'image/{k}' for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        image_mask={k:f'image_mask/{k}' for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        state='state',actions='actions'))])
    _,_,data,accum=setup(config())
    def make():
        return IndexedOpenPILoader(IndexedDataset(index),HierarchicalSampler(index),dc,
                                  batch_size=16,num_batches=2,sharding=data)
    loader=make();iterator=iter(loader)
    first=next(iterator);saved=loader.state_dict()
    expected=next(iterator);ids=loader.last_sample_ids.copy();iterator.close()
    restored=make();restored.load_state_dict(saved);it=iter(restored)
    actual=next(it);it.close()
    np.testing.assert_array_equal(ids,restored.last_sample_ids)
    assert_trees_close(actual,expected)
    batches=batch_preparer(16,8,data,accum)(first)
    assert batches.actions.shape==(2,8,50,32)
    assert all(s.data.shape==(2,1,50,32) for s in batches.actions.addressable_shards)
    assert set(np.asarray(batches.action_loss_mask).sum(-1).flat)<= {15,30,50}


def test_wrong_device_count_fails_before_loading():
    cfg=config();cfg['parallelism']['num_devices']=1
    with pytest.raises(ValueError,match='Expected 1 visible local devices'):setup(cfg)
