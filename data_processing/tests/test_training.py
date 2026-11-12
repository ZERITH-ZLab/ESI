import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('openpi')
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from openpi.training.utils import TrainState
from openpi.shared.normalize import NormStats,load as load_norms
from openpi.models.model import restore_params

from data_processing.training.step import masked_mean,train_step
from data_processing.training import checkpoints
from data_processing.training.config import load_config
from data_processing.openpi_loader.loader import TrainingBatch
from data_processing.sampling.sampler import HierarchicalSampler
from data_processing.tests.test_openpi_loader import loader_index


class TinyModel(nnx.Module):
    def __init__(self):
        self.weight=nnx.Param(jnp.asarray(.25))

    def compute_loss(self,rng,observation,actions,train):
        del rng,train
        prediction=self.weight.value*observation[:,None,:]
        return jnp.mean((prediction-actions)**2,axis=-1)


def state_and_batch():
    model=TinyModel(); params=nnx.state(model)
    tx=optax.adam(1e-3)
    state=TrainState(step=0,params=params,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(params),
                     ema_decay=None,ema_params=None)
    batch=TrainingBatch(jnp.ones((4,1)),jnp.arange(4*50,dtype=jnp.float32).reshape(4,50,1)/100,
        jnp.arange(50)[None,:]<jnp.array([15,25,40,50])[:,None])
    return state,batch


def test_masked_loss_equal_window_weight_and_zero_masked_output_gradient():
    loss=jnp.array([[2.,2.,999.,999.],[4.,4.,4.,4.]])
    mask=jnp.array([[True,True,False,False],[True]*4])
    assert float(masked_mean(loss,mask))==3.
    grad=jax.grad(masked_mean)(loss,mask)
    np.testing.assert_array_equal(grad,np.array([[.25,.25,0,0],[.125]*4]))
    assert float(masked_mean(loss,jnp.ones_like(mask)))==float(loss.mean())
    assert np.isnan(float(masked_mean(loss,jnp.zeros_like(mask))))
    with pytest.raises(ValueError): masked_mean(loss,jnp.ones_like(loss))


def test_gradient_accumulation_matches_full_batch_update():
    state,batch=state_and_batch();cfg=SimpleNamespace(trainable_filter=nnx.Param)
    rng=jax.random.key(13)
    def split(n):return jax.tree.map(lambda x:x.reshape((n,4//n)+x.shape[1:]),batch)
    full,info=train_step(cfg,rng,state,split(1))
    accumulated,other=jax.jit(lambda st,b:train_step(cfg,rng,st,b))(state,split(2))
    assert int(full.step)==int(accumulated.step)==1
    assert float(info['update_norm'])>0
    for a,b in zip(jax.tree.leaves(full),jax.tree.leaves(accumulated)):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
    assert float(info['loss'])==pytest.approx(float(other['loss']))


class TinyLoader:
    def __init__(self,index):
        self.sampler=HierarchicalSampler(index)
    def state_dict(self):return self.sampler.state_dict()
    def load_state_dict(self,value):self.sampler.load_state_dict(value)
    def data_config(self):
        return SimpleNamespace(asset_id='arx_x5_sim',norm_stats={k:NormStats(
            mean=np.zeros(14),std=np.ones(14),q01=-np.ones(14),q99=np.ones(14)) for k in ('state','actions')})


def test_atomic_checkpoint_resume_next_update_and_inference_layout(loader_index,tmp_path):
    loader=TinyLoader(loader_index[0]);state,batch=state_and_batch()
    cfg=SimpleNamespace(trainable_filter=nnx.Param);rng=jax.random.key(5)
    batches=jax.tree.map(lambda x:x.reshape((2,2)+x.shape[1:]),batch)
    state,_=train_step(cfg,rng,state,batches)
    loader.sampler.draw_indices(4)
    with checkpoints.manager(tmp_path/'checkpoints',2) as manager:
        path=checkpoints.save(manager,state,loader,fingerprint='test',
                              train_rng_data=np.asarray(jax.random.key_data(rng)).tolist())
        assert path.name=='1'
        target=jax.eval_shape(lambda:state)
        new_loader=TinyLoader(loader_index[0])
        restored,progress=checkpoints.restore(manager,target,new_loader,fingerprint='test')
        np.testing.assert_array_equal(loader.sampler.draw_indices(20),new_loader.sampler.draw_indices(20))
        resumed_rng=jax.random.wrap_key_data(np.array(progress['train_rng_data'],dtype=np.uint32))
        a,_=train_step(cfg,rng,state,batches)
        b,_=train_step(cfg,resumed_rng,restored,batches)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_array_equal(x,y)
        assert load_norms(path/'assets/arx_x5_sim')['actions'].q99.tolist()==[1.]*14
        # Native inference reader accepts the params item saved by our manager.
        params=restore_params(path/'params',dtype=jnp.float32)
        np.testing.assert_allclose(params['weight'],state.params.to_pure_dict()['weight'])
        with pytest.raises(ValueError,match='differs'):
            checkpoints.restore(manager,target,new_loader,fingerprint='changed')
        for state in (a,dataclasses_replace_step(a,3)):
            checkpoints.save(manager,state,loader,fingerprint='test',train_rng_data=[0,5])
        assert sorted(manager.all_steps())==[2,3]


def dataclasses_replace_step(state,step):
    import dataclasses
    return dataclasses.replace(state,step=step)


def test_config_rejects_ambiguous_batch_and_schedule(tmp_path):
    path=Path(__file__).parents[1]/'training/config.example.json'
    cfg=load_config(path)
    cfg['training']['micro_batch_size']=3
    p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='divisible'):load_config(p)
    cfg['training']['micro_batch_size']=2
    cfg['optimizer']['warmup_steps']=cfg['training']['num_steps']
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='warmup'):load_config(p)


def test_parallel_config_validation_and_legacy_defaults(tmp_path):
    path=Path(__file__).parents[1]/'training/config.8gpu_smoke.json'
    cfg=load_config(path)
    assert cfg['training']['micro_batch_size']//cfg['parallelism']['num_devices']==1
    p=tmp_path/'cfg.json'
    cfg['training']['micro_batch_size']=4
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Global micro_batch_size'):load_config(p)
    cfg['training']['micro_batch_size']=8
    cfg['parallelism']['fsdp_devices']=3
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='fsdp_devices'):load_config(p)
    del cfg['parallelism']
    p.write_text(json.dumps(cfg))
    assert load_config(p)['parallelism']==dict(num_devices=1,fsdp_devices=1)


def test_training_entry_stop_and_resume_with_native_init_and_optimizer(loader_index,tmp_path,monkeypatch):
    """Full entry plumbing with a tiny model, real OpenPI init/Adam/checkpoints.

    Heavy model and image I/O are substituted; no giant checkpoint is copied.
    """
    import dataclasses
    import orbax.checkpoint as ocp
    from openpi.training import config as native_config
    from data_processing.training import runner
    from hybrid_rollout.robodojo.pi05_server import checkpoint as base_contract

    path=Path(__file__).parents[1]/'training/config.example.json'
    cfg=load_config(path)
    cfg['base_checkpoint']=str(tmp_path/'base')
    cfg['output']=str(tmp_path/'run')
    cfg['runtime']['require_gpu']=False
    cfg['training'].update(num_steps=3,batch_size=2,micro_batch_size=1,num_workers=0)
    cfg['optimizer']['warmup_steps']=0
    cfg['checkpoint']['save_interval']=2
    base=Path(cfg['base_checkpoint']);base.mkdir()
    with ocp.PyTreeCheckpointer() as cp:
        cp.save(str(base/'params'),{'params':nnx.state(TinyModel()).to_pure_dict()})
    model_cfg=SimpleNamespace(create=lambda rng:TinyModel())
    native_cfg=dataclasses.replace(native_config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0'),model=model_cfg)
    monkeypatch.setattr(base_contract,'data_contract',lambda _: (native_cfg,None))
    monkeypatch.setattr(base_contract,'checkpoint_identity',lambda _:dict(checkpoint_sha256='tiny-model-fixture'))

    class BatchLoader(TinyLoader):
        fingerprint='tiny-loader-fixture'
        def __iter__(self):
            for _ in range(3):
                self.last_sample_ids=self.sampler.draw_sample_ids(2)
                yield TrainingBatch(jnp.ones((2,1)),jnp.ones((2,50,1)),jnp.ones((2,50),dtype=bool))
    monkeypatch.setattr(runner,'make_loader',lambda *args,**kwargs:BatchLoader(loader_index[0]))
    result=runner.run(cfg,stop_after=1)
    assert result['step']==1 and result['status']=='stopped_at_requested_step'
    result=runner.run(cfg,resume=True)
    assert result['step']==3 and result['status']=='completed'
    logs=[json.loads(s) for s in (Path(cfg['output'])/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs if 'step' in r]==[1,2,3]
    assert all(r['update_norm']>0 for r in logs if 'step' in r)
    assert (Path(result['checkpoint'])/'assets/arx_x5_sim/norm_stats.json').exists()
