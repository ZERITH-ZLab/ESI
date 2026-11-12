"""Finite full-parameter pi05 training; all generated artifacts live under output."""
import dataclasses
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from data_processing.openpi_loader.loader import create_data_loader
from data_processing.subtasks.io import digest,read,require,write
from data_processing.training_index.build import checksum
from . import VERSION


def event(name,**fields):
    print(json.dumps(dict(event=name,**fields)),flush=True)


def setup_runtime(cfg):
    # Must happen before importing JAX/OpenPI. Override an inherited inference
    # preallocation flag with this run's explicit training configuration.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=str(cfg['runtime']['jax_preallocate']).lower()
    os.environ['JAX_COMPILATION_CACHE_DIR']=str(Path(cfg['output'])/'jax_cache')


def make_loader(cfg,**kwargs):
    tr=cfg['training']
    return create_data_loader(cfg['index_root'],cfg['base_checkpoint'],
        norm_stats_dir=cfg['norm_stats_dir'],sampling_config=cfg['sampling'],
        instruction_mode=cfg.get('conditioning', {}).get('mode', 'task'),
        batch_size=tr['batch_size'],num_batches=tr['num_steps'],
        num_workers=tr['num_workers'],cache_episodes=tr['cache_episodes'],**kwargs)


def check_data(cfg):
    """Read one real batch with this exact config. No model/optimizer allocated."""
    setup_runtime(cfg)
    loader=make_loader(cfg)
    iterator=iter(loader)
    try:
        batch=next(iterator)
        result=dict(status='passed',windows=len(loader.dataset),contract=loader.contract,
            action_shape=list(batch.actions.shape),valid_steps=np.asarray(batch.action_loss_mask).sum(axis=1).tolist(),
            sample_ids=loader.last_sample_ids.tolist(),training_performed=False)
    finally:
        iterator.close()
    event('training_data_check',**result)
    return result


def run(cfg,*,resume=False,stop_after=None):
    setup_runtime(cfg)
    import jax
    import flax.nnx as nnx
    import openpi
    from openpi.training import optimizer,sharding,weight_loaders
    from hybrid_rollout.robodojo.pi05_server.checkpoint import data_contract,checkpoint_identity
    from .step import train_step
    from . import checkpoints
    from .parallel import setup,batch_preparer
    from .distributed import primary,output_lock,barrier

    if stop_after is not None:
        require(type(stop_after) is int and stop_after>0,'stop_after must be positive')
    mesh,replicated,data_sharding,accumulated_sharding=setup(cfg)
    topology=dict(num_devices=jax.device_count(),process_count=jax.process_count(),
        local_devices=jax.local_device_count(),fsdp_devices=cfg['parallelism']['fsdp_devices'],
        mesh_shape=dict(mesh.shape),device_kind=jax.devices()[0].device_kind,
        global_batch_size=cfg['training']['batch_size'],global_micro_batch_size=cfg['training']['micro_batch_size'],
        per_device_micro_batch_size=cfg['training']['micro_batch_size']//jax.device_count(),
        accumulation_steps=cfg['training']['batch_size']//cfg['training']['micro_batch_size'])
    event('training_topology',**topology)
    root=Path(cfg['openpi_source'])
    require(Path(openpi.__file__).resolve().parent == (root/'src/openpi').resolve(),
            'Imported OpenPI differs from configured openpi_source')
    source=root/'scripts/train.py'
    spec=importlib.util.spec_from_file_location('indexed_native_openpi_train',source)
    native=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=native
    spec.loader.exec_module(native)
    tr,opt,ck=cfg['training'],cfg['optimizer'],cfg['checkpoint']
    base_cfg,_=data_contract(cfg['base_checkpoint'])
    schedule=optimizer.CosineDecaySchedule(warmup_steps=opt['warmup_steps'],peak_lr=opt['learning_rate'],
        decay_steps=tr['num_steps'],decay_lr=opt['end_learning_rate'])
    train_cfg=dataclasses.replace(base_cfg,batch_size=tr['batch_size'],num_train_steps=tr['num_steps'],
        seed=tr['seed'],fsdp_devices=cfg['parallelism']['fsdp_devices'],ema_decay=None,freeze_filter=nnx.Nothing(),wandb_enabled=False,
        optimizer=optimizer.AdamW(**{k:v for k,v in opt.items() if k not in
                                    ('warmup_steps','learning_rate','end_learning_rate')}),
        lr_schedule=schedule,weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(cfg['base_checkpoint'])/'params')))
    output=Path(cfg['output'])
    with output_lock(output):
        require(resume or not (output/'run.json').exists(),'Output already initialized; use --resume or a new output')
        require(not resume or (output/'run.json').exists(),'No initialized run to resume')
        barrier('output_preflight_checked')
        distributed=jax.process_count()>1
        loader=make_loader(cfg,sharding=accumulated_sharding if distributed else data_sharding,
                           **({'micro_batch_size':tr['micro_batch_size']} if distributed else {}))
        prepare_batch=batch_preparer(tr['batch_size'],tr['micro_batch_size'],data_sharding,accumulated_sharding)
        event('checking_base_checkpoint',path=cfg['base_checkpoint'])
        identity=checkpoint_identity(cfg['base_checkpoint'])
        source_hashes={str(p.relative_to(root)):checksum(p) for p in
            [source,*sorted((root/'src/openpi').rglob('*.py'))]}
        implementation={p.name:checksum(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        contract=dict(version=VERSION,config=cfg,loader_fingerprint=loader.fingerprint,
            base_checkpoint_sha256=identity['checkpoint_sha256'],implementation=implementation,openpi_code=source_hashes,
            conditioning=getattr(loader, 'contract', {}).get('conditioning', {}),
            finetune='full_parameters',ema=False,loss='mean_over_windows(sum_valid_loss/valid_length)',
            attention='unchanged',framework='OpenPI/JAX')
        fingerprint=digest(contract)
        if resume:
            require(read(output/'run.json')['fingerprint']==fingerprint,'Run config/code/data changed; use a new output')
        elif primary():
            write(output/'run.json',dict(fingerprint=fingerprint,**contract))
        if primary() and hasattr(loader, 'contract'):
            write(output/'conditioning_report.json', loader.contract.get('conditioning', {}))
        rng=jax.random.key(tr['seed']); train_rng,init_rng=jax.random.split(rng)
        with checkpoints.manager(output/'checkpoints',ck['max_to_keep']) as mngr:
            resuming=resume and mngr.latest_step() is not None
            event('initializing_model',resume_checkpoint=mngr.latest_step() if resuming else None)
            state,state_sharding=native.init_train_state(train_cfg,init_rng,mesh,resume=resuming)
            if resuming:
                state,progress=checkpoints.restore(mngr,state,loader,fingerprint=fingerprint,
                                                   target_sharding=state_sharding)
                train_rng=jax.random.wrap_key_data(np.asarray(progress['train_rng_data'],dtype=np.uint32))
            jax.block_until_ready(state)
            start=int(state.step)
            require(loader.sampler.draw_count==start*tr['batch_size'],'Optimizer and consumed data progress differ')
            end=min(tr['num_steps'],start+stop_after) if stop_after else tr['num_steps']
            pstep=jax.jit(functools.partial(train_step,train_cfg),
                in_shardings=(replicated,state_sharding,accumulated_sharding),
                out_shardings=(state_sharding,replicated),donate_argnums=(1,))
            event('training_start',start_step=start,end_step=end,batch_size=tr['batch_size'],
                  micro_batch_size=tr['micro_batch_size'],output=str(output))
            iterator=iter(loader)
            try:
                with (output/'metrics.jsonl' if primary() else output/f'metrics.rank{jax.process_index()}.jsonl').open('a',buffering=1) as metrics:
                    metrics.write(json.dumps(dict(event='session_start',start_step=start,resume=resuming,**topology))+'\n')
                    for _ in range(start,end):
                        tick=time.monotonic()
                        batch=next(iterator)
                        load_s=time.monotonic()-tick
                        if not distributed:
                            counts=np.asarray(batch.action_loss_mask).sum(axis=1)
                            require((counts>0).all(),'Empty supervision mask')
                        tick=time.monotonic()
                        batches=batch if distributed else prepare_batch(batch)
                        with sharding.set_mesh(mesh):
                            state,info=pstep(train_rng,state,batches)
                        info={k:float(v) for k,v in jax.device_get(info).items()}
                        require(all(np.isfinite(v) for v in info.values()),'Nonfinite loss/gradient/update; no bad checkpoint saved')
                        step=int(state.step)
                        record=dict(step=step,**info,learning_rate=float(schedule.create()(step-1)),
                            load_s=load_s,update_s=time.monotonic()-tick,sample_ids=loader.last_sample_ids.tolist())
                        metrics.write(json.dumps(record)+'\n');event('train_step',**record)
                        if step%ck['save_interval']==0 or step==end:
                            save_start=time.monotonic()
                            path=checkpoints.save(mngr,state,loader,fingerprint=fingerprint,
                                train_rng_data=np.asarray(jax.random.key_data(train_rng)).tolist())
                            if primary():
                                write(output/'latest.json',dict(step=step,checkpoint=str(path),
                                    inference_checkpoint=str(path),norm_stats=str(path/'assets'/loader.data_config().asset_id/'norm_stats.json')))
                            barrier(f'checkpoint_{step}_published')
                            saved=dict(event='checkpoint_saved',step=step,path=str(path),save_s=time.monotonic()-save_start)
                            metrics.write(json.dumps(dict(event='checkpoint_saved',checkpoint_step=step,
                                                          save_s=saved['save_s']))+'\n')
                            event('checkpoint_saved',**{k:v for k,v in saved.items() if k!='event'})
                latest=mngr.latest_step()
                result=dict(status='completed' if end==tr['num_steps'] else 'stopped_at_requested_step',
                    step=int(state.step),target_steps=tr['num_steps'],output=str(output),
                    checkpoint=str(output/'checkpoints'/str(latest)) if latest is not None else None)
                if primary():
                    write(output/'result.json',result)
                event('training_finished',**result)
                return result
            finally:
                iterator.close()
