"""Atomic optimizer/params/normalizer/consumed-loader progress checkpoints."""
from pathlib import Path
import jax
import orbax.checkpoint as ocp
from openpi.shared import normalize
from openpi.shared import array_typing as at
from openpi.training import checkpoints as native
from data_processing.subtasks.io import require


def manager(directory,max_to_keep):
    from .distributed import barrier
    # Orbax 0.11's constructor may inspect the directory on a non-primary
    # process before rank zero creates it. mkdir is idempotent on shared CFS;
    # each rank resolves it locally before any manager scans existing steps.
    Path(directory).mkdir(parents=True,exist_ok=True)
    barrier('checkpoint_root_ready')
    return ocp.CheckpointManager(Path(directory),item_handlers={
        'assets':native.CallbackHandler(), 'train_state':ocp.PyTreeCheckpointHandler(),
        'params':ocp.PyTreeCheckpointHandler(), 'progress':ocp.JsonCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep,create=True,
            enable_async_checkpointing=True,async_options=ocp.AsyncOptions(timeout_secs=7200)))


def save(mngr,state,loader,*,fingerprint,train_rng_data):
    step=int(state.step)
    progress=dict(step=step,run_fingerprint=fingerprint,train_rng_data=train_rng_data,
                  loader_state=loader.state_dict())
    dc=loader.data_config()
    require(dc.norm_stats is not None and dc.asset_id, 'Checkpoint must include active normalization')
    def assets(directory):
        normalize.save(directory/dc.asset_id,dc.norm_stats)
    with at.disable_typechecking():
        stripped,params=native._split_params(state)
    mngr.save(step,items=dict(train_state=stripped,params={'params':params},assets=assets,progress=progress),
              metrics={'step':step})
    mngr.wait_until_finished()
    require(mngr.latest_step()==step,'Checkpoint was not committed')
    return Path(mngr.directory)/str(step)


def restore(mngr,target,loader,*,fingerprint,target_sharding=None):
    step=mngr.latest_step()
    require(step is not None,'No completed checkpoint to resume')
    progress=mngr.restore(step,items={'progress':None})['progress']
    require(progress['run_fingerprint']==fingerprint,'Resume config/code/data differs; use a new run directory')
    require(progress['step']==step,'Checkpoint step/progress mismatch')
    # Specify the destination placement rather than silently using saved device IDs.
    # The runner always supplies OpenPI's current FSDP sharding tree.
    if target_sharding is None:
        target_sharding=jax.tree.map(lambda x:getattr(x,'sharding',None) or
            jax.sharding.SingleDeviceSharding(jax.devices()[0]),target)
    with at.disable_typechecking():
        stripped,params=native._split_params(target)
        stripped_sharding,params_sharding=native._split_params(target_sharding)
        items=dict(train_state=stripped,params={'params':params})
        shardings=dict(train_state=stripped_sharding,params={'params':params_sharding})
        restored=mngr.restore(step,items=items,restore_kwargs={key:{
            'restore_args':ocp.checkpoint_utils.construct_restore_args(value,shardings[key])
        } for key,value in items.items()})
        state=native._merge_params(restored['train_state'],restored['params'])
    require(int(state.step)==step,'Restored optimizer step mismatch')
    loader.load_state_dict(progress['loader_state'])
    return state,progress
