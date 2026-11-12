"""Two-process integration check, invoked once per rank with a shared test directory."""
import functools
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
from openpi.training import sharding
from openpi.training.utils import TrainState

from data_processing.training.distributed import barrier, initialize, local_positions
from data_processing.training.parallel import setup
from data_processing.training.step import train_step
from data_processing.training import checkpoints
from data_processing.tests.test_training import TinyLoader
from data_processing.tests.test_openpi_loader import loader_index, openpi_config
from data_processing.openpi_loader.loader import TrainingBatch, IndexedOpenPILoader
from data_processing.openpi_loader.dataset import IndexedDataset
from data_processing.sampling.sampler import HierarchicalSampler


class MatrixModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.full((1024,1024),.001,jnp.float32))

    def compute_loss(self,rng,observation,actions,train):
        observation = sharding.activation_sharding_constraint(observation)
        prediction = (observation@self.weight.value).mean(-1)[:,None,None]
        return ((prediction-actions)**2).mean(-1)


def main():
    initialize()
    rank = jax.process_index()
    root = Path(os.environ['MULTIHOST_TEST_DIR'])
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        loader_index.__wrapped__(root)
    barrier('fixture')
    cfg = dict(parallelism=dict(num_devices=jax.device_count(), fsdp_devices=jax.local_device_count()),
               runtime=dict(require_gpu=False))
    mesh, rep, _, accum = setup(cfg)
    # Exercise real indexed loader, local sample selection and consumption resume.
    dc = openpi_config()
    from openpi import transforms as tr
    dc.model_transforms = tr.Group(inputs=[tr.PadStatesAndActions(32), tr.RepackTransform(dict(
        image={k:f'image/{k}' for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        image_mask={k:f'image_mask/{k}' for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        state='state',actions='actions'))])
    def make():
        return IndexedOpenPILoader(IndexedDataset(root/'index'), HierarchicalSampler(root/'index'), dc,
            batch_size=2, micro_batch_size=2, num_batches=3, sharding=accum)
    loader = make(); it = iter(loader)
    next(it); saved = loader.state_dict(); expected = next(it); ids = loader.last_sample_ids.copy(); it.close()
    restored = make(); restored.load_state_dict(saved); it = iter(restored); actual = next(it); it.close()
    np.testing.assert_array_equal(ids, restored.last_sample_ids)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
        for x,y in zip(a.addressable_shards,b.addressable_shards,strict=True):
            np.testing.assert_array_equal(x.data,y.data)
    # Synthetic full-parameter Adam update: masked loss and accumulation must
    # agree with a local reference despite parameter and data sharding.
    model = MatrixModel(); params = nnx.state(model); tx = optax.adam(1e-3)
    state = TrainState(step=0,params=params,model_def=nnx.graphdef(model),tx=tx,
                      opt_state=tx.init(params),ema_decay=None,ema_params=None)
    shape = jax.eval_shape(lambda:state)
    placement = sharding.fsdp_sharding(shape,mesh)
    distributed = jax.tree.map(lambda x,s:jax.make_array_from_process_local_data(s,np.asarray(x)),state,placement)
    b = TrainingBatch(np.arange(8,dtype=np.float32)[:,None]*np.ones((8,1024),np.float32)*.01,
        np.arange(8*50,dtype=np.float32).reshape(8,50,1)*.001,
        np.arange(50)[None,:]<np.tile([15,25,40,50],2)[:,None])
    pos = local_positions(8,4,2,rank)
    local = jax.tree.map(lambda x:x[pos].reshape((2,2)+x.shape[1:]),b)
    global_batch = jax.tree.map(lambda x:jax.make_array_from_process_local_data(accum,x,
        global_shape=(2,4)+x.shape[2:]),local)
    step_cfg = SimpleNamespace(trainable_filter=nnx.Param); rng = jax.random.key(7)
    pstep = jax.jit(functools.partial(train_step,step_cfg),
        in_shardings=(rep,placement,accum),out_shardings=(placement,rep))
    with sharding.set_mesh(mesh):
        updated, info = pstep(rng,distributed,global_batch)
    reference, ref_info = jax.jit(functools.partial(train_step,step_cfg))(rng,state,
        jax.tree.map(lambda x:jnp.asarray(x.reshape((2,4)+x.shape[1:])),b))
    for k in info:
        np.testing.assert_allclose(info[k],ref_info[k],rtol=2e-5,atol=2e-7)
    tl = TinyLoader(root/'index'); tl.sampler.draw_indices(8)
    if rank == 0:
        import time
        time.sleep(2)  # Rank 1 must safely arrive at a nonexistent root first.
    with checkpoints.manager(root/'checkpoints',2) as manager:
        checkpoints.save(manager,updated,tl,fingerprint='multihost',train_rng_data=[0,7])
        replacement = TinyLoader(root/'index')
        resumed,_ = checkpoints.restore(manager,shape,replacement,fingerprint='multihost',target_sharding=placement)
        np.testing.assert_array_equal(tl.sampler.draw_indices(16),replacement.sampler.draw_indices(16))
        for a,b in zip(jax.tree.leaves(updated),jax.tree.leaves(resumed),strict=True):
            for x,y in zip(a.addressable_shards,b.addressable_shards,strict=True):
                np.testing.assert_array_equal(x.data,y.data)
    print(f'MULTIHOST PASSED rank={rank}',flush=True)
    barrier('finished')
    jax.distributed.shutdown()


if __name__ == '__main__':
    main()
