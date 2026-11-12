"""OpenPI FSDP/data mesh for one process per node, including accumulated batches."""
import jax
import numpy as np
from openpi.training import sharding
from data_processing.subtasks.io import require


def setup(cfg):
    """Fail before model loading if the allocation does not match the config."""
    par=cfg['parallelism']
    from .distributed import initialize
    initialize()
    require(jax.device_count()==par['num_devices'] and
            jax.local_device_count()*jax.process_count()==jax.device_count(),
            f"Expected {par['num_devices']} visible local devices, found "
            f'{jax.local_device_count()} local / {jax.device_count()} global')
    if cfg['runtime']['require_gpu']:
        require(all(d.platform=='gpu' for d in jax.devices()),'GPU required; use check-data for CPU loader validation')
    if jax.process_count() == 1:
        mesh=sharding.make_mesh(par['fsdp_devices'])
    else:
        # Contiguous process-major ordering keeps an eight-way FSDP group
        # inside each eight-GPU node; batch-axis replicas span nodes.
        devices=sorted(jax.devices(), key=lambda d:(d.process_index,d.id))
        mesh=jax.sharding.Mesh(np.array(devices).reshape(-1,par['fsdp_devices']),
                               (sharding.BATCH_AXIS,sharding.FSDP_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    data=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # scan's accumulation dimension is replicated. Only actual examples are split.
    accumulated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(None,sharding.DATA_AXIS))
    return mesh,replicated,data,accumulated


def batch_preparer(batch_size,micro_batch_size,data_sharding,accumulated_sharding):
    """Reshape AND reshard in one compiled operation, preserving global row order."""
    accumulation=batch_size//micro_batch_size
    def prepare(batch):
        return jax.tree.map(lambda x:x.reshape((accumulation,micro_batch_size)+x.shape[1:]),batch)
    return jax.jit(prepare,in_shardings=(data_sharding,),out_shardings=accumulated_sharding)
