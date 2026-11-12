"""One JAX process per node; every node follows the same global sample stream."""
import contextlib
import os

import numpy as np

from data_processing.subtasks.io import require
from data_processing.subtasks.pipeline import batch_lock


def initialize():
    import jax
    count = int(os.environ.get('JAX_PROCESS_COUNT', os.environ.get('WORLD_SIZE', '1')))
    if count > 1 and not jax.distributed.is_initialized():
        rank = int(os.environ.get('JAX_PROCESS_INDEX', os.environ.get('RANK', '-1')))
        address = os.environ.get('JAX_COORDINATOR_ADDRESS')
        if not address:
            require(os.environ.get('MASTER_ADDR'), 'Missing MASTER_ADDR/JAX_COORDINATOR_ADDRESS')
            # Leave the platform/PyTorch rendezvous port unused.
            port = int(os.environ.get('MASTER_PORT', '29500')) + 1
            address = f"{os.environ['MASTER_ADDR']}:{port}"
        require(0 <= rank < count, 'Invalid JAX node rank')
        jax.distributed.initialize(coordinator_address=address, num_processes=count,
                                   process_id=rank, initialization_timeout=300)


def primary():
    import jax
    return jax.process_index() == 0


def barrier(name):
    import jax
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(name)


@contextlib.contextmanager
def output_lock(path):
    """Only rank zero locks shared output; all ranks participate in checkpoints."""
    with batch_lock(path) if primary() else contextlib.nullcontext():
        barrier('output_lock_acquired')
        yield
        barrier('output_finished')


def local_positions(batch_size, micro_batch_size, process_count, process_index):
    require(batch_size % micro_batch_size == 0 and micro_batch_size % process_count == 0,
            'Batch/microbatch must divide evenly across nodes')
    require(0 <= process_index < process_count, 'Invalid process index')
    local_micro = micro_batch_size // process_count
    return np.arange(batch_size).reshape(-1, micro_batch_size)[:,
        process_index*local_micro:(process_index+1)*local_micro].reshape(-1)
