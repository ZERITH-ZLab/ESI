import numpy as np
from data_processing.openpi_loader.loader import RowSampler, ProcessRowSampler
from data_processing.sampling.sampler import HierarchicalSampler
from data_processing.training.distributed import local_positions
from data_processing.tests.test_openpi_loader import loader_index


def test_nodes_partition_every_microbatch_and_resume(loader_index):
    sampler = HierarchicalSampler(loader_index[0])
    full = np.array(list(RowSampler(sampler, 768))).reshape(3, 256)
    nodes = [np.array(list(ProcessRowSampler(sampler, 768, 256, 128, 2, rank))).reshape(3, 128)
             for rank in range(2)]
    rebuilt = np.empty_like(full)
    for rank, local in enumerate(nodes):
        positions = local_positions(256, 128, 2, rank)
        rebuilt[:, positions] = local
    np.testing.assert_array_equal(rebuilt, full)
    assert sampler.draw_count == 0
    sampler.draw_indices(256)
    for rank in range(2):
        resumed = list(ProcessRowSampler(sampler, 512, 256, 128, 2, rank))
        np.testing.assert_array_equal(resumed, nodes[rank][1:].reshape(-1))


def test_no_cross_node_overlap_in_positions():
    for nodes in (1, 2, 4):
        p = [local_positions(256, 128, nodes, i) for i in range(nodes)]
        np.testing.assert_array_equal(np.sort(np.concatenate(p)), np.arange(256))
