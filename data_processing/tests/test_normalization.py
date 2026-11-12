import numpy as np
import pytest

from data_processing.normalization.compute import robust_statistics, collect_vectors
from data_processing.tests.test_openpi_loader import loader_index
from data_processing.openpi_loader.dataset import IndexedDataset
from data_processing.normalization.compute import validate_statistics
from data_processing.normalization import VERSION
from data_processing.subtasks.io import write
from data_processing.training_index.build import checksum


def test_exact_robust_quantiles_trim_extremes_without_double_trimming():
    v = np.tile(np.arange(1000,dtype=np.float64)[:,None],(1,14))
    v[0] = -1e12; v[-1] = 1e12
    norms, report = robust_statistics(v)
    assert norms['q01'][0] == pytest.approx(9.99)
    assert norms['q99'][0] == pytest.approx(989.01)
    assert norms['mean'][0] == pytest.approx(np.arange(10,990).mean())
    assert norms['std'][0] == pytest.approx(np.arange(10,990).std())
    assert report[0]['excluded_low'] == report[0]['excluded_high'] == 10
    assert v[0,0] == -1e12  # Statistics must not change labels.


def test_invalid_or_collapsed_data_requires_explicit_handling():
    v=np.ones((100,14))
    with pytest.raises(ValueError,match='collapsed'): robust_statistics(v)
    v[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'): robust_statistics(v)


def test_collect_only_supervised_window_relative_actions(loader_index):
    index, states, commands=loader_index
    ds=IndexedDataset(index)
    def transform(raw):
        action=raw['action'].copy()
        dims=[0,1,2,3,4,5,7,8,9,10,11,12]
        action[:,dims]-=raw['observation.state'][dims]
        return dict(state=raw['observation.state'],actions=action)
    s,a=collect_vectors(ds,transform)
    assert s.shape == (3,14) and a.shape == (95,14)
    np.testing.assert_array_equal(s,states[[0,40,85]])
    np.testing.assert_allclose(a[:30,0],commands[:30,0]-states[0,0])
    np.testing.assert_allclose(a[30:80,0],commands[40:90,0]-states[40,0])
    np.testing.assert_allclose(a[80:,0],commands[85:,0]-states[85,0])
    np.testing.assert_array_equal(a[:,6],np.concatenate([commands[:30,6],commands[40:90,6],commands[85:,6]]))


def test_normalizer_cannot_be_reused_for_another_index_or_modified(tmp_path):
    write(tmp_path/'norm_stats.json',dict(norm_stats={}))
    write(tmp_path/'manifest.json',dict(version=VERSION,status='completed',index_sha256='index-a',
        model_config='pi05',horizon=50,files_sha256={'norm_stats.json':checksum(tmp_path/'norm_stats.json')}))
    assert validate_statistics(tmp_path,'index-a','pi05',50)['status']=='completed'
    with pytest.raises(ValueError,match='different training index'):
        validate_statistics(tmp_path,'index-b','pi05',50)
    write(tmp_path/'norm_stats.json',dict(changed=True))
    with pytest.raises(ValueError,match='artifact changed'):
        validate_statistics(tmp_path,'index-a','pi05',50)
