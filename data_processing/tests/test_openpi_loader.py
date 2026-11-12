import json
import pickle
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from data_processing.subtasks.io import write
from data_processing.training_index import VERSION as INDEX_VERSION
from data_processing.training_index.build import SCHEMA, checksum
from data_processing.sampling.sampler import HierarchicalSampler
from data_processing.openpi_loader.dataset import IndexedDataset
from data_processing.openpi_loader.loader import IndexedOpenPILoader, RowSampler


@pytest.fixture
def loader_index(tmp_path):
    archive = tmp_path/'raw'; archive.mkdir()
    index = tmp_path/'index'; index.mkdir()
    states = np.arange(100*14, dtype=np.float32).reshape(100,14)/1000
    commands = states + 10  # Distinguish command labels from measured state.
    np.savez(tmp_path/'steps.npz', state=states, command=commands)
    with sqlite3.connect(index/'index.sqlite') as db:
        db.executescript(SCHEMA)
        ep = {r[1]: '' for r in db.execute('PRAGMA table_info(episodes)')}
        ep.update(episode_key='ep', episode_id='episode', source_archive=str(archive),
            policy_version='policy', collection_round='round', task='task', variant='random',
            episode_success=1, total_steps=100, control_dt=.04, task_instruction='Move object',
            steps_npz=str(tmp_path/'steps.npz'), action_array='command', state_array='state',
            shard_episode_index=0, metadata_json=json.dumps(dict(policy_config='test',
                evaluation_case=dict(task='task',variant='random',case_id='case',eval_seed=0,layout_id=0))))
        db.execute(f'INSERT INTO episodes ({",".join(ep)}) VALUES ({",".join("?" for _ in ep)})', list(ep.values()))
        db.execute('INSERT INTO sources VALUES (?,?)', (ep['steps_npz'],checksum(ep['steps_npz'])))
        for sid, t, valid, end in [(10,0,30,30),(20,40,50,90),(30,85,15,100)]:
            # Non-contiguous IDs catch accidental sample_id-as-row-index use.
            path = archive/f'{t:06d}.npz'
            image = np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
            np.savez(path, states=states[t], instruction=np.array('Move object'),
                     cam_high=image, cam_left_wrist=image+1, cam_right_wrist=image+2)
            db.execute('INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                ('ep',t,t,1,'gpt' if t!=40 else 'pi05','s','transport','success',0,1,1,'[]',path.name,checksum(path),'unused'))
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)',
                (sid,'ep',t,end,valid,min(50,100-t),max(0,t+50-100),valid if t!=40 else 0,
                 'full' if valid==50 else 'gpt_tail'))
    write(index/'manifest.json',dict(version=INDEX_VERSION,status='completed',
        files_sha256={'index.sqlite':checksum(index/'index.sqlite')},config=dict(horizon=50,
        supervision='continuous_valid_prefix',interior_tail='real_actions',episode_padding='repeat_last',
        instruction='episode_task',action_label='executed_command')))
    return index, states, commands


def test_real_tail_padding_start_observation_and_row_ids(loader_index):
    index, states, commands = loader_index
    ds = IndexedDataset(index)
    assert ds.sample_ids.tolist() == [10,20,30]
    a, b, c = [ds[i] for i in range(3)]
    np.testing.assert_array_equal(a['action'], commands[:50])
    np.testing.assert_array_equal(a['action_loss_mask'], [True]*30+[False]*20)
    np.testing.assert_array_equal(b['action'], commands[40:90])
    assert b['action_loss_mask'].all()
    np.testing.assert_array_equal(c['action'][:15], commands[85:])
    np.testing.assert_array_equal(c['action'][15:], np.repeat(commands[-1:],35,axis=0))
    np.testing.assert_array_equal(c['observation.state'], states[85])
    assert c['action_loss_mask'].sum() == 15
    assert a['observation.images.cam_high'].shape == (3,4,5)
    assert a['observation.images.cam_high'][2,1,3] == 26
    assert a['prompt'] == 'Move object'
    # Transformation must never mutate the cached original command arrays.
    a['action'][:] = 999
    np.testing.assert_array_equal(ds[0]['action'], commands[:50])
    assert not pickle.loads(pickle.dumps(ds))._cache
    with pytest.raises(ValueError, match='row index'): ds[10]


def test_changed_source_is_rejected(loader_index):
    index, _, _ = loader_index
    ds = IndexedDataset(index)
    path = ds.episodes['ep']['steps_npz']
    with open(path,'ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='Source changed'): ds[0]


def test_changed_observation_is_rejected(loader_index):
    index, _, _ = loader_index
    ds = IndexedDataset(index)
    from pathlib import Path
    path = Path(ds.episodes['ep']['source_archive'])/ds.rows[0]['observation_relpath']
    with path.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='Observation changed'): ds[0]


def test_prefetch_rng_does_not_commit_and_resumes(loader_index):
    sampler = HierarchicalSampler(loader_index[0])
    planned = list(RowSampler(sampler,100))
    assert sampler.draw_count == 0
    assert sampler.draw_indices(4).tolist() == planned[:4]
    checkpoint = sampler.state_dict()
    # Discard all prefetched work beyond those four consumed rows.
    assert list(RowSampler(sampler,96)) == planned[4:]
    sampler.draw_indices(50)
    sampler.load_state_dict(checkpoint)
    assert sampler.draw_indices(96).tolist() == planned[4:]


def openpi_config():
    pytest.importorskip('openpi')
    from openpi import transforms as tr
    from openpi.policies.aloha_policy import AlohaInputs
    from openpi.shared.normalize import NormStats
    return SimpleNamespace(repo_id='indexed',norm_stats={k:NormStats(mean=np.zeros(14),std=np.ones(14)) for k in ('state','actions')},
        use_quantile_norm=False,
        repack_transforms=tr.Group(inputs=[tr.RepackTransform(dict(
            state='observation.state',actions='action',prompt='prompt',images={
                c:f'observation.images.{c}' for c in ('cam_high','cam_left_wrist','cam_right_wrist')}))]),
        data_transforms=tr.Group(inputs=[AlohaInputs(adapt_to_pi=False),tr.DeltaActions(tr.make_bool_mask(6,-1,6,-1))]),
        model_transforms=tr.Group(inputs=[tr.PadStatesAndActions(32)]))


def test_official_transform_equivalence_and_mask_preservation(loader_index):
    from data_processing.openpi_loader.dataset import TransformedDataset
    cfg = openpi_config()
    from openpi.training.data_loader import transform_dataset
    ds = IndexedDataset(loader_index[0]); adapted = TransformedDataset(ds,cfg)
    for i in range(len(ds)):
        raw = ds[i]
        expected = transform_dataset([raw],cfg)[0]
        actual = adapted[i]
        raw = ds[i]  # Official DeltaActions mutates its input action array.
        np.testing.assert_array_equal(actual['actions'], expected['actions'])
        np.testing.assert_array_equal(actual['state'], expected['state'])
        np.testing.assert_array_equal(actual['action_loss_mask'],raw['action_loss_mask'])
        assert actual['actions'].shape == (50,32)
        # Joints relative to the window start, grippers remain absolute.
        np.testing.assert_allclose(actual['actions'][:,0],
            (raw['action'][:,0]-raw['observation.state'][0])/(1+1e-6),rtol=1e-6)
        np.testing.assert_allclose(actual['actions'][:,6],raw['action'][:,6]/(1+1e-6),rtol=1e-6)


@pytest.mark.parametrize('workers', [0, 2])
def test_native_batch_and_consumed_checkpoint_resume(loader_index, workers):
    cfg = openpi_config()
    # No tokenizer in this unit fixture; remove prompt before native collation.
    from openpi import transforms as tr
    cfg.model_transforms = tr.Group(inputs=[tr.PadStatesAndActions(32),tr.RepackTransform(dict(
        image={k:f'image/{k}' for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        image_mask={k:f'image_mask/{k}' for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        state='state',actions='actions'))])
    def make():
        return IndexedOpenPILoader(IndexedDataset(loader_index[0]),HierarchicalSampler(loader_index[0]),
                                   cfg,batch_size=2,num_batches=4,num_workers=workers)
    loader=make(); it=iter(loader)
    first=next(it)
    assert first.actions.shape == (2,50,32) and first.action_loss_mask.shape == (2,50)
    assert first.observation.images['base_0_rgb'].shape == (2,4,5,3)
    state=loader.state_dict()
    expected=next(it); ids=loader.last_sample_ids.copy(); it.close()
    restored=make(); restored.load_state_dict(state)
    ri=iter(restored); actual=next(ri); ri.close()
    np.testing.assert_array_equal(restored.last_sample_ids,ids)
    np.testing.assert_array_equal(actual.actions,expected.actions)
    np.testing.assert_array_equal(actual.action_loss_mask,expected.action_loss_mask)
