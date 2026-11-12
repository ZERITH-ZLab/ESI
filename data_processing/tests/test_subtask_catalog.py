"""Readable aliases must preserve identities, references and resume behaviour."""
import csv
from pathlib import Path

import pytest

from data_processing.subtasks.catalog import build_catalog, episode_name, link_directory
from data_processing.subtasks.io import read, write


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / 'batch'
    key = 'b4b4d8007bbc0152023514ea'
    archive = '/data/experiment/arrange_largest_number/random/eval_seed_0/layout_0/replica_3/repeat_001/attempt_000'
    row = dict(key=key, episode_id='original-episode-id', archive=archive,
               path=f'prepared/{key}/input.json')
    missing = dict(row, key='missing', episode_id='pending-episode')
    write(root / 'manifest.json', dict(episodes=[row, missing]))
    run = root / 'runs' / '5503c88ca0d24593cfc7'
    write(run / 'config.json', dict(model='gpt-6.1-sol', reasoning_effort='medium'))
    write(run / 'protocol.json', dict(version='subtasks.v6'))
    write(run / 'segments.json', [])
    ep = run / 'episodes' / key
    write(ep / 'annotation.json', dict(original=True))
    write(ep / 'status.json', dict(state='annotated'))
    return root, run, row


def test_catalog_old_run_is_idempotent_and_keeps_original_files(batch):
    root, run, row = batch
    original = {p: p.read_bytes() for p in root.rglob('*.json')}
    result = build_catalog(root)
    assert result['model_calls'] == 0
    alias = Path(result['runs'][0]['named_run_root'])
    assert alias.name.startswith('subtasks.v6__gpt-6.1-sol__medium__')
    assert alias.resolve() == run
    named_ep = alias / 'episodes_by_name' / episode_name(row)
    assert 'layout_0__replica_3__repeat_001__attempt_000' in named_ep.name
    assert (named_ep / 'annotation.json').resolve() == run / 'episodes' / row['key'] / 'annotation.json'
    assert read(named_ep / 'annotation.json') == {'original': True}
    assert all(p.read_bytes() == value for p, value in original.items())
    catalog = read(run / 'episodes.index.json')
    assert catalog['episodes'][0]['source_archive'] == row['archive']
    assert catalog['episodes'][1]['annotation'] is None
    assert catalog['episodes'][1]['state'] == 'not_started'
    with (run / 'episodes.index.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['episode_id'] == row['episode_id']
    assert len(rows) == 2
    assert build_catalog(root) == result
    assert build_catalog(root, run.name) == result


def test_failed_episode_is_not_presented_as_final_labels(batch):
    root, run, row = batch
    ep = run / 'episodes' / row['key']
    (ep / 'annotation.json').unlink()
    write(ep / 'draft_00.json', {'segments': []})
    write(ep / 'status.json', {'state': 'failed'})
    build_catalog(root)
    item = read(run / 'episodes.index.json')['episodes'][0]
    assert item['state'] == 'failed' and item['annotation'] is None
    assert (run / 'episodes_by_name' / item['name'] / 'draft_00.json').is_file()


def test_names_are_safe_and_disambiguate_same_case_across_experiments(batch):
    _, _, row = batch
    other = dict(row, key='different-key', archive=row['archive'].replace('/experiment/', '/experiment2/'))
    assert episode_name(other) != episode_name(row)
    malicious = dict(row, archive='/../' + '/'.join(['a b/../?'] * 7))
    name = episode_name(malicious)
    assert '/' not in name and len(name.encode()) <= 255


def test_catalog_does_not_overwrite_user_paths(batch, tmp_path):
    root, run, _ = batch
    link = tmp_path / 'name'
    link.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        link_directory(link, run)
    link.rmdir()
    link.symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(ValueError, match='elsewhere'):
        link_directory(link, run)
    with pytest.raises(ValueError, match='Invalid run ID'):
        build_catalog(root, '../escape')
