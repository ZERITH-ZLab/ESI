import json
from pathlib import Path

import pytest

from data_processing.subtasks.io import read, write
from data_processing.tests.test_subtasks import episode, candidate
from data_processing.tests.test_training_index import indexed_sources
from data_processing.training_index.build import build
from hybrid_rollout.robodojo.tools.test_extract_episode import fixture
from hybrid_rollout.robodojo.tools.extract_episode import extract


def test_finalized_wall_time_requires_explicit_opt_in_and_complete_receipts(tmp_path):
    raw=fixture(tmp_path)
    summary=read(raw/'sim/summary.json');summary['reason']='controller_error';write(raw/'sim/summary.json',summary)
    write(raw.parent/'sample.json',dict(attempts=[dict(archive=str(raw),state='wall_time_limit',finished_at=1)]))
    write(raw/'component_reset.json',dict(all_owned_processes_exited=True,ports_released=True))
    with pytest.raises(ValueError,match='Requires terminal'):
        extract(raw,tmp_path/'strict')
    assert extract(raw,tmp_path/'allowed',allow_wall_time_limit=True)['finalized_interruption']=='wall_time_limit'
    (raw/'sim/e/action_000001.json').unlink()
    with pytest.raises(ValueError,match='Action count'):
        extract(raw,tmp_path/'broken',allow_wall_time_limit=True)


def test_command_only_export_can_build_index(indexed_sources,tmp_path):
    run,export,*_=indexed_sources
    collection=read(export/'dataset_collection.json')
    row=read(export/'shard_000/export_manifest.json')['episodes'][0]
    row.update(state='completed',sidecar='shard_000/research/episode_000000')
    write(export/'dataset_collection.json',dict(schema='robodojo_command_collection.v1',status='completed',
        action_label='command',total_episodes=1,episodes=[row]))
    result=build(run,[export],tmp_path/'index','p','r',50,2)
    assert (tmp_path/'index/manifest.json').exists()
