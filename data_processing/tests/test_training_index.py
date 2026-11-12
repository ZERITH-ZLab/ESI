from copy import deepcopy
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
import pytest

from data_processing.tests.test_subtasks import episode, candidate
from data_processing.tests.test_subtasks_v6 import reviews_for
from data_processing.subtasks.audit import validate_final
from data_processing.subtasks.io import read, write
from data_processing.subtasks.pipeline import prepare_batch, protocol, load_config
from data_processing.subtasks.prepare import prepare
from data_processing.training_index.build import build, checksum, validate
from data_processing.training_index.windows import select_windows, window_positions


@pytest.mark.parametrize('gpt,cutoff', [(None,50), ((60,70),69), ((80,95),85), ((49,50),50)])
def test_confirmed_cutoff_examples(gpt, cutoff):
    controls=['pi05']*110
    if gpt:
        controls[gpt[0]:gpt[1]]=['gpt']*(gpt[1]-gpt[0])
    runs, windows=select_windows([True]*100+[False]*10,controls)
    assert runs[0]['last_allowed_start']==cutoff
    assert [w['start_step'] for w in windows]==list(range(cutoff+1))
    assert all(w['valid_length']==min(50,100-w['start_step']) for w in windows)
    if gpt==(60,70):
        # This is NOT a per-start-control threshold: intervening pi05 starts survive.
        assert windows[55]['selection']=='gpt_tail' and controls[55]=='pi05'


def test_invalid_gap_stops_loss_but_not_real_tail_and_no_time_splicing():
    valid=[True]*30+[False]*10+[True]*60
    _, windows=select_windows(valid,['gpt']*30+['pi05']*70)
    first=windows[0]
    positions,mask=window_positions(0,100,first['valid_length'])
    assert positions==list(range(50))
    assert mask==[True]*30+[False]*20  # Valid actions after the gap stay masked.
    assert not any(30<=w['start_step']<40 for w in windows)
    assert next(w for w in windows if w['start_step']==40)['valid_length']==50


def test_short_runs_episode_padding_and_zero_gpt_executions():
    assert select_windows([True]*14,['gpt']*14)[1]==[]
    assert select_windows([True]*30,['pi05']*30)[1]==[]
    _,w=select_windows([True]*30,['gpt']*30)
    assert w[-1]['start_step']==15 and w[-1]['valid_length']==15
    positions,mask=window_positions(15,30,15)
    assert positions==list(range(15,30))+[29]*35
    assert mask==[True]*15+[False]*35


def test_random_window_selection_matches_independent_start_rule():
    rng=random.Random(23)
    for _ in range(100):
        n=rng.randint(1,200)
        valid=[rng.random()>.05 for _ in range(n)]
        controls=[rng.choice(['gpt','pi05']) for _ in range(n)]
        _,windows=select_windows(valid,controls)
        expected=[]
        for t in range(n):
            if not valid[t]:
                continue
            end=t
            while end<n and valid[end]:end+=1
            # With >=15 remaining, a later/equal GPT action justifies this
            # short window. Earlier GPT actions alone cannot justify it.
            if end-t>=50 or (end-t>=15 and 'gpt' in controls[t:end]):expected.append(t)
        assert [w['start_step'] for w in windows]==expected


@pytest.fixture
def indexed_sources(episode,candidate,tmp_path):
    summary=read(episode/'sim/summary.json')
    summary.update(control_dt=.04,evaluation_case=dict(task='task',variant='random',
                                                     case_id='case-0',eval_seed=0,layout_id=0))
    write(episode/'sim/summary.json',summary)
    policy=dict(checkpoint_sha256='checkpoint-a',config='pi05-test',action_horizon=50,action_dim=14)
    write(episode/'policy_identity.json',policy)
    batch=tmp_path/'annotations'
    selection=prepare_batch([episode],batch)
    row=selection['episodes'][0]
    run=batch/'runs/test-run'
    write(run/'protocol.json',protocol());write(run/'config.json',load_config())
    dest=run/'episodes'/row['key']
    annotation=validate_final(prepare(episode),candidate,dict(reviews=reviews_for(candidate)))
    write(dest/'annotation.json',annotation);write(dest/'candidate.json',candidate)
    write(dest/'audit_final.json',dict(reviews=reviews_for(candidate)))
    export=tmp_path/'export';folder=export/'shard_000';side=folder/'research/episode_000000'
    side.mkdir(parents=True)
    n=10;states=np.arange(11,dtype=np.float32)[:,None]*np.ones((1,14),np.float32)/100
    commands=states[1:].copy();refs=[]
    for t in range(n+1):
        rel=f'sim/test-episode/observations/{t:06d}.npz';p=episode/rel;p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p,states=states[t],instruction=np.array('Move the object'),
                 cam_high=np.zeros((4,4,3),np.uint8))
        refs.append(dict(step=t,path=rel,sha256=checksum(p)))
        if t<n:
            write(episode/f'sim/test-episode/action_{t:06d}.json',
                  dict(valid=True,step_id=t+1,executed_action=commands[t].tolist()))
    np.savez(side/'steps.npz',state=states[:-1],command=commands,step=np.arange(n),decision_id=[0]*4+[2]*6)
    write(side/'observations.json',refs)
    write(side/'episode.json',dict(summary=summary,task_instruction='Move the object',provenance=dict(policy_identity=policy)))
    files=['steps.npz','observations.json','episode.json']
    write(side/'manifest.json',dict(schema='robodojo_training_extract.v2',source_archive=str(episode),
        episode_id='test-episode',steps=10,control_dt=.04,cameras=['cam_high'],
        file_sha256={name:checksum(side/name) for name in files}))
    erow=dict(episode_id='test-episode',episode_index=0,steps=10,mcap=None)
    write(folder/'export_manifest.json',dict(status='passed',action_label='command',episodes=[erow]))
    write(export/'dataset_collection.json',dict(status='completed',action_label='command',total_episodes=1,
          shards=[dict(shard='shard_000',episodes=[erow])]))
    return run,export,tmp_path/'index'


def test_build_reuse_rows_provenance_and_source_validation(indexed_sources):
    run,export,out=indexed_sources
    args=(run,[export],out,'policy-a','round0')
    report=build(*args,min_valid=5)
    assert report['episodes']==1 and report['steps']==10 and report['windows']==2
    assert report['gpt_valid_steps']==6 and report['gpt_lost_starts']==4
    assert validate(out)['samples']==2
    with sqlite3.connect(out/'index.sqlite') as db:
        db.row_factory=sqlite3.Row
        rows=list(db.execute('SELECT * FROM training_samples ORDER BY sample_id'))
        assert [r['start_step'] for r in rows]==[4,5]
        assert rows[0]['start_control']=='gpt' and rows[0]['is_decision_start']==1
        assert rows[0]['valid_length']==6 and rows[0]['real_length']==6 and rows[0]['pad_length']==44
        assert rows[0]['action_array']=='command' and rows[0]['state_array']=='state'
        assert db.execute('SELECT COUNT(*) FROM steps WHERE training_valid=0').fetchone()[0]==4
    assert build(*args,min_valid=5)['state']=='reused'
    with pytest.raises(ValueError,match='configuration/code changed'):
        build(*args,min_valid=6)
    write(run/'config.json',dict(changed=True))
    with pytest.raises(ValueError,match='source changed'):
        validate(out)


def test_reject_corrupt_export_and_never_publish_partial(indexed_sources):
    run,export,out=indexed_sources
    p=export/'shard_000/research/episode_000000/steps.npz'
    with p.open('ab') as f:f.write(b'corrupted')
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        build(run,[export],out,'policy-a','round0')
    assert not (out/'manifest.json').exists()


def test_missing_annotation_does_not_silently_drop_episode(indexed_sources):
    run,export,out=indexed_sources
    next((run/'episodes').glob('*/annotation.json')).unlink()
    with pytest.raises(FileNotFoundError):
        build(run,[export],out,'policy-a','round0')
    assert not (out/'manifest.json').exists()


def test_v3_extraction_commands_are_accepted_without_state_relabeling(indexed_sources):
    run,export,out=indexed_sources
    side=export/'shard_000/research/episode_000000'
    manifest=read(side/'manifest.json');manifest['schema']='robodojo_training_extract.v3'
    write(side/'manifest.json',manifest)
    report=build(run,[export],out,'policy-a','round0',min_valid=5)
    assert report['episodes']==1
    assert validate(out)['samples']==2
    with sqlite3.connect(out/'index.sqlite') as db:
        assert db.execute('SELECT DISTINCT action_array FROM training_samples').fetchall()==[('command',)]


def test_online_annotation_route_index_dispatch(indexed_sources):
    from data_processing.online_subtask_outcomes.prepare import prepare_batch as online_prepare_batch, prepare as online_prepare
    from data_processing.online_subtask_outcomes.contract import validate_final as online_validate
    from data_processing.online_subtask_outcomes.pipeline import protocol as online_protocol
    from data_processing.tests.test_online_subtask_outcomes import online, candidate as online_candidate, audit
    old_run,export,out=indexed_sources
    old_manifest=read(old_run.parent.parent/'manifest.json')
    archive=Path(old_manifest['episodes'][0]['archive'])
    online.__wrapped__(archive)
    # Match existing exported task text exactly.
    h=read(archive/'controller/history.json')
    for r in h:
        c=r['conditioning'];c['episode_instruction']='Move the object'
        c['effective_prompt']='Move the object'+('' if c['subtask_text'] is None else '\nCurrent subtask: '+c['subtask_text'])
    write(archive/'controller/history.json',h)
    write(archive/'sim/reset.json',dict(episode_id='test-episode',instruction='Move the object'))
    batch=old_run.parent.parent.parent/'online'
    m=online_prepare_batch([archive],batch);run=batch/'runs/new'
    write(run/'protocol.json',online_protocol());write(run/'config.json',load_config())
    dest=run/'episodes'/m['episodes'][0]['key']
    c=online_candidate();a=audit()
    write(dest/'candidate.json',c);write(dest/'audit_final.json',a)
    write(dest/'annotation.json',online_validate(online_prepare(archive),c,a))
    report=build(run,[export],out,'round2','2',min_valid=5)
    assert report['windows']==2 and report['gpt_valid_steps']==6
    with sqlite3.connect(out/'index.sqlite') as db:
        assert db.execute('select annotation_version from episodes').fetchone()[0]=='online_subtask_outcomes.v1'
