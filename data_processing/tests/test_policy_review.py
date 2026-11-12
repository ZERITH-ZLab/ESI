import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from data_processing.policy_review import server


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    a=tmp_path/'results/demo/standard/layout_9/repeat_001/attempt_000'
    dump(a/'sim/summary.json',dict(episode_id='ep',step_id=3,video_frames=4,control_dt=.04,
        success=False,terminated=False,truncated=True,
        evaluation_case=dict(variant='standard',layout_id=9,policy_rng_seed=0)))
    c=dict(subtask=None,subtask_text=None,episode_instruction='Task.',effective_prompt='Task.',
           episode_id='ep',step_id=0,prediction_id='p0')
    def row(i,start,end,condition,mode='student'):
        value=dict(decision=i,start_tick=start,end_tick=end,executed_steps=end-start,
            prediction_id=f'p{i}',response=dict(mode=mode,reason='<script>test</script>',assessment={}))
        if condition is not False:value['conditioning']=condition
        return value
    sub=dict(c,subtask={'skill_type':'grasp','arm':'left'},subtask_text='Left arm: Grasp digit 1.',
             effective_prompt='Task.\nCurrent subtask: Left arm: Grasp digit 1.',step_id=2,prediction_id='p1')
    rows=[row(0,0,2,c),row(1,2,2,sub,'eef'),row(2,2,3,dict(sub,prediction_id='p2'))]
    rows[1]['correction_result']={'status':'planning_failed'}
    dump(a/'controller/history.json',rows)
    dump(a/'controller/controller_protocol.json',{'controller_route':'subtask_feedforward_v1'})
    (a/'sim/sensors.mp4').write_bytes(b'0123456789')
    monkeypatch.setattr(server,'probe_video',lambda p:dict(frames=4,fps=25,width=1920,height=360))
    return a,rows


def test_scan_and_conditioning_distinctions(tmp_path,monkeypatch):
    a,rows=fixture(tmp_path,monkeypatch);s=server.Store([tmp_path],[a])
    catalog=s.scan([tmp_path]);assert len(catalog['episodes'])==1
    key=catalog['episodes'][0]['id'];d=s.episode(key)
    assert d['counts']=={'null':1,'subtask':2} and d['zero_steps']==1 and not d['issues']
    assert d['decisions'][1]['start']==d['decisions'][1]['end']==2
    assert d['decisions'][1]['failure']=='planning_failed'
    assert d['decisions'][2]['conditioning']['effective_prompt'].endswith('Grasp digit 1.')
    del rows[0]['conditioning'];dump(a/'controller/history.json',rows)
    d=s.episode(key);assert d['decisions'][0]['state']=='missing' and d['issues']
    for r in rows:r.pop('conditioning',None)
    dump(a/'controller/history.json',rows);dump(a/'controller/controller_protocol.json',{})
    d=s.episode(key);assert d['counts']=={'legacy':3} and not d['issues']


def test_freeform_ablation_conditioning(tmp_path,monkeypatch):
    a,rows=fixture(tmp_path,monkeypatch)
    for row in rows[1:]:
        c=row['conditioning']
        c.update(subtask=c['subtask_text'],subtask_format='free_text_v1')
    dump(a/'controller/history.json',rows)
    s=server.Store([tmp_path],[a]);key=s.scan([a])['episodes'][0]['id']
    d=s.episode(key)
    assert not d['issues'] and d['decisions'][1]['text']=='Left arm: Grasp digit 1.'
    assert d['decisions'][1]['skill'] is None
    assert d['decisions'][1]['arm'] is None
    assert d['decisions'][1]['instruction_format']=='free_text'
    assert d['instruction_formats']=={'free_text':2}
    client=TestClient(server.create_app([tmp_path],[a]))
    key=client.get('/api/scan').json()['episodes'][0]['id']
    response=client.get('/api/episode/'+key)
    assert response.status_code==200
    assert response.json()['decisions'][1]['conditioning']==rows[1]['conditioning']


def test_mixed_freeform_structured_and_null(tmp_path,monkeypatch):
    a,rows=fixture(tmp_path,monkeypatch)
    c=rows[1]['conditioning'];c.update(subtask=c['subtask_text'],subtask_format='free_text_v1')
    dump(a/'controller/history.json',rows)
    s=server.Store([tmp_path],[a]);key=s.scan([a])['episodes'][0]['id'];d=s.episode(key)
    assert d['counts']=={'null':1,'subtask':2} and not d['issues']
    assert d['instruction_formats']=={'free_text':1,'structured':1}
    assert [x['instruction_format'] for x in d['decisions']]==[None,'free_text','structured']


def test_reject_misaligned_video_and_history(tmp_path,monkeypatch):
    a,rows=fixture(tmp_path,monkeypatch);s=server.Store([tmp_path],[a]);key=s.scan([a])['episodes'][0]['id']
    monkeypatch.setattr(server,'probe_video',lambda p:dict(frames=3,fps=25))
    with pytest.raises(ValueError,match='视频帧数'):s.episode(key)
    rows[2]['start_tick']=1;dump(a/'controller/history.json',rows)
    with pytest.raises(ValueError,match='Non-contiguous'):s.episode(key)


def test_identity_issue_not_silently_corrected(tmp_path,monkeypatch):
    a,rows=fixture(tmp_path,monkeypatch);rows[2]['conditioning']['prediction_id']='stale'
    rows[2]['conditioning']['effective_prompt']='Actual wrong prompt'
    dump(a/'controller/history.json',rows);s=server.Store([tmp_path],[a]);key=s.scan([a])['episodes'][0]['id']
    d=s.episode(key);assert len(d['issues'])==2
    assert d['decisions'][2]['conditioning']['effective_prompt']=='Actual wrong prompt'


def test_api_range_and_no_escape(tmp_path,monkeypatch):
    a,_=fixture(tmp_path,monkeypatch);app=server.create_app([tmp_path],[a]);c=TestClient(app)
    key=c.get('/api/scan').json()['episodes'][0]['id']
    assert c.get('/api/episode/'+key).status_code==200
    assert c.get('/api/episode/not-found').status_code==404
    assert c.get('/api/scan',params={'path':'/etc'}).status_code==400
    for header,expected in [('bytes=2-5',b'2345'),('bytes=-2',b'89'),('bytes=8-',b'89')]:
        r=c.get('/api/video/'+key,headers={'Range':header})
        assert r.status_code==206 and r.content==expected
        assert int(r.headers['Content-Length'])==len(expected)
    assert c.get('/api/video/'+key,headers={'Range':'bytes=20-'}).status_code==416
    assert c.get('/api/video/'+key,headers={'Range':'bytes=0-1,4-5'}).status_code==416
    assert c.head('/api/video/'+key).content==b''
    assert c.get('/api/video/'+key).content==b'0123456789'
    assert c.get('/').status_code==200 and c.get('/static/app.mjs').status_code==200
    assert c.get('/static/../server.py').status_code==404
    (a/'sim/sensors.mp4').unlink();(a/'sim/sensors.mp4').symlink_to('/etc/passwd')
    assert c.get('/api/video/'+key).status_code==404
    assert c.get('/api/episode/'+key).status_code==422


def test_duplicate_scans_and_snapshot_pruning(tmp_path,monkeypatch):
    a,_=fixture(tmp_path,monkeypatch)
    b=a/'source_snapshot/fake';dump(b/'sim/summary.json',{});dump(b/'controller/history.json',[])
    s=server.Store([tmp_path],[a]);r=s.scan([tmp_path,a]);assert len(r['episodes'])==1


def test_online_annotation_scan_overlay_and_stale_rejection(tmp_path,monkeypatch):
    import hashlib
    a,rows=fixture(tmp_path,monkeypatch)
    dest=tmp_path/'annotations/batch/runs/run1/episodes/key1'
    segments=[dict(decision_id=r['decision'],start_step=r['start_tick'],end_step=r['end_tick'],
        online_conditioning=r['conditioning'],outcome='success',outcome_basis='inferred',
        outcome_rationale='Later feedback confirms completion',needs_review=False,attempt_id=0,
        training_eligible=True,outcome_evidence=[]) for r in rows if r['executed_steps']]
    ann=dict(route='online_subtask_outcome',version='online_subtask_outcomes.v1',segments=segments,
        metadata=dict(source_archive=str(a),episode_id='ep',source_sha256={
            'controller/history.json':hashlib.sha256((a/'controller/history.json').read_bytes()).hexdigest()}))
    dump(dest/'annotation.json',ann)
    store=server.Store([tmp_path],[dest.parent.parent])
    entries=store.scan([tmp_path/'annotations'])['episodes']
    assert len(entries)==1 and entries[0]['experiment']=='batch / run1'
    data=store.episode(entries[0]['id'])
    assert data['annotation_path']==str(dest/'annotation.json')
    assert data['decisions'][0]['annotation']['outcome']=='success'
    assert 'annotation' not in data['decisions'][1]  # zero-step rejection
    assert data['decisions'][0]['conditioning']['subtask'] is None
    ann['segments'][0]['end_step']=100;dump(dest/'annotation.json',ann)
    with pytest.raises(ValueError,match='区间'):store.episode(entries[0]['id'])
    ann['segments'][0]['end_step']=2;dump(dest/'annotation.json',ann)
    rows[0]['response']['reason']='changed';dump(a/'controller/history.json',rows)
    with pytest.raises(ValueError,match='原数据已变化'):store.episode(entries[0]['id'])


def test_failed_annotation_shows_raw_not_draft(tmp_path,monkeypatch):
    a,_=fixture(tmp_path,monkeypatch)
    batch=tmp_path/'batch';dest=batch/'runs/run1/episodes/key1'
    dump(batch/'manifest.json',dict(episodes=[dict(key='key1',archive=str(a))]))
    dump(dest/'status.json',dict(state='failed',error='network failure'))
    dump(dest/'draft_00.json',dict(segments=[dict(outcome='success')]))
    store=server.Store([tmp_path],[batch]);entries=store.scan([batch])['episodes']
    assert len(entries)==1 and entries[0]['annotation_status']=='failed'
    data=store.episode(entries[0]['id'])
    assert not any(d.get('annotation') for d in data['decisions'])
    assert 'network failure' in data['issues'][-1]
