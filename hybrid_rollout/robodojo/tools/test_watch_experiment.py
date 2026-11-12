import json
from .watch_experiment import Progress, latest_batch, render, publish


def test_incremental_and_partial(tmp_path):
    log = tmp_path/'controller.log'
    event = json.dumps(dict(event='tool_done', step_id=12, counters=dict(student_steps=12)))
    log.write_text('noise\n' + event[:20])
    p = Progress();p.update(log)
    assert p.step is None and p.offset == 6
    with log.open('a') as f:
        f.write(event[20:]+'\n')
    p.update(log)
    assert p.step == 12 and p.counters['student_steps'] == 12
    offset = p.offset;p.update(log)
    assert p.offset == offset
    log.write_text('{}\n');p.update(log)
    assert p.step is None


def test_render_and_batch(tmp_path):
    archive = tmp_path/'case';(archive/'controller').mkdir(parents=True)
    (archive/'controller/run.json').write_text('{"max_episode_steps":1050}')
    row = dict(index=3, archive=str(archive), variant='random', layout_id=0,
        state='starting', started_utc='2026-10-03T00:00:00+00:00')
    batch = dict(state='running', planned_rollouts=5, episodes=[row])
    path = tmp_path/'_replicas/replica_3/attempt_0/batch.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(batch))
    assert latest_batch(tmp_path)[0] == path
    p = Progress();p.step=620;p.mode='eef';p.goal='抓取数字'
    p.counters=dict(student_steps=450, edited_steps=20, recovery_steps=150)
    message = render('test', batch, row, p)
    assert 'case 4/5' in message and '620/1050' in message
    assert 'GPT修正 170步' in message and '抓取数字' in message


def test_publish_utf8_and_auth(monkeypatch):
    from . import watch_experiment as mod
    class Reply:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):return b'{}'
    captured=[]
    def fake(request,timeout):
        captured.append(request);return Reply()
    monkeypatch.setattr(mod,'urlopen',fake)
    publish(dict(url='https://example.com/topic',token='secret'), 'a'+'中'*2000)
    assert len(captured[0].data)<=3900
    captured[0].data.decode('utf-8')
    assert captured[0].get_header('Authorization')=='Bearer secret'
