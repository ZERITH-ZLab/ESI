"""Read-only collection inventory and bounded evidence/image extraction."""
import json
from pathlib import Path
from .bundle import digest


def read(path):
    return json.loads(Path(path).read_text())


def inventory(suite, cfg):
    from ..account_pool import collection_completed
    from ..collection_suite import selected_cases
    shared = suite.parent.parent
    selected, missing, anomalies = [], [], []
    for task in cfg['tasks']:
        panel = read(suite/'panels'/f'{task}.json')
        root = shared/'results'/(cfg['name']+'__'+task)
        replica = root/'_replicas'/f"replica_{panel['cases'][0]['replica_id']}"
        batches = sorted(replica.glob('attempt_*/batch.json'),
                         key=lambda p: int(p.parent.name.split('_')[-1]))
        if not batches:
            missing.extend(c['case_id'] for c in selected_cases(panel, cfg));continue
        runtime = read(batches[-1]).get('resume_identity')
        for case in selected_cases(panel, cfg):
            row = collection_completed(replica, runtime, case['case_id'], 1)
            if row is None:
                missing.append(case['case_id']);continue
            archive = Path(row['archive'])
            if not archive.resolve().is_relative_to(root.resolve()):
                raise ValueError('Collection receipt points outside its experiment')
            hist = archive/'controller/history.json'
            if not hist.is_file():
                anomalies.append(dict(case_id=case['case_id'], reason='missing_history', archive=str(archive)))
                continue
            history = read(hist)
            actual_identity = read(archive/'resume_identity.json')
            if actual_identity.get('eval_protocol'):
                raise ValueError('Official evaluation data cannot drive prompt evolution')
            result = read(archive/'controller/result.json') if (archive/'controller/result.json').exists() else {}
            selected.append(dict(key=digest([task, case['case_id']])[:20], task=task,
                case_id=case['case_id'], split=case['variant'], archive=str(archive),
                history_sha256=digest(history), source_identity=actual_identity,
                result={k:result.get(k) for k in ('success','complete','reason','step_id')},
                failure_kind=row.get('failure_kind'), receipt_state=row.get('state')))
    return dict(episodes=selected, missing_cases=missing, anomalies=anomalies,
                selection='existing collection_completed receipt rule; never choose best score')


def ledger(row):
    archive = Path(row['archive']); history = read(archive/'controller/history.json')
    if digest(history) != row['history_sha256']:
        raise ValueError('History changed after inventory freeze')
    records=[]
    for h in history:
        response=h.get('response') or {}; conditioning=h.get('conditioning') or {}
        motion=h.get('motion_result') or {}; correction=h.get('correction_result') or {}
        records.append(dict(decision=h['decision'], start_tick=h.get('start_tick'),
            end_tick=h.get('end_tick'), executed_steps=h.get('executed_steps'),
            controller=response.get('mode'), assessment=response.get('assessment', {}),
            reason=response.get('reason'), subtask=conditioning.get('subtask_text'),
            current_intent=conditioning.get('current_intent'),
            motion={k:motion.get(k) for k in ('tool','status','arm','executed_steps')},
            correction={k:correction.get(k) for k in ('status','executed_steps')},
            automatic_hold_steps=h.get('force_hold_steps',0),
            privileged_history_exposed=h.get('privileged_history_exposed',False)))
    first=next(iter(sorted((archive/'controller/observations').glob('*/observation.json'))),None)
    instruction=read(first).get('instruction') if first else None
    # No scene truth, native object poses, or unbounded RPC payloads enter the model.
    return dict(task=row['task'], case_id=row['case_id'], instruction=instruction,
                result=row['result'], failure_kind=row['failure_kind'], records=records)


def chunks(records, max_chars):
    """Chronological chunks, with one overlapping context record; never truncate evidence."""
    current=[];size=0
    for row in records:
        n=len(json.dumps(row,ensure_ascii=False))
        if n>max_chars:raise ValueError('Single decision exceeds input budget')
        if current and size+n>max_chars:
            yield current
            previous=current[-1]
            current=[previous] if len(json.dumps(previous))+n<=max_chars else []
            size=sum(len(json.dumps(r,ensure_ascii=False)) for r in current)
        current.append(row);size+=n
    if current:yield current


def load_images(row, requests, cfg, destination, seen, remaining, available_decisions):
    """Only requested observations, bounded sampling; no arbitrary model-chosen paths."""
    from PIL import Image
    root=Path(row['archive'])/'controller/observations';files=[];manifest=[];denied=[]
    allowed={'cam_high','cam_left_wrist','cam_right_wrist'}
    for request in requests:
        start,end=request['start_decision'],request['end_decision'];mode=request['sampling']
        if start not in available_decisions or end not in available_decisions or start>end:
            denied.append(dict(request=request,reason='invalid decision range'));continue
        if not request['question'].strip() or request['camera'] not in allowed:
            denied.append(dict(request=request,reason='missing question or invalid view'));continue
        if mode=='dense' and end-start>2*cfg['dense_decision_radius']:
            denied.append(dict(request=request,reason='dense range exceeds local event budget'));continue
        if mode=='sparse':
            ids=sorted(set([start,end]+list(range(start,end+1,cfg['sparse_decision_stride']))))
        else:ids=list(range(start,end+1))
        for decision in ids:
            key=f'{decision}:{request["camera"]}'
            if key in seen:continue
            if len(files)>=min(cfg['max_images_per_call'],remaining):
                denied.append(dict(request=request,reason='image budget exhausted'));break
            source=root/f'{decision:03}'/(request['camera']+'.png')
            meta=source.parent/'observation.json'
            if not source.is_file() or not meta.is_file():
                denied.append(dict(decision=decision,reason='observation missing'));continue
            if not source.resolve().is_relative_to(root.resolve()):raise ValueError('Image escaped observation root')
            destination.mkdir(parents=True,exist_ok=True)
            target=destination/(key.replace(':','_')+'.jpg')
            with Image.open(source) as im:
                im=im.convert('RGB');im.thumbnail((cfg['image_max_edge'],)*2);im.save(target,quality=85)
            seen.add(key);files.append(target)
            manifest.append(dict(image_id=key, decision=decision, step_id=read(meta).get('step_id'),
                camera=request['camera'],question=request['question'],source=str(source),image=str(target)))
    return files,manifest,denied
