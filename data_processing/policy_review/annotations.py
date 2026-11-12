"""Optional online-outcome overlay. Failed drafts never masquerade as final labels."""
import hashlib
from pathlib import Path
from .server import read


def entry_from_annotation(store, directory):
    directory = store.permitted(directory)
    annotation = directory / 'annotation.json'
    ann = read(annotation, {})
    status = read(directory / 'status.json', {})
    run = directory.parent.parent
    batch = run.parent.parent
    manifest = read(store.permitted(batch / 'manifest.json'), {})
    row = next((r for r in manifest.get('episodes', []) if r['key'] == directory.name), {})
    if ann and ann.get('route') != 'online_subtask_outcome':
        raise ValueError('旧版离线分段标注请用原标注查看器（18790）；此处接入在线结果标注')
    archive = store.permitted(ann.get('metadata', {}).get('source_archive') or row['archive'])
    summary = read(store.file(archive, 'sim/summary.json'))
    case = summary.get('evaluation_case') or {}
    repeat = case.get('policy_rng_seed')
    return dict(id=hashlib.sha256(str(directory).encode()).hexdigest()[:24],
        experiment=f'{batch.name} / {run.name}', archive=str(archive),
        variant=case.get('variant', 'unknown'), layout=case.get('layout_id'),
        repeat=repeat+1 if isinstance(repeat,int) else None, attempt=archive.name,
        success=summary.get('success'), complete=bool(summary.get('terminated') or summary.get('truncated')),
        steps=summary.get('step_id'), annotation=str(annotation), annotation_status=status.get('state', 'annotated' if ann else 'missing'),
        annotation_error=status.get('error'))


def attach_annotation(store, data):
    entry = data['entry']
    path = store.permitted(entry['annotation'])
    ann = read(path)
    if not ann:
        data['issues'].append('无最终标注；不显示 draft/candidate 的结果。' + (entry.get('annotation_error') or ''))
        return data
    if ann.get('route') != 'online_subtask_outcome':
        raise ValueError('不支持的标注路线')
    for name, expected in ann['metadata']['source_sha256'].items():
        source = store.file(entry['archive'], name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('标注对应原数据已变化，拒绝显示过期结果')
    labels = {s['decision_id']: s for s in ann['segments']}
    expected = {d['id'] for d in data['decisions'] if d['executed_steps'] > 0}
    if set(labels) != expected or len(labels) != len(ann['segments']):
        raise ValueError('标注与执行决策覆盖不一致')
    for d in data['decisions']:
        s = labels.get(d['id'])
        if s is None:
            continue
        if (s['start_step'], s['end_step'], s['online_conditioning']) != (d['start'], d['end'], d['conditioning']):
            raise ValueError('标注区间或在线指令与原数据不一致')
        d['annotation'] = s
    data['annotation_path'] = str(path)
    data['annotation_version'] = ann['version']
    return data
