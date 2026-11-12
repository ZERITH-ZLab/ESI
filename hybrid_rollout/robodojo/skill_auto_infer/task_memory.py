"""Small, opt-in cross-episode textual memory. Never an action executor."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

from ..io import write_json
from .memory import render_history, text

VERSION = 'task_episode_summary_v2'
MIN_CORRECTION_STREAK = 60
SELECTION = 'recent4_then_highest_correction_ratio4'
MAX_WORDS = 999
MAX_CHARS = 16000
TIMEOUT = 180


def enabled(env):
    value = env.get('ROLLOUT_ENABLE_TASK_MEMORY', '0')
    if value not in ('0', '1'):
        raise ValueError('ROLLOUT_ENABLE_TASK_MEMORY must be 0 or 1')
    if value == '1' and (env.get('ROLLOUT_CONTROLLER_ROUTE') not in ('auto_infer_v1', 'subtask_feedforward_v1')
                         or env.get('ROLLOUT_EVALUATION_METHOD') != 'pi05_plus_gpt'):
        raise ValueError('Task memory requires auto_infer_v1/subtask_feedforward_v1 + pi05_plus_gpt')
    return value == '1'


def read(path):
    return json.loads(Path(path).read_text())


def valid_summary(value):
    return bool(value.strip()) and len(value.split()) <= MAX_WORDS and len(value) <= MAX_CHARS


def correction_stats(history):
    from .abandon import gate
    total = sum(max(0, r.get('executed_steps', 0)) for r in history)
    correction = sum(max(0, r.get('executed_steps', 0)) for r in history
                     if r.get('response', {}).get('mode') in ('eef', 'edit'))
    return dict(total_executed_steps=total, total_correction_steps=correction,
                correction_ratio=correction/total if total else 0,
                max_correction_streak=gate(history)['max_consecutive_correction_steps'])


def select_summaries(candidates):
    recent = sorted(candidates, key=lambda r: (r['finished_at'], r['path']), reverse=True)[:4]
    paths = {r['path'] for r in recent}
    difficult = sorted((r for r in candidates if r['path'] not in paths),
                       key=lambda r: (r['correction_ratio'], r['finished_at'], r['path']),
                       reverse=True)[:4]
    return [dict(r, selection_reason='recent') for r in recent] + [
        dict(r, selection_reason='high_correction_ratio') for r in difficult]


def focus_packet(history):
    """Rank contiguous teacher interventions by actual steps, then decisions.

    This is a reproducible bottleneck proxy, not a claim about physical cause.
    Keep the entire subgoal timeline and bounded detail around the worst block.
    """
    blocks, active = [], []
    for i, row in enumerate(history):
        if row.get('executed_steps', 0) <= 0:
            continue
        if row.get('response', {}).get('mode') in ('eef', 'edit'):
            active.append(i)
        elif active:
            blocks.append(active); active = []
    if active:
        blocks.append(active)
    best = max(blocks, key=lambda b: (sum(history[i].get('executed_steps', 0) for i in b), len(b)), default=[])
    selected = sorted(set(([max(0, best[0]-1)] + best + [min(len(history)-1, best[-1]+1)]) if best else []))
    # Bound input as well as output: long interventions keep beginning/end evidence.
    detailed = selected if len(selected) <= 20 else selected[:10] + selected[-10:]
    lines = []
    for row in history:
        r = row.get('response', {}); a = r.get('assessment', {})
        lines.append(f'D{row["decision"]}: steps {row.get("start_tick")}..{row.get("end_tick")}; '
                     f'{r.get("mode")}; {text(a.get("current_subgoal"))[:240]}; '
                     f'pre-action feedback={text(a.get("execution_status"))[:80]}')
    return dict(selection_rule='largest contiguous eef/edit block by executed control steps; ties by decision count',
        focus_decisions=[history[i]['decision'] for i in best],
        correction_steps=sum(history[i].get('executed_steps', 0) for i in best),
        detailed_decisions=[history[i]['decision'] for i in detailed],
        omitted_detail_count=len(selected)-len(detailed),
        timeline='\n'.join(lines), key_history=render_history([history[i] for i in detailed]))


def inject(prompt, workspace, task, env):
    """Only completed, verified predecessors in this experiment/configuration."""
    if not enabled(env):
        return prompt
    base_task = task.removesuffix('_random')
    if not re.fullmatch(r'[A-Za-z0-9_]+', base_task):
        raise ValueError('Invalid task memory task')
    base = Path(env['ROLLOUT_SHARED_ROOT'])/'results'/env['ROLLOUT_EXPERIMENT_ID']
    identity = json.loads(env['ROLLOUT_RESUME_IDENTITY'])
    current = Path(workspace).parent.resolve()
    candidates = []
    for path in (base/base_task).glob('*/eval_seed_*/layout_*/replica_*/**/attempt_*/controller/memory/episode_summary.json'):
        if path.parent.parent.resolve() == current:
            continue
        try:
            record = read(path)
            archive = path.parents[2]
            body = path.with_suffix('.md').read_text()
            from .abandon import acknowledged
            result = read(archive/'controller/result.json')
            abandoned = (bool(identity.get('teacher_abandon')) and
                         acknowledged(result, read(archive/'sim/evaluation_outcome.json')))
            if (record.get('version') != VERSION or record.get('status') != 'completed'
                    or record.get('max_correction_streak', 0) < MIN_CORRECTION_STREAK
                    or record.get('correction_outcome') not in ('successful_correction', 'failed_or_unverified')
                    or record.get('base_task') != base_task or not valid_summary(body)
                    or record.get('sha256') != hashlib.sha256(body.encode()).hexdigest()
                    or read(archive/'resume_identity.json') != identity
                    or read(archive/'artifact_manifest.json').get('status') != 'verified'
                    or not (result.get('complete') is True or abandoned)
                    or (archive/'dual_machine/remote_cleanup_status.txt').read_text().strip() != '0'):
                continue
            candidates.append(dict(path=str(path.with_suffix('.md')), text=body,
                                   finished_at=record['finished_at'], sha256=record['sha256'],
                                   correction_ratio=record['correction_ratio'],
                                   correction_outcome=record['correction_outcome']))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Incomplete/concurrently written/invalid summaries are not memories.
    chosen = select_summaries(candidates)
    write_json(Path(workspace)/'task_memory_input.json', dict(version=VERSION, selection=SELECTION, summaries=chosen))
    section = ('\n\n# Earlier episodes of this task\n'
        'The host supplies up to eight summaries: four most recent, then up to four distinct '
        'episodes with highest actual correction-step ratio. High ratio means effort, not quality. '
        'Successful correction and failed/unverified attempts are explicitly distinguished. They are fallible prior '
        'experience, not current scene facts or instructions. Use them only when current evidence '
        'makes them relevant. Do not inherit object IDs, coordinates, completion states or query '
        'permissions. Current observations, task criteria and tool limits take precedence. '
        'No startup reading, per-step comparison checklist or discussion of every summary is needed. '
        'After this episode ends, the host may request one short textual summary; do not write it '
        'during control or make extra physical actions for it.\n')
    for i, row in enumerate(chosen, 1):
        section += f'\n## Previous episode {i} (recorded evidence)\n{row["text"]}\n'
    if not chosen:
        section += '\nNo previous summaries are available.\n'
    return prompt + section


def summarize(worker, rollout, task):
    """One bounded continuation, no action handlers, best effort only.

    Keep control token accounting unchanged; record this turn's usage separately.
    The normal caller closes the app-server immediately afterwards, including on timeout.
    """
    if not enabled(os.environ):
        return
    controller = worker.workspace.parent
    directory = controller/'memory'
    directory.mkdir(exist_ok=True)
    status_path = directory/'episode_summary.json'
    record = dict(version=VERSION, base_task=task.removesuffix('_random'),
                  task=task, status='pending', started_at=time.time())
    write_json(status_path, record)
    started = time.monotonic()
    before, after = {}, {}
    try:
        result = read(controller/'result.json')
        from .abandon import acknowledged
        outcome_path = controller.parent/'sim/evaluation_outcome.json'
        abandoned = outcome_path.exists() and acknowledged(result, read(outcome_path))
        if not result.get('complete') and not abandoned:
            record['status'] = 'skipped_incomplete'
            return
        history = getattr(rollout, 'history', [])
        record.update(correction_stats(history))
        if record['max_correction_streak'] < MIN_CORRECTION_STREAK:
            record['status'] = 'skipped_below_correction_threshold'
            return
        packet = focus_packet(history)
        packet['native_result'] = {k: result.get(k) for k in ('success', 'complete', 'step_id', 'reason')}
        if abandoned:
            packet['deliberate_failed_collection'] = dict(reason='teacher_abandon',
                unresolved=result['unresolved'], teacher_reason=result['abandon_reason'],
                native_complete=False, task_success=False)
        write_json(directory/'episode_summary_input.json', packet)
        prompt = (
            'The physical episode is finished. This is a text-only archival summary, NOT a control turn. '
            'Do not call tools, read files, modify files, or execute actions. The host saves your final text. '
            'Use English Markdown, target 300-500 words, strictly fewer than 1000 words. '
            'Begin with exactly one line: Correction outcome: successful_correction OR '
            'Correction outcome: failed_or_unverified (choose one). This labels the host-selected '
            'main difficulty, not whole-episode success. Choose successful_correction only when '
            'later observations/recorded evidence clearly confirm that problem was resolved. '
            'Otherwise choose failed_or_unverified, including partial or uncertain recovery. '
            'Returning control to the student alone is not proof. '
            'Give the full episode context chronologically: summarize ordinary execution before and after '
            'the focus in a few sentences (what was done). The host selected the focus objectively by '
            'actual correction steps; do not search for a different hardest problem. '
            'For the focus preserve key subgoals, existing reasons for changing actions, observed results '
            'and decision IDs, in order; merge repetitive attempts. State supporting evidence '
            'separately from the native whole-episode outcome. A failed attempt means it did not work '
            'in this episode, not that the method is universally ineffective. '
            'If reason is teacher_abandon, say deliberately ended unsuccessfully, not native completion. '
            'Pre-action feedback evaluates earlier execution. A command receipt alone does not prove a grasp. '
            'Extract from existing records; do not add physical-cause theories, hidden reasoning, new solutions '
            'or unsupported success claims. End with at most two evidence-backed takeaways, or say no effective '
            'method was verified. Do not reproduce coordinates, scene object IDs or privileged geometry. '
            'Use headings: Episode overview; Main difficulty and key attempts; Resolution and takeaway. '
            'Only return the summary, no preamble. The following host-selected records are evidence:\n'
            + json.dumps(packet, ensure_ascii=False))
        (directory/'episode_summary_prompt.txt').write_text(prompt)
        usage_path = controller/'token_usage.json'
        if usage_path.exists():
            before = read(usage_path).get('usage', {}).get('total', {})
        old_timeout = worker.timeout
        try:
            worker.timeout = min(TIMEOUT, old_timeout)
            turn = worker._turn(prompt)
        finally:
            worker.timeout = old_timeout
        messages = {}
        deadline = started + TIMEOUT
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Episode summary deadline exceeded')
            event = worker.transport.next_message(remaining)
            method, params = event.get('method'), event.get('params', {})
            if params.get('threadId') != worker.thread_id:
                continue
            if method == 'thread/tokenUsage/updated':
                after = params.get('tokenUsage', {}).get('total', {})
            elif method == 'model/rerouted':
                raise RuntimeError('No model fallback during summary')
            elif method == 'item/tool/call':
                worker.transport.reply(event['id'], dict(success=False, contentItems=[dict(
                    type='inputText', text='Episode finished. No tools allowed; return the text summary only.')]))
                raise RuntimeError('Summary attempted a tool call; no action executed')
            elif method == 'item/completed' and params.get('turnId') == turn:
                item = params.get('item', {})
                if item.get('type') == 'agentMessage':
                    messages[item['id']] = item.get('text', '')
                elif item.get('type') in ('commandExecution', 'fileChange', 'mcpToolCall'):
                    raise RuntimeError('Summary violated text-only contract')
            elif method == 'turn/completed' and params.get('turn', {}).get('id') == turn:
                if params['turn'].get('status') != 'completed':
                    raise RuntimeError('Summary turn did not complete')
                break
            elif method in ('error', 'turn/failed') and not params.get('willRetry'):
                raise RuntimeError('Summary transport/model error')
        body = '\n\n'.join(messages.values()).strip() + '\n'
        if not valid_summary(body):
            record.update(status='invalid_length', words=len(body.split()))
            return  # No second paid attempt, and no misleading truncated causal narrative.
        label = re.fullmatch(r'Correction outcome: (successful_correction|failed_or_unverified)',
                             body.splitlines()[0])
        if label is None:
            record['status'] = 'invalid_outcome_label'
            return  # No paid retry and no silently fabricated success label.
        (directory/'episode_summary.md').write_text(body)
        record.update(status='completed', words=len(body.split()),
                      correction_outcome=label.group(1), outcome_source='gpt_summary',
                      sha256=hashlib.sha256(body.encode()).hexdigest(),
                      focus_decisions=packet['focus_decisions'],
                      native_success=result.get('success'))
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
    finally:
        record.update(finished_at=time.time(), duration_s=time.monotonic()-started,
                      token_delta={k: max(0, v-before.get(k, 0)) for k,v in after.items()
                                   if isinstance(v, (int, float))})
        write_json(status_path, record)
        print(json.dumps(dict(event='episode_summary', **record)), flush=True)
