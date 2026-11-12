"""Native-budget, non-privileged hybrid regression evaluation (5 cases x 3 seeds).

These cases were selected from training layouts: this is a paired regression
panel, not an unseen benchmark. The collection route is independent.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .evaluation import canonical_sha256, file_record, layout_files, read_panel, verify_assets
from .io import require, write_json

SCHEMA = 'hybrid_rollout.robodojo.regression_eval_panel.v1'
VERSION = 'native_hybrid_eval_v1'
CASES = (('standard', 9), ('standard', 5), ('standard', 14), ('random', 4), ('random', 1))
NATIVE_BUDGET = 1050
REQUIRED_ENV = dict(ROLLOUT_STRICT_EVAL='1', ROLLOUT_ENABLE_SCENE_QUERY='0',
    ROLLOUT_ENABLE_TASK_MEMORY='0', ROLLOUT_ENABLE_TEACHER_ABANDON='0',
    ROLLOUT_COLLECTION_STEP_LIMIT='0', ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1',
    ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt', ROLLOUT_ENABLE_MOTION_TOOLS='1',
    ROLLOUT_ALLOW_SOURCE_UPGRADE='0', MAX_DECISIONS='0', CODEX_MAX_TOTAL_TOKENS='0')


def required_env(route='auto_infer_v1'):
    require(route in ('auto_infer_v1', 'subtask_feedforward_v1'), 'Unsupported eval controller route')
    return dict(REQUIRED_ENV, ROLLOUT_CONTROLLER_ROUTE=route)


def enabled(env=None):
    return (os.environ if env is None else env).get('ROLLOUT_STRICT_EVAL', '0') == '1'


def specs(seed):
    for index, (variant, layout) in enumerate(CASES):
        task = 'arrange_largest_number'
        yield dict(case_id=f'{task}__{variant}__g0__l{layout}', task=task,
            runtime_task=task + ('_random' if variant == 'random' else ''),
            variant=variant, replica_id=3, rollout_index=index, eval_seed=0,
            layout_id=layout, reset_seed=layout, simulator_initial_seed=0, policy_rng_seed=seed)


def queue_specs():
    """One shared queue; no synchronization barrier between policy seeds."""
    for seed in (0, 1, 2):
        for case in specs(seed):
            yield dict(case, case_id=f'{case["case_id"]}__pseed{seed}',
                       rollout_index=seed * len(CASES) + case['rollout_index'],
                       evaluation_repeat=seed + 1)


def validate(panel):
    if panel.get('task_eval_kind') == 'task_subset_v1':
        from .task_eval import validate_task_eval
        return validate_task_eval(panel)
    require(panel.get('schema') == SCHEMA, 'Not a regression eval panel')
    require(panel.get('panel_sha256') == canonical_sha256(
        {k: v for k, v in panel.items() if k != 'panel_sha256'}), 'Eval panel hash mismatch')
    seed = panel.get('policy_rng_seed')
    queued = panel.get('queue_mode') == 'all_policy_seeds_v1'
    if queued:
        require(seed is None, 'Unified queue uses per-case policy seeds')
    else:
        require('queue_mode' not in panel, 'Unknown eval queue mode')
        require(type(seed) is int and seed in (0, 1, 2), 'Expected paired policy RNG seed 0, 1 or 2')
    require(panel.get('protocol') == VERSION and panel.get('native_step_limit') == NATIVE_BUDGET,
            'Eval protocol/native budget changed')
    expected_cases = list(queue_specs() if queued else specs(seed))
    require(len(panel.get('cases', [])) == len(expected_cases), 'Wrong regression case count')
    for case, expected in zip(panel['cases'], expected_cases):
        require(all(case.get(k) == v for k, v in expected.items()), 'Eval case/seed/order changed')
        require(case['native_support_trajectories'] == [], 'Unexpected support trajectories')
        path = Path(case['layout']['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe layout path')
    return panel


def freeze(source, seed):
    source = Path(source).resolve()
    files = sorted({p for folder in ('env', 'env_cfg', 'task/RoboDojo', 'utils', 'src/eval_client')
        for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')})
    require(bool(files), 'Missing native source')
    cases = []
    for spec in specs(seed):
        layout = layout_files(source, spec['runtime_task'], 0)[spec['layout_id']]
        cases.append(dict(spec, layout=file_record(source, layout), native_support_trajectories=[]))
    panel = dict(schema=SCHEMA, panel_id=f'arrange_regression5_pseed{seed}_v1',
        protocol=VERSION, native_step_limit=NATIVE_BUDGET, policy_rng_seed=seed, eval_seed=0,
        scope='training_layout_regression_not_heldout',
        native_source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        native_source_files=[file_record(source, p) for p in files], cases=cases)
    panel['panel_sha256'] = canonical_sha256(panel)
    return validate(panel)


def freeze_queue(source):
    panel = freeze(source, 0)
    layouts = {(c['variant'], c['layout_id']): c['layout'] for c in panel['cases']}
    panel.update(panel_id='arrange_regression15_queue_v1', policy_rng_seed=None,
                 queue_mode='all_policy_seeds_v1', cases=[
                     dict(c, layout=layouts[c['variant'], c['layout_id']], native_support_trajectories=[])
                     for c in queue_specs()])
    panel.pop('panel_sha256')
    panel['panel_sha256'] = canonical_sha256(panel)
    return validate(panel)


def runtime_settings(env, panel):
    if panel.get('schema') != SCHEMA:
        require(not enabled(env), 'Strict eval requires its own frozen eval panel')
        return {}
    validate(panel)
    from .prompt_profile import identity as prompt_identity
    prompt_settings = prompt_identity(env)
    from .placement_order import identity as order_identity
    prompt_settings.update(order_identity(env, panel))
    for name, value in required_env(env.get('ROLLOUT_CONTROLLER_ROUTE')).items():
        require(env.get(name) == value, f'Eval requires {name}={value}')
    require(not env.get('ROLLOUT_WORKER_CONTROL_FILE'), 'Eval cannot use the collection worker scheduler')
    workers = int(env.get('ROLLOUT_PARALLEL_WORKERS', '1'))
    require(1 <= workers <= 4, 'Eval supports 1..4 workers')
    require(workers == 1 or bool(env.get('ROLLOUT_ACCOUNT_POOL_FILE')), 'Parallel eval requires an account pool')
    require(env.get('ROLLOUT_RUN_MODE') in ('fresh', 'resume'), 'Eval supports fresh/resume only')
    if panel.get('queue_mode'):
        require(not env.get('ROLLOUT_POLICY_RNG_SEED'), 'Unified queue assigns policy seeds per episode')
        if panel.get('task_eval_kind'):
            require(int(env.get('ROLLOUT_EVAL_NATIVE_BUDGET', '-1')) == panel['native_step_limit'],
                    'Eval budget setting differs from frozen panel')
        return dict(eval_protocol=VERSION, native_step_limit=panel['native_step_limit'],
                    eval_queue_mode=panel['queue_mode'], policy_rng_seeds=sorted({c['policy_rng_seed'] for c in panel['cases']}),
                    native_scores_visible_to_teacher=False, **prompt_settings)
    require(int(env.get('ROLLOUT_POLICY_RNG_SEED', '-1')) == panel['policy_rng_seed'], 'Policy seed differs from panel')
    return dict(eval_protocol=VERSION, native_step_limit=NATIVE_BUDGET,
                policy_rng_seed=panel['policy_rng_seed'], native_scores_visible_to_teacher=False, **prompt_settings)


# Executed action and planning diagnostics remain available; scoring and scene
# truth stay in the host archives. Filter a COPY at the final model boundary.
HIDDEN_KEYS = frozenset(('native_success', 'success', 'native_score', 'score', 'reward',
    'task_success', 'evaluation_outcome', 'evaluation_case', 'native_conditions',
    'success_within_original_limit', 'success_after_original_limit',
    'partial_score_at_abandon', 'scene_truth', 'object_id_map', 'identity_images',
    'numbered_images', 'simulator_decision_receipt', 'force_hold_events',
    'force_hold_state_transitions', 'actor', 'actor_path'))


def teacher_packet(packet):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in HIDDEN_KEYS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    result = clean(copy.deepcopy(packet))
    # Final result contains evaluator-only reasons/statuses and artifact paths.
    result.pop('result', None)
    return result


def history_spec():
    return dict(type='function', name='robodojo_history',
        description='Read the host-maintained decision history of THIS episode (including after compaction). No filesystem access or cross-episode memory. Optional from_decision filters earlier entries.',
        inputSchema=dict(type='object', properties={'from_decision': {'type': 'integer', 'minimum': 0}},
                         required=[], additionalProperties=False))


def history_packet(history, arguments):
    from .skill_auto_infer.memory import render_history
    from .io import InputError
    start = arguments.get('from_decision', 0)
    if set(arguments) - {'from_decision'} or type(start) is not int or start < 0:
        raise InputError('robodojo_history accepts only nonnegative integer from_decision')
    return {'decision_history': render_history([row for row in history if row['decision'] >= start])}


def report(shared, base, *, unified=False):
    """Experiment-scoped receipts: never combine two checkpoints in a ledger."""
    rows = []
    panel_path = Path(shared)/'eval_panels'/base/'queue.json'
    panel = read_panel(panel_path) if unified and panel_path.exists() else None
    all_cases = panel['cases'] if panel is not None else list(queue_specs())
    from .eval_exclusions import exclusions
    omitted = exclusions(shared, base) if unified else {}
    require(not (set(omitted)-{c['case_id'] for c in all_cases}), 'Excluded case missing from report panel')
    for seed in sorted({c['policy_rng_seed'] for c in all_cases}):
        root = Path(shared)/'results'/(base if unified else f'{base}_pseed{seed}')
        attempts = {}
        for path in root.glob('_replicas/replica_3/attempt_*/batch.json'):
            batch = json.loads(path.read_text())
            for row in batch.get('episodes', []):
                if row.get('state') != 'skipped_completed':
                    attempts.setdefault(row['case_id'], []).append(row)
        cases = [c for c in all_cases if c['policy_rng_seed'] == seed] if unified else specs(seed)
        for case in cases:
            history = attempts.get(case['case_id'], [])
            complete = []
            for row in history:
                outcome, result, reset = (row.get(k) or {} for k in ('evaluation_outcome', 'result', 'reset'))
                if (row.get('artifact_status') == 'verified' and outcome.get('valid_for_success_rate')
                        and outcome.get('complete') and result.get('complete')
                        and reset.get('all_owned_processes_exited') and reset.get('ports_released')
                        and type(outcome.get('native_success')) is bool):
                    complete.append(row)
            item = dict(case_id=case['case_id'], variant=case['variant'], layout_id=case['layout_id'],
                policy_rng_seed=seed, attempts=len(history),
                state='completed' if len(complete) == 1 else 'duplicate_review_required' if complete
                      else 'incomplete' if history else 'not_started',
                attempt_archives=[r['archive'] for r in history])
            if case['case_id'] in omitted:
                item.update(state='excluded', exclusion=omitted[case['case_id']])
                rows.append(item)
                continue
            if len(complete) == 1:
                row = complete[0]
                outcome, result = row['evaluation_outcome'], row['result']
                item.update(archive=row['archive'], native_success=outcome['native_success'],
                    native_score=outcome.get('native_score'), steps=result.get('step_id'),
                    student_steps=result.get('student_steps'),
                    gpt_steps=result.get('edited_steps', 0) + result.get('recovery_steps', 0))
            rows.append(item)
    done = [r for r in rows if r['state'] == 'completed']
    original_planned = len(all_cases) if unified else 15
    planned = original_planned-len(omitted)
    return dict(schema='robodojo.regression_eval_summary.v1', experiment=base, planned=planned,
        original_planned=original_planned, excluded=len(omitted), excluded_cases=list(omitted.values()),
        completed=len(done), successes=sum(r['native_success'] for r in done),
        failures=sum(not r['native_success'] for r in done),
        incomplete_or_unstarted=planned-len(done), episodes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fresh', 'resume'], nargs='?', default='fresh')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1,
                        help='Concurrent episodes per seed round; 2..4 requires an account pool')
    args = parser.parse_args()
    env = dict(os.environ, **REQUIRED_ENV)
    env['ROLLOUT_PARALLEL_WORKERS'] = str(args.workers)
    require(args.workers == 1 or bool(env.get('ROLLOUT_ACCOUNT_POOL_FILE')), 'Set ROLLOUT_ACCOUNT_POOL_FILE for parallel eval')
    base = env.get('ROLLOUT_EXPERIMENT_ID', 'arrange_regression5_base_v1')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', base), 'Invalid experiment ID')
    shared = Path(env['ROLLOUT_SHARED_ROOT'])
    source = env['ROBODOJO_SOURCE']
    for key in ('ROLLOUT_WORKER_CONTROL_FILE', 'ROLLOUT_CASE_IDS', 'ROLLOUT_CASE_ID',
                'ROLLOUT_COLLECTION_REPEAT_ID', 'ROLLOUT_COLLECTION_JOB_INDEX'):
        env.pop(key, None)
    env.update(ROBODOJO_TASK='arrange_largest_number', ROLLOUT_COUNT='5', ROLLOUT_ATTEMPT='auto')
    launcher = Path(__file__).with_name('launch_dual.sh')
    failures = []
    with tempfile.TemporaryDirectory(prefix='robodojo-eval-plan-') as temporary:
        for seed in (0, 1, 2):
            experiment = f'{base}_pseed{seed}'
            panel_path = shared/'eval_panels'/base/f'pseed{seed}.json'
            if panel_path.exists():
                panel = read_panel(panel_path)
                require(panel['schema'] == SCHEMA and panel['policy_rng_seed'] == seed, 'Wrong existing eval panel')
            else:
                panel = freeze(source, seed)
            verify_assets(panel, source, verify_source=False)
            mode = args.mode
            root = shared/'results'/experiment
            prior = list(root.glob('_replicas/replica_3/attempt_*')) + list(root.glob('_account_pool/run_*/pool_run.json'))
            if mode == 'resume' and not prior:
                mode = 'fresh'  # Later seed campaigns may never have started.
            if args.dry_run:
                panel_path = Path(temporary)/f'pseed{seed}.json'
            write_needed = not panel_path.exists()
            if write_needed:
                panel_path.parent.mkdir(parents=True, exist_ok=True)
                with panel_path.open('x') as stream:
                    json.dump(panel, stream, ensure_ascii=False, indent=2)
            child = dict(env, ROLLOUT_EXPERIMENT_ID=experiment, ROLLOUT_RUN_MODE=mode,
                ROLLOUT_POLICY_RNG_SEED=str(seed), ROLLOUT_EVAL_MANIFEST=str(panel_path))
            print(json.dumps(dict(event='eval_campaign', experiment=experiment, policy_rng_seed=seed,
                mode=mode, episodes=5, native_budget=NATIVE_BUDGET, output=str(root))), flush=True)
            completed = subprocess.run(['bash', str(launcher)] + (['--dry-run'] if args.dry_run else []), env=child)
            if completed.returncode:
                failures.append(dict(seed=seed, returncode=completed.returncode))
                break
    if not args.dry_run:
        summary_path = shared/'evaluation'/base/'summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, report(shared, base))
        print(json.dumps(dict(event='eval_summary', path=str(summary_path))), flush=True)
    print(json.dumps(dict(event='eval_finished', failures=failures, planned_episodes=15)), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
