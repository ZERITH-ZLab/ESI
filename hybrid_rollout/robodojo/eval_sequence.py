"""Run checkpoint campaigns in order, each with one shared 15-episode queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from . import regression_eval as evaluation
from .evaluation import read_panel, verify_assets
from .io import require, write_json


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    require(cfg.get('schema') == 'robodojo.eval_sequence.v1', 'Invalid eval sequence config')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', cfg['name']), 'Invalid sequence name')
    require(type(cfg['workers']) is int and cfg['workers'] >= 1, 'Workers must be positive')
    if cfg.get('scheduling') != 'cross_task_queue_v1':
        require(cfg['workers'] <= 4, 'Legacy sequential eval supports at most 4 workers; use the unified queue for configurable slots')
    models = cfg['models']
    require(bool(models), 'No models configured')
    require(len({m['experiment'] for m in models}) == len(models), 'Duplicate experiment names')
    for model in models:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', model['experiment']), 'Invalid experiment name')
        require(model['policy_input'] in ('task_only', 'task_subtask'), 'Unknown policy_input')
        from .prompt_profile import profile, BASELINE, NO_GUIDANCE, FREEFORM, PRETRAINED, TRANSFER_BASELINE, ENV_KEY
        selected = profile({ENV_KEY:model.get('prompt_profile','guided_v1')})
        if selected in (NO_GUIDANCE, FREEFORM, PRETRAINED, TRANSFER_BASELINE):
            require(model['policy_input']=='task_subtask', 'Ablation requires feedforward subtask conditioning')
        if selected == BASELINE:
            require(model['policy_input']=='task_only', 'Baseline cannot use subtask conditioning')
            require(model.get('downsampling',{}).get('mode','off')=='off', 'Baseline cannot downsample')
        if 'task' in model:
            from .task_eval import selected_specs
            selected_specs(model['task'], model['cases'])
        if 'placement_order' in model:
            from .placement_order import validate, VERSION
            validate(model['placement_order'])
            require(cfg.get('experiment_kind') in (VERSION, 'paired_order_queue_v1') and
                    model.get('task') == 'arrange_largest_number' and
                    model.get('cases') == [['standard', 0]] and model['policy_input'] == 'task_subtask',
                    'Placement order requires its explicit paired eval configuration')
        if 'downsampling' in model:
            from .downsampling import configuration
            require(model['downsampling']['mode'] in ('off', 'error'), 'Invalid downsampling mode')
            configuration(dict(ROLLOUT_ACTION_DOWNSAMPLING=model['downsampling']['mode'],
                ROLLOUT_DOWNSAMPLING_CONFIG=json.dumps(model['downsampling'].get('config', {}))))
        ckpt = Path(model['checkpoint'])
        require(ckpt.is_absolute(), 'Checkpoint must be an absolute path')
        require((ckpt/'params').is_dir(), f'Missing inference params: {ckpt}')
        require((ckpt/'assets/arx_x5_sim/norm_stats.json').is_file(), f'Missing normalization: {ckpt}')
    return cfg


def campaign_env(env, cfg, model, panel_path):
    route = 'subtask_feedforward_v1' if model['policy_input'] == 'task_subtask' else 'auto_infer_v1'
    child = dict(env, **evaluation.required_env(route))
    from .placement_order import ENV_KEY as ORDER_ENV
    child.pop(ORDER_ENV, None)  # Never leak a previous order into an unrelated campaign.
    if 'placement_order' in model:
        child[ORDER_ENV] = model['placement_order']
    from .prompt_profile import ENV_KEY, BASELINE, GUIDED, NO_GUIDANCE, FREEFORM, PRETRAINED, TRANSFER_BASELINE
    child[ENV_KEY] = model.get('prompt_profile', GUIDED)
    if child[ENV_KEY] in (BASELINE, NO_GUIDANCE, TRANSFER_BASELINE):
        child['ROLLOUT_TASK_REFERENCE_IMAGES'] = '0'
    elif child[ENV_KEY] in (FREEFORM, PRETRAINED):
        child['ROLLOUT_TASK_REFERENCE_IMAGES'] = '1'
    for key in ('ROLLOUT_WORKER_CONTROL_FILE', 'ROLLOUT_CASE_IDS', 'ROLLOUT_CASE_ID',
                'ROLLOUT_COLLECTION_REPEAT_ID', 'ROLLOUT_COLLECTION_JOB_INDEX',
                'ROLLOUT_POLICY_RNG_SEED'):
        child.pop(key, None)
    root = Path(child['ROLLOUT_SHARED_ROOT'])/'results'/model['experiment']
    prior = (list(root.glob('_replicas/replica_3/attempt_*')) +
             list(root.glob('_account_pool/run_*/pool_run.json')))
    child.update(ROBODOJO_TASK=model.get('task', 'arrange_largest_number'),
        ROLLOUT_COUNT=str(len(model['cases']) if 'task' in model else 15),
        ROLLOUT_ATTEMPT='auto', ROLLOUT_PARALLEL_WORKERS=str(cfg['workers']),
        ROLLOUT_ACCOUNT_POOL_FILE=cfg['account_pool_file'],
        ROLLOUT_EXPERIMENT_ID=model['experiment'], CHECKPOINT=model['checkpoint'],
        ROLLOUT_RUN_MODE='resume' if prior else 'fresh', ROLLOUT_EVAL_MANIFEST=str(panel_path))
    if 'task' in model:
        panel = read_panel(panel_path)
        child['ROLLOUT_EVAL_NATIVE_BUDGET'] = str(panel['native_step_limit'])
    child['ROLLOUT_ACTION_DOWNSAMPLING'] = 'off'
    if 'task' in model or 'downsampling' in model or child[ENV_KEY] in (BASELINE, NO_GUIDANCE, FREEFORM, PRETRAINED, TRANSFER_BASELINE):
        # Pin new eval settings; legacy configs retain their gripper behavior.
        ds = model.get('downsampling', {'mode': 'off'})
        child.update(ROLLOUT_ACTION_DOWNSAMPLING=ds['mode'],
            ROLLOUT_DOWNSAMPLING_CONFIG=json.dumps(ds.get('config', {})),
            ROBODOJO_PI_FORCE_HOLD_ENABLED='0', ROBODOJO_PI_FORCE_HOLD_CONFIG='{}')
    return child


def save_report(shared, model):
    summary = evaluation.report(shared, model['experiment'], unified=True)
    summary.update(checkpoint=model['checkpoint'], policy_input=model['policy_input'])
    path = Path(shared)/'evaluation'/model['experiment']/'summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, summary)
    print(json.dumps(dict(event='eval_summary', path=str(path), completed=summary['completed'],
        successes=summary['successes'], failures=summary['failures'],
        incomplete=summary['incomplete_or_unstarted'])), flush=True)
    return summary


def run_sequence(cfg, env, *, dry_run=False):
    require(cfg.get('scheduling') != 'cross_task_queue_v1',
            'Cross-task queue requires launch_dual.sh --eval-queue --config <config>')
    shared = Path(env['ROLLOUT_SHARED_ROOT'])
    sequence_root = shared/'evaluation'/cfg['name']
    print(json.dumps(dict(event='eval_sequence_plan', output=str(sequence_root),
        workers=cfg['workers'], models=cfg['models'],
        planned_episodes=sum(len(m['cases']) if 'task' in m else 15 for m in cfg['models']),
        wait_for_resources=True)), flush=True)
    lock = None
    if not dry_run:
        sequence_root.mkdir(parents=True, exist_ok=True)
        lock = (sequence_root/'sequence.lock').open('a')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise ValueError('This evaluation sequence is already running')
    try:
        if not dry_run:
            # A repeated command resumes this exact ordered experiment, never overwrites it.
            frozen = dict(config=cfg, teacher={k: env.get(k) for k in (
                'ROLLOUT_MODEL', 'ROLLOUT_REASONING_EFFORT', 'ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT')})
            config_path = sequence_root/'sequence.json'
            if config_path.exists():
                require(json.loads(config_path.read_text()) == frozen,
                        'Sequence config changed; use a new sequence and experiment names')
            else:
                write_json(config_path, frozen)
        with tempfile.TemporaryDirectory(prefix='robodojo-eval-sequence-') as temporary:
            for model in cfg['models']:
                name = model['experiment']
                panel_path = shared/'eval_panels'/name/'queue.json'
                if 'task' in model:
                    from .task_eval import freeze
                    expected_panel = freeze(env['ROBODOJO_SOURCE'], model['task'], model['cases'])
                    panel = read_panel(panel_path) if panel_path.exists() else expected_panel
                    require(panel == expected_panel, 'Task panel changed; use a new experiment')
                else:
                    panel = read_panel(panel_path) if panel_path.exists() else evaluation.freeze_queue(env['ROBODOJO_SOURCE'])
                planned = len(panel['cases'])
                require(panel.get('queue_mode') == 'all_policy_seeds_v1', 'Expected unified eval panel')
                verify_assets(panel, env['ROBODOJO_SOURCE'], verify_source=False)
                if dry_run:
                    panel_path = Path(temporary)/f'{name}.json'
                if not panel_path.exists():
                    panel_path.parent.mkdir(parents=True, exist_ok=True)
                    with panel_path.open('x') as stream:
                        json.dump(panel, stream, ensure_ascii=False, indent=2)
                if not dry_run and evaluation.report(shared, name, unified=True)['completed'] == planned:
                    save_report(shared, model)
                    print(json.dumps(dict(event='eval_model_skipped_completed', experiment=name)), flush=True)
                    continue
                child = campaign_env(env, cfg, model, panel_path)
                print(json.dumps(dict(event='eval_model_start', experiment=name,
                    checkpoint=model['checkpoint'], policy_input=model['policy_input'],
                    mode=child['ROLLOUT_RUN_MODE'], episodes=planned,
                    output=str(shared/'results'/name))), flush=True)
                command = ['bash', str(Path(__file__).with_name('launch_dual.sh')),
                           '--wait-for-resources'] + (['--dry-run'] if dry_run else [])
                try:
                    result = subprocess.run(command, env=child)
                finally:
                    if not dry_run:
                        summary = save_report(shared, model)
                if result.returncode or (not dry_run and summary['completed'] != planned):
                    print(json.dumps(dict(event='eval_sequence_stopped', experiment=name,
                        returncode=result.returncode, message='Next model not started; rerun the same command to resume incomplete episodes')), flush=True)
                    return 1
        print(json.dumps(dict(event='eval_sequence_finished', planned_episodes=sum(
            len(m['cases']) if 'task' in m else 15 for m in cfg['models']),
                              dry_run=dry_run)), flush=True)
        return 0
    finally:
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('eval_sequence.trained.json')))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run_sequence(read_config(args.config), dict(os.environ), dry_run=args.dry_run))


if __name__ == '__main__':
    main()
