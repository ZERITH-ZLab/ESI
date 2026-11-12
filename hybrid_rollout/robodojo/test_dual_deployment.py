import json
from pathlib import Path
import subprocess
import sys

import pytest

from .batch import completed_archive
from .deployment import snapshot, source_identity
from .dual_campaign import configure, resource_keys
from .evaluation import case_identity
from .io import write_json
from .test_evaluation import fake_panel_file


def test_resource_keys_distinguish_nodes_and_reserve_gpu_and_campaign():
    env = dict(ROLLOUT_SIM_HOST='same-ip', ROLLOUT_SIM_SSH_PORT='4205',
               ROLLOUT_SIM_REMOTE_PORT='19113', SIM_GPU='0',
               ROLLOUT_EXPERIMENT_ID='test', ROLLOUT_REPLICA_ID='8')
    first = resource_keys(env)
    other_node = resource_keys(dict(env, ROLLOUT_SIM_SSH_PORT='5727'))
    assert first[0] != other_node[0] and first[1] != other_node[1]
    assert first[2] == other_node[2]
    other_port = resource_keys(dict(env, ROLLOUT_SIM_REMOTE_PORT='19114'))
    assert first[0] != other_port[0] and first[1] == other_port[1]


def configuration(tmp_path):
    _, panel_path, panel = fake_panel_file(tmp_path)
    env = dict(ROLLOUT_EVAL_MANIFEST=str(panel_path), ROBODOJO_TASK='fold_clothes',
        ROLLOUT_COUNT='5', ROLLOUT_EXPERIMENT_ID='test', ROLLOUT_RUN_MODE='fresh',
        ROLLOUT_ATTEMPT='auto', ROLLOUT_SHARED_ROOT=str(tmp_path))
    return env, panel


def test_task_and_cases_resolve_without_replica_input(tmp_path):
    env, panel = configuration(tmp_path)
    actual, cases = configure(env)
    assert actual['ROLLOUT_REPLICA_ID'] == '8'
    assert len(cases) == 5 and cases[-1]['variant'] == 'random'
    env['ROLLOUT_CASE_IDS'] = cases[4]['case_id']+','+cases[1]['case_id']
    actual, selected = configure(env)
    assert actual['ROLLOUT_COUNT'] == '2'
    assert [c['case_id'] for c in selected] == env['ROLLOUT_CASE_IDS'].split(',')


def test_modes_never_overwrite(tmp_path):
    env, _ = configuration(tmp_path)
    base = tmp_path/'results/test/_replicas/replica_8'
    (base/'attempt_0').mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'): configure(env)
    env['ROLLOUT_RUN_MODE'] = 'resume'
    actual, _ = configure(env)
    assert actual['ROLLOUT_ATTEMPT'] == '1'
    assert actual['ROLLOUT_SKIP_COMPLETED_CASES'] == '1'
    env['ROLLOUT_RUN_MODE'] = 'rerun'
    assert configure(env)[0]['ROLLOUT_SKIP_COMPLETED_CASES'] == '0'
    env['ROLLOUT_ATTEMPT'] = '0'
    with pytest.raises(ValueError, match='already exists'): configure(env)


def test_snapshot_excludes_private_files_and_is_reproducible(tmp_path):
    root = tmp_path/'repo'
    paths = ['hybrid_rollout/__init__.py', 'hybrid_rollout/robodojo/skill/run.py',
             'hybrid_rollout/robodojo/skill/SKILL.md', 'hybrid_rollout/robodojo/.env.server',
             'hybrid_rollout/robodojo/__pycache__/a.pyc', 'hybrid_rollout/robodojo/test_a.py',
             'hybrid_rollout/robodojo/tools/debug/preview.py']
    for name in paths:
        p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('sample')
    (root/'hybrid_rollout/robodojo/requirements-motion.txt').write_text('# Existing NumPy/SciPy only.\n')
    dest = tmp_path/'snapshot'
    assert snapshot(root, dest) == source_identity(root) == source_identity(dest)
    assert (dest/'hybrid_rollout/robodojo/skill/SKILL.md').exists()
    assert (dest/'hybrid_rollout/robodojo/requirements-motion.txt').exists()
    for name in paths[3:]: assert not (dest/name).exists()
    # Editing the launcher's common knobs must not invalidate otherwise identical resume.
    (root/'hybrid_rollout/robodojo/launch_dual.sh').write_text('RUN_MODE=resume')
    assert source_identity(root) == source_identity(dest)
    with pytest.raises(FileExistsError): snapshot(root, dest)


@pytest.mark.parametrize('mismatch', [None, 'panel', 'method', 'checkpoint', 'source', 'cleanup', 'incomplete'])
def test_resume_requires_full_identity_and_complete_evidence(tmp_path, mismatch):
    env, panel = configuration(tmp_path)
    _, cases = configure(env); case = cases[0]; identity = case_identity(panel, case)
    runtime = dict(evaluation_method='pi05_plus_gpt', checkpoint='one', source='abc')
    root = tmp_path/'results/test/fold_clothes/standard/eval_seed_0/layout_0/replica_8/attempt_0'
    (root/'controller').mkdir(parents=True); (root/'sim').mkdir()
    saved = dict(runtime)
    if mismatch in ('method', 'checkpoint', 'source'):
        saved['evaluation_method' if mismatch == 'method' else mismatch] = 'different'
    result_identity = dict(identity)
    if mismatch == 'panel': result_identity['panel_sha256'] = 'wrong'
    write_json(root/'resume_identity.json', saved)
    write_json(root/'component_reset.json', dict(all_owned_processes_exited=mismatch != 'cleanup', ports_released=True))
    write_json(root/'artifact_manifest.json', dict(status='verified', evaluation_method='pi05_plus_gpt'))
    write_json(root/'controller/result.json', dict(complete=mismatch != 'incomplete', evaluation_case=result_identity))
    # An unsuccessful native episode must still be skipped, not cherry-picked.
    write_json(root/'sim/evaluation_outcome.json', dict(complete=True, valid_for_success_rate=True,
        native_success=False, evaluation_case=identity))
    found = completed_archive(tmp_path, 'test', case, '8', identity, runtime)
    assert found == (root if mismatch is None else None)


def test_remote_status_rejects_stale_pid_without_killing(tmp_path):
    import os
    p = tmp_path/'sim.pid'; p.write_text(str(os.getpid()))
    p.with_suffix('.pid.start').write_text('wrong-birth-time')
    script = Path(__file__).with_name('remote_simulator.sh')
    result = subprocess.run(['bash', str(script), 'status', str(p)])
    assert result.returncode != 0


def test_launch_dry_run_has_no_model_or_output_writes(tmp_path):
    import os
    script = Path(__file__).with_name('launch_dual.sh')
    env = dict(os.environ, ROLLOUT_SHARED_ROOT=str(tmp_path), ROBODOJO_PYTHON=sys.executable,
        ROBODOJO_TASK='arrange_largest_number', ROLLOUT_RUN_MODE='fresh',
        ROLLOUT_CASE_IDS='', ROLLOUT_EXPERIMENT_ID='dry', ROLLOUT_ATTEMPT='auto')
    result = subprocess.run(['bash', str(script), '--dry-run'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['task'] == 'arrange_largest_number'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('controller_exit', [0, 7])
@pytest.mark.parametrize('scene_query', ['0', '1'])
@pytest.mark.parametrize('motion_tools', ['0', '1'])
def test_shell_launcher_lifecycle_with_fake_services(tmp_path, controller_exit, scene_query, motion_tools, runtime_snapshot=False, ssh_user="root"):
    """Real shell/process ownership, fake SSH transport and no GPU/model calls."""
    import os
    import socket
    def free_port():
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); return s.getsockname()[1]
    bindir = tmp_path/'bin'; bindir.mkdir()
    ssh = bindir/'ssh'
    ssh.write_text('#!'+sys.executable+'\n'+'''import os,sys,time
args=sys.argv[1:]
assert os.environ.get('ROLLOUT_SIM_SSH_USER', 'root')+'@fake' in args, args
if '-N' in args:
    while True: time.sleep(1)
index=next(i for i,x in enumerate(args) if x==os.environ.get('ROLLOUT_SIM_SSH_USER', 'root')+'@fake')
# Real SSH does not inherit these local feature flags.
for key in ('ROBODOJO_GRIPPER_CONTACT_ENABLED', 'ROBODOJO_PI_FORCE_HOLD_ENABLED',
            'ROBODOJO_PI_FORCE_HOLD_CONFIG', 'ROBODOJO_GRIPPER_HOLD_CONFIG'):
    os.environ.pop(key, None)
os.execvp('bash',['bash','-c',' '.join(args[index+1:])])
''')
    ssh.chmod(0o755)
    wrapper = bindir/'python-service'
    wrapper.write_text('#!'+sys.executable+'\n'+'''import json,os,socket,sys,time
from pathlib import Path
a=sys.argv[1:]
if '-c' in a and 'joint_motion import VERSION, Limits' in a[a.index('-c')+1]:
    raise SystemExit(0)  # Numerical planner dependencies are exercised in planner tests.
module=a[a.index('-m')+1] if '-m' in a else ''
def arg(name): return a[a.index(name)+1]
if module.endswith('pi05_server.server'):
    assert arg('--policy-rng-seed') == os.environ.get('ROLLOUT_POLICY_RNG_SEED', '0')
    Path(arg('--identity-output')).write_text('{}')
elif module.endswith('robodojo_server.cooked_collision'):
    raise SystemExit(0)  # Geometry is tested separately; this fixture has no robot assets.
elif module.endswith('robodojo_server.server'):
    assert os.environ['ROBODOJO_GRIPPER_CONTACT_ENABLED'] == '1'
    assert os.environ['ROBODOJO_PI_FORCE_HOLD_ENABLED'] == os.environ['MOCK_MOTION_TOOLS']
    assert json.loads(os.environ['ROBODOJO_PI_FORCE_HOLD_CONFIG']) == {'stable_delta_m': 0.0002}
    assert json.loads(os.environ['ROBODOJO_GRIPPER_HOLD_CONFIG']) == dict(min_steps=3, stable_transitions=2, max_steps=7, delta_m=.0002)
    assert ('--enable-scene-query' in a) == (os.environ['MOCK_SCENE_QUERY'] == '1')
    assert ('--enable-motion-tools' in a) == (os.environ['MOCK_MOTION_TOOLS'] == '1')
    if os.environ.get('ROLLOUT_COLLECTION_STEP_LIMIT') == '1500':
        assert arg('--collection-step-limit') == '1500'
    else:
        assert '--collection-step-limit' not in a
    s=socket.socket(); s.bind(('127.0.0.1',int(arg('--port')))); s.listen()
    print(json.dumps({'event':'ready'}),flush=True)
elif module.endswith(('skill.run', 'skill_auto_infer.run')):
    assert ('--enable-scene-query' in a) == (os.environ['MOCK_SCENE_QUERY'] == '1')
    Path(arg('--output')).mkdir()
    raise SystemExit(int(os.environ['MOCK_CONTROLLER_EXIT']))
else:
    os.execv(sys.executable,[sys.executable,*a])
while True: time.sleep(1)
''')
    wrapper.chmod(0o755)
    panel_dir = tmp_path/'batch'; panel_dir.mkdir()
    for name in ('evaluation_manifest.json', 'case_000.json'):
        (panel_dir/name).write_text('{}')
    repo = Path(__file__).parents[2]
    if runtime_snapshot:
        from .deployment import RUNTIME_SOURCE_SCOPE
        frozen = tmp_path/'release-runtime'
        snapshot(repo, frozen, scope=RUNTIME_SOURCE_SCOPE)
        repo = frozen
    archive = tmp_path/'episode'
    env = dict(os.environ, PATH=str(bindir)+':'+os.environ['PATH'], CODE_ROOT=str(repo),
        ROBODOJO_PYTHON=str(wrapper), OPENPI_PYTHON=str(wrapper), OPENPI_SOURCE=str(tmp_path),
        CHECKPOINT='unused', CODEX_BIN='unused', ROLLOUT_ARCHIVE=str(archive),
        ROLLOUT_SHARED_ROOT=str(tmp_path), ROLLOUT_CASE_FILE=str(panel_dir/'case_000.json'),
        ROLLOUT_EVAL_MANIFEST=str(panel_dir/'evaluation_manifest.json'),
        ROBODOJO_TASK='fold_clothes', ROLLOUT_EVAL_SEED='0', ROLLOUT_LAYOUT_ID='0',
        ROLLOUT_SIM_HOST='fake', ROLLOUT_SIM_SSH_USER=ssh_user, ROLLOUT_SIM_SSH_KEY='unused', ROLLOUT_SIM_PYTHON=str(wrapper),
        ROLLOUT_SIM_ROBODOJO_SOURCE=str(tmp_path), ROLLOUT_SIM_REMOTE_PORT=str(free_port()),
        SIM_PORT=str(free_port()), POLICY_PORT=str(free_port()), ROLLOUT_EXPERIMENT_ID='fake',
        ROLLOUT_REPLICA_ID='8', ROLLOUT_ATTEMPT='0', ROLLOUT_INDEX='0',
        ROLLOUT_STARTUP_TIMEOUT='15', MOCK_CONTROLLER_EXIT=str(controller_exit),
        ROLLOUT_PI05_LOAD_LOCK=str(tmp_path/'load.lock'),
        MOCK_SCENE_QUERY=scene_query, ROLLOUT_ENABLE_SCENE_QUERY=scene_query,
        MOCK_MOTION_TOOLS=motion_tools, ROLLOUT_ENABLE_MOTION_TOOLS=motion_tools,
        ROBODOJO_GRIPPER_CONTACT_ENABLED='1', ROBODOJO_PI_FORCE_HOLD_ENABLED=motion_tools,
        ROBODOJO_PI_FORCE_HOLD_CONFIG='{"stable_delta_m": 0.0002}',
        ROBODOJO_GRIPPER_HOLD_CONFIG='{"min_steps":3,"stable_transitions":2,"max_steps":7,"delta_m":0.0002}',
        ROLLOUT_COLLECTION_STEP_LIMIT='1500' if motion_tools == '1' else '0',
        ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1' if scene_query == '1' or motion_tools == '1' else 'baseline',
        ROLLOUT_CODEX_HOME_DIR=str(tmp_path/'private-login'))
    env.pop('ROLLOUT_RESUME_IDENTITY', None)
    remote_env = tmp_path/'remote-env.sh'
    remote_env.write_text('export ROBODOJO_PI_FORCE_HOLD_ENABLED=0\n')
    env['ROLLOUT_SIM_ENV_SCRIPT'] = str(remote_env)
    script = Path(__file__).with_name('dual_machine_episode_entrypoint.sh')
    process = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == controller_exit, process.stdout + process.stderr
    assert process.stdout.index('pi05_load_acquired') < process.stdout.index('Starting pi05:')
    assert process.stdout.index('pi05_load_released') < process.stdout.index('Starting controller:')
    assert (archive/'dual_machine/remote_cleanup_status.txt').read_text().strip() == '0'
    assert (archive/'dual_machine/episode_exit_status.txt').read_text().strip() == str(controller_exit)
    assert (archive/'source_snapshot/hybrid_rollout/robodojo/remote_simulator.sh').is_file()
    assert not (archive/'dual_machine/codex_state').exists()
    assert (tmp_path/'private_runtime/fake/replica_8/attempt_0/case_0/codex_state').is_dir()


def test_nonroot_shell_lifecycle(tmp_path):
    test_shell_launcher_lifecycle_with_fake_services(
        tmp_path, 0, '0', '1', ssh_user='robot')
