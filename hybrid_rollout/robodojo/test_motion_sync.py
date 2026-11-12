"""CPU integration checks for current executor identity and offline export."""

import json
from pathlib import Path

import numpy as np
import pytest

from .robodojo_server.joint_motion import VERSION
from .robodojo_server.session import RoboDojoSession
from .test_native_evaluation import NativeLikeEnv
from .flywheel.export import CAMERAS, export_archive


def test_new_snapshot_inventory_includes_active_short_executor():
    from .deployment import source_files
    root = Path(__file__).resolve().parents[2]
    inventory = {str(relative): path for path, relative in source_files(root)}
    prefix = 'hybrid_rollout/robodojo/robodojo_server/'
    for name in ('short_motion.py', 'joint_motion.py', 'motion_boundary.py',
                 'client.py', 'session.py'):
        assert inventory[prefix + name] == root / prefix / name
    assert not any(Path(relative).name.startswith('test_') for relative in inventory)


def test_session_and_resume_identity_use_current_executor(tmp_path, monkeypatch):
    from . import dual_campaign, collection
    from .pi05_server import checkpoint

    session = RoboDojoSession(NativeLikeEnv(), tmp_path / 'sim', 'build_tower',
                             motion_enabled=True)
    assert session.metadata['motion_executor'] == VERSION
    monkeypatch.setattr(dual_campaign, 'model_selection', lambda env: ('model', 'high'))
    monkeypatch.setattr(dual_campaign, 'controller_route_identity', lambda env: {})
    monkeypatch.setattr(dual_campaign, 'source_identity', lambda path: 'a' * 64)
    monkeypatch.setattr(dual_campaign, 'native_identity', lambda path: 'b' * 64)
    monkeypatch.setattr(checkpoint, 'checkpoint_identity',
                        lambda path: {'checkpoint_sha256': 'c' * 64})
    monkeypatch.setattr(dual_campaign, 'read_panel', lambda path: {})
    monkeypatch.setattr(collection, 'runtime_settings', lambda env, panel: {})
    monkeypatch.setattr(dual_campaign.subprocess, 'check_output',
                        lambda args, **kwargs: 'd' * 64 if args[0] == 'ssh' else 'fake-cli')
    env = dict(ROLLOUT_SIM_PYTHON='python', ROLLOUT_SIM_ROBODOJO_SOURCE='native',
               ROLLOUT_SIM_SSH_KEY='unused', ROLLOUT_SIM_SSH_PORT='22',
               ROLLOUT_SIM_HOST='unused', ROLLOUT_EVAL_MANIFEST_SHA256='panel',
               ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt', CHECKPOINT='checkpoint',
               CODE_ROOT='source', ROBODOJO_SOURCE='native', OPENPI_SOURCE='openpi',
               CODEX_BIN='codex', MAX_DECISIONS='0', ROLLOUT_MAX_SECONDS='60',
               ROLLOUT_EVAL_MANIFEST='panel.json', ROLLOUT_ENABLE_MOTION_TOOLS='1')
    assert dual_campaign.runtime_identity(env)['motion_executor'] == VERSION


def archive_fixture(tmp_path, version):
    """One synthetic executed EEF tick; no simulator, model or teacher call."""
    archive = tmp_path / 'results' / 'sync' / '_replicas' / 'replica_0' / 'attempt_0'
    raw = archive / 'sim' / 'episode'
    observations = raw / 'observations'
    observations.mkdir(parents=True)
    workspace = archive / 'controller' / 'codex_workspace'
    workspace.mkdir(parents=True)

    def write(relative, value):
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    case = dict(case_id='case', task='build_tower', runtime_task='build_tower',
                variant='standard', eval_seed=0, layout_id=0, reset_seed=0)
    write('sim/reset.json', dict(episode_id='episode', instruction='Build a tower.',
                                metadata=dict(evaluation_case=case, motion_executor=version)))
    write('sim/summary.json', dict(complete=True, step_id=1, terminated=True, truncated=False))
    write('sim/evaluation_outcome.json', dict(complete=True, valid_for_success_rate=True,
                                            native_success=True))
    write('artifact_manifest.json', dict(status='verified'))
    write('controller/run.json', dict(control_dt=.04, action_dim=14, teacher_model='fixture',
                                     teacher_reasoning_effort='high', teacher_prompt_sha256='p',
                                     student_identity_sha256='s'))
    record = dict(decision=0, start_tick=0, end_tick=1, executed_steps=1, prediction_id='pi',
                  response=dict(mode='eef', steps=1, reason='Fixture.', assessment={}),
                  correction_result=dict(status='targets_reached_not_grasp_verified'))
    write('controller/history.json', [record])
    write('controller/request_000.json', dict(request_id='request', step_id=0))
    write('resume_identity.json', dict(max_seconds=60))
    write('source_snapshot/source_manifest.json', dict(sha256='snapshot'))
    batch = archive.parent / 'batch.json'
    batch.write_text(json.dumps(dict(episodes=[dict(archive=str(archive), started_utc='start',
                                                  finished_utc='end')])))
    write('archive_location.json', dict(batch_path=str(batch), repeat_id=0))
    states = np.zeros(14, np.float32)
    for step in range(2):
        np.savez(observations / f'{step:06d}.npz', states=states,
                 **{camera: np.zeros((2, 2, 3), np.uint8) for camera in CAMERAS})
    np.savez(archive / 'controller/proposal_000.npz', states=states)
    np.savez(archive / 'controller/execution_000.npz', actions=states[None])
    write('sim/episode/action_000000.json', dict(step_id=1, valid=True,
          executed_action=states.tolist(), obs=dict(states=states.tolist()),
          terminated=True, truncated=False))
    (workspace / 'rpc_out.jsonl').write_text('')
    (workspace / 'rpc_in.jsonl').write_text('')
    spans = []
    for identity, stage, start, end in [('reset', 'rpc.reset', 1., 2.),
                                       ('short', 'rpc.eef_short_motion', 3., 7.)]:
        spans.extend([dict(event='start', writer='one', span_id=identity, stage=stage, mono=start),
                      dict(event='end', writer='one', span_id=identity, stage=stage, mono=end,
                           duration_s=end-start, status='ok')])
    timing = archive / 'timing/controller.jsonl'
    timing.parent.mkdir()
    timing.write_text('\n'.join(json.dumps(row) for row in spans))
    return archive


def test_export_accepts_current_identity_and_counts_short_rpc_time(tmp_path):
    archive = archive_fixture(tmp_path, VERSION)
    output = tmp_path / 'export'
    result = export_archive(archive, output, collection_commit='frozen',
                            training_case_id='case', horizon=1, heldout_case_ids=[])
    assert result['eligible_steps'] == 1 and result['source_counts'] == {'gpt_eef': 1}
    episode = json.loads((output / 'episode.json').read_text())
    assert episode['cost']['control_loop_wall_s'] == 5.
    assert episode['gpt_eef_steps'] == 1
    assert json.loads((output / 'steps.json').read_text())[0]['action'] == [0.] * 14


def test_export_refuses_old_executor_before_creating_output(tmp_path):
    archive = archive_fixture(tmp_path, 'link6_unified_reference_v4')
    output = tmp_path / 'export'
    with pytest.raises(ValueError, match='unexpected IK executor'):
        export_archive(archive, output, collection_commit='frozen', training_case_id='case',
                       horizon=1, heldout_case_ids=[])
    assert not output.exists()
