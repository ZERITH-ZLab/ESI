"""Independent CPU contract cases; no native dynamics or grasp-success claim."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import joint_motion as motion, short_motion, gripper_hold, motion_boundary
from .action_edit_kinematics import transform
from .test_joint_motion import FakeSession, point
from .test_short_motion import target
from .session import RoboDojoSession


@pytest.fixture
def actual(tmp_path, monkeypatch):
    session = FakeSession(tmp_path)
    native_step = session.chunk_step
    session.chunk_step = lambda actions, **kw: native_step(np.asarray(actions, np.float32), **kw)
    session.finger_trace = lambda n: np.zeros((2, 2))
    original_update = session.update_obs
    def update():
        original_update()
        session.obs.update(finger_positions_available=True,
                           finger_positions_m=np.asarray(session.finger_trace(session.step_id), float),
                           finger_positions_step_id=session.step_id)
    session.update_obs = update
    session.update_obs()
    monkeypatch.setattr(motion, 'robot_state', lambda kin, arm: (
        session.q[arm].copy(), np.array([[-3., 3.]] * 6), np.ones(6) * 3.14))
    monkeypatch.setattr(motion, 'robot_velocity', lambda kin, arm: np.zeros(6))
    session.gripper_hold_config = gripper_hold.HoldConfig(
        min_steps=3, stable_transitions=2, max_steps=7, delta_m=.00001)
    return session


def do_hold(session, remaining=20):
    action = np.asarray(session.obs['states'], np.float32).copy()
    action[6] = 0.
    goals = {arm: gripper_hold.observed(session, arm) for arm in ('left', 'right')}
    receipt = {}
    saved = []
    rows = gripper_hold.hold(session, action, ['left'], goals,
        session.gripper_hold_config, remaining, motion.DEFAULTS, 'independent',
        receipt, lambda extra=(): saved.append(len(extra)))
    return rows, receipt


def test_success_has_only_post_close_consecutive_samples(actual):
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'fingers_stopped_not_grasp_verified'
    assert len(rows) == 3
    assert [s['stable_transitions']['left'] for s in receipt['samples']] == [0, 1, 2]
    assert [s['step_id'] for s in receipt['samples']] == [1, 2, 3]
    assert all(a[6] == 0 for a in actual.commands)
    assert 'grasp_verified' not in receipt or receipt['grasp_verified'] is False


def test_early_quiet_samples_then_finger_startup_reset_window(actual):
    trace = [0., 0., 0., .003, .003, .003, .003]
    actual.gripper_hold_config = replace(actual.gripper_hold_config, min_steps=5)
    actual.finger_trace = lambda n: np.full((2, 2), trace[min(n, len(trace)-1)])
    actual.update_obs()
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'fingers_stopped_not_grasp_verified'
    assert len(rows) == 5
    assert [s['stable_transitions']['left'] for s in receipt['samples']] == [0, 1, 0, 1, 2]


def test_never_stable_times_out_and_no_overrun(actual):
    actual.finger_trace = lambda n: np.full((2, 2), n * .001)
    actual.update_obs()
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'finger_stability_timeout'
    assert len(rows) == actual.gripper_hold_config.max_steps
    assert actual.step_id == len(rows)


@pytest.mark.parametrize('bad', ['missing', 'stale', 'nonfinite'])
def test_telemetry_failure_refuses_first_close(actual, bad):
    if bad == 'missing':
        actual.obs.pop('finger_positions_m')
    elif bad == 'stale':
        actual.obs['finger_positions_step_id'] = -1
    else:
        actual.obs['finger_positions_m'][0, 0] = np.nan
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'finger_telemetry_unavailable'
    assert rows == [] and actual.commands == [] and actual.step_id == 0


def test_command_echo_does_not_replace_actual_finger_motion(actual):
    actual.finger_trace = lambda n: np.full((2, 2), n * .001)
    actual.update_obs()
    rows, receipt = do_hold(actual)
    assert all(a[6] == 0 for a in actual.commands)  # Setpoint is constant.
    assert receipt['status'] == 'finger_stability_timeout'
    assert len(rows) == 7


def test_hold_pose_drift_aborts_even_if_fingers_stopped(actual, monkeypatch):
    original = actual.chunk_step
    def drifting(actions, **kw):
        result = original(actions, **kw)
        actual.obs['eef_positions'][0, 0] += .006
        return result
    monkeypatch.setattr(actual, 'chunk_step', drifting)
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'hold_pose_error'
    assert len(rows) == 1


def test_hold_needs_tight_pose_before_close(actual):
    actual.obs['eef_positions'][0, 0] = .006
    action = np.asarray(actual.obs['states'], np.float32).copy()
    action[6] = 0
    goals = {a: np.eye(4) for a in ('left', 'right')}
    receipt = {}
    rows = gripper_hold.hold(actual, action, ['left'], goals, actual.gripper_hold_config,
        20, motion.DEFAULTS, 'i', receipt, lambda extra=(): None)
    assert receipt['status'] == 'arrival_not_reached'
    assert not rows and not actual.commands


def test_hold_budget_counts_real_ack_and_never_oversteps(actual):
    actual.metadata['max_episode_steps'] = 2
    rows, receipt = do_hold(actual, remaining=2)
    assert receipt['status'] == 'step_budget'
    assert len(rows) == actual.step_id == 2
    assert actual._motion_action_history[-1]['step_id'] == 2


def test_missing_ack_does_not_become_success_or_history(actual, monkeypatch):
    monkeypatch.setattr(actual, 'chunk_step', lambda actions, **kw: dict(steps=[], step_id=0))
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'ack_missing_or_mismatched'
    assert rows == [] and actual.step_id == 0
    assert not getattr(actual, '_motion_action_history', [])


def test_terminal_ack_prefix_stops_without_future_hold(actual):
    actual.terminal_at = 1
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'native_terminal'
    assert len(rows) == actual.step_id == len(actual.commands) == 1
    assert actual._motion_action_history[-1]['step_id'] == 1


def test_hold_collision_is_hard_failure_before_close(actual):
    def blocked(arm, joints):
        raise motion.SelfCollisionError('independent hold chord collision')
    actual.kinematics.check_self_collision = blocked
    rows, receipt = do_hold(actual)
    assert receipt['status'] == 'self_collision' and not rows
    assert actual.step_id == 0 and actual.commands == []
    assert not getattr(actual, '_motion_action_history', [])


def test_real_pi_chunk_has_identical_actions_and_no_extra_steps(tmp_path):
    class Env:
        def __init__(self):
            self.take_action_cnt, self.end_flag, self.success = [0], [False], [False]
            self.step_lim, self.commands = 100, []
        def take_action(self, command):
            self.commands.append(np.r_[command['left_arm_joint_state'], command['left_ee_joint_state'],
                command['right_arm_joint_state'], command['right_ee_joint_state']].copy())
            self.take_action_cnt[0] += 1
    s = object.__new__(RoboDojoSession)
    s.gripper_contact = s.gripper_contact_error = s.gripper_contact_cache = None
    s.pi_force_hold = None
    s.poisoned = False
    s.gripper_contact_enabled = True
    s.env, s.episode_dir, s.episode_id, s.step_id = Env(), tmp_path, 'pi', 0
    s.finished = s.terminated = s.truncated = s.motion_enabled = s.pi05_collision_shadow = False
    s.source, s.control_epoch, s.metadata = 'student', 0, dict(control_dt=.04, max_episode_steps=100)
    s.gripper_hold_config = gripper_hold.HoldConfig(3, 2, 7, .00001)
    s.obs = dict(states=np.zeros(14, np.float32))
    def observe(record):
        s.obs = dict(states=s.env.commands[-1].copy())
        return s.obs
    s._observe, s.record_native_stages = observe, lambda: None
    actions = np.arange(56, dtype=np.float32).reshape(4, 14) / 100
    result = RoboDojoSession.chunk_step(s, actions)
    np.testing.assert_array_equal(s.env.commands, actions)
    assert len(result['steps']) == s.step_id == s.env.take_action_cnt[0] == 4
    assert len(list(tmp_path.glob('action_*.json'))) == 4
    np.testing.assert_array_equal(s._motion_action_history[-1]['joints'], actions[-1])


def test_long_multivia_close_waits_for_its_own_pose_and_remainder_keeps_issued_zero(actual, monkeypatch):
    original_update = actual.update_obs
    def independent_echo():
        original_update()
        actual.obs['states'][6] = .3  # Incomplete measured/echo closure, not the issued setpoint.
    actual.update_obs = independent_echo
    actual.update_obs()
    actual.obs['states'][6] = 1.
    result = motion.execute(actual, 'left', [point(.01), point(.02, gripper='close'),
                                             point(.03, gripper='keep')], 0, 'independent')
    receipt = result['motion_result']
    assert receipt['status'] == 'waypoints_reached_not_grasp_verified'
    actions = np.asarray(actual.commands)
    closed = np.flatnonzero(actions[:, 6] == 0.)
    assert len(closed) >= 3
    first = int(closed[0])
    assert abs(float(actions[first, 0]) - .02) <= motion.DEFAULTS.arrival_m
    assert np.all(actions[:first, 6] >= .3)
    assert np.all(actions[first:, 6] == 0.), 'keep must preserve issued 0 across segment restart, not echo .3'
    assert len(receipt['events']) == 1
    canonical = actual.episode_dir / 'motion/000.json'
    persisted = json.loads(canonical.read_text())
    assert len(persisted['segments']) == 2 and persisted['executed_steps'] == actual.step_id
    refs = [segment['receipt']['candidate_plan_npz'] for segment in persisted['segments']]
    assert len(set(refs)) == 2 and all(__import__('pathlib').Path(ref).is_file() for ref in refs)
    assert all(str(canonical.parent / '000_segments') in ref for ref in refs)
    assert len(list((canonical.parent / '000_segments').glob('*_plan.npz'))) == 2
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        motion.execute(actual, 'left', [point(.03)], 0, 'independent')
    assert canonical.read_bytes() == before


def test_short_arms_close_at_their_own_target_event(actual):
    targets = [dict(left=target(.01, opening=0.), right=target(.005, opening=1.)),
               dict(left=target(.02, opening=0.), right=target(.02, opening=0.))]
    result = short_motion.execute(actual, targets, 2, 0, 'independent')
    receipt = result['correction_result']
    assert receipt['status'] == 'targets_reached_not_grasp_verified'
    actions = np.asarray(actual.commands)
    first_left = int(np.flatnonzero(actions[:, 6] == 0.)[0])
    first_right = int(np.flatnonzero(actions[:, 13] == 0.)[0])
    assert first_right > first_left
    assert abs(float(actions[first_left, 0]) - .01) <= motion.DEFAULTS.arrival_m
    assert abs(float(actions[first_right, 7]) - .02) <= motion.DEFAULTS.arrival_m
    assert np.all(actions[:first_left, 6] == 1.)
    assert np.all(actions[:first_right, 13] == 1.)
    assert len(receipt['events']) == 2
    assert len(result['steps']) == actual.step_id <= 60


@pytest.mark.parametrize('command, expected', [('keep',1.), ('open',1.), ('close',0.)])
def test_single_stationary_long_command(actual, command, expected):
    result = motion.execute(actual, 'left', [point(0., gripper=command)], 0, 'independent')
    receipt = result['motion_result']
    assert receipt['status'] == 'waypoints_reached_not_grasp_verified'
    if command == 'keep':
        assert not actual.commands and not receipt.get('events', [])
    else:
        assert all(a[6] == expected for a in actual.commands)
    if command == 'close':
        assert len(actual.commands) == 3 and len(receipt['events']) == 1


def test_short_keep_preserves_original_continuous_channel(actual):
    targets = [dict(left=target(opening=.2), right=target(opening=.7)),
               dict(left=target(opening=.4), right=target(opening=.9))]
    result = short_motion.execute(actual, targets, 5, 0, 'independent',
                                  gripper_intent=dict(left='keep', right='keep'))
    assert result['correction_result']['status'] == 'targets_reached_not_grasp_verified'
    actions = np.asarray(actual.commands)
    np.testing.assert_allclose(actions[:, 6], np.linspace(.2, .4, 5), atol=1e-7)
    np.testing.assert_allclose(actions[:, 13], np.linspace(.7, .9, 5), atol=1e-7)
    assert len(actions) == 5


def test_hold_exception_after_native_ack_records_ack_gap(actual, monkeypatch):
    original = actual.chunk_step
    def ack_then_error(actions, **kw):
        original(actions, **kw)
        raise RuntimeError('post ACK observation failure')
    monkeypatch.setattr(actual, 'chunk_step', ack_then_error)
    action = np.asarray(actual.obs['states'], np.float32).copy()
    action[6] = 0.
    receipt = {}
    with pytest.raises(RuntimeError, match='post ACK'):
        gripper_hold.hold(actual, action, ['left'], {a: np.eye(4) for a in ('left','right')},
            actual.gripper_hold_config, 20, motion.DEFAULTS, 'i', receipt, lambda extra=(): None)
    assert actual.step_id == len(actual.commands) == 1
    assert receipt['executed_steps'] == 1 and receipt['returned_ack_rows'] == 0
    assert receipt['ack_rows_missing'] == 1 and receipt['status'] != 'fingers_stopped_not_grasp_verified'
    np.testing.assert_array_equal(actual._motion_action_history[-1]['joints'], actual.commands[-1])


def test_macro_timeout_never_releases_lift(actual):
    actual.finger_trace = lambda n: np.full((2, 2), n * .001)
    actual.update_obs()
    result = motion.execute(actual, 'left', [point(.01, gripper='close'), point(.04)], 0, 'i')
    receipt = result['motion_result']
    assert receipt['status'] == 'finger_stability_timeout'
    assert max(float(a[0]) for a in actual.commands) <= .01 + 1e-6
    assert len(receipt['segments']) == 1
    assert receipt['executed_steps'] == actual.step_id == len(result['steps'])


def test_macro_collision_during_hold_preserves_prefix_and_never_releases_lift(actual):
    check = actual.kinematics.check_self_collision
    def fail_after_first_close(arm, joints):
        if actual.commands and actual.commands[-1][6] == 0.:
            raise motion.SelfCollisionError('late hold collision')
        return check(arm, joints)
    actual.kinematics.check_self_collision = fail_after_first_close
    result = motion.execute(actual, 'left', [point(.01, gripper='close'), point(.04)], 0, 'i')
    receipt = result['motion_result']
    assert receipt['status'] == 'self_collision'
    assert sum(a[6] == 0. for a in actual.commands) == 1
    assert max(float(a[0]) for a in actual.commands) <= .01 + 1e-6
    assert len(result['steps']) == actual.step_id == receipt['executed_steps']
    assert actual._motion_action_history[-1]['step_id'] == actual.step_id


def test_macro_exception_after_hold_ack_archives_actual_ack_gap(actual, monkeypatch):
    original = actual.chunk_step
    def fail_after_close_ack(actions, **kw):
        result = original(actions, **kw)
        if actions[0][6] == 0.:
            raise RuntimeError('independent post close ACK failure')
        return result
    monkeypatch.setattr(actual, 'chunk_step', fail_after_close_ack)
    with pytest.raises(RuntimeError, match='independent post close'):
        motion.execute(actual, 'left', [point(.01, gripper='close'), point(.04)], 0, 'i')
    receipt = json.loads((actual.episode_dir / 'motion/000.json').read_text())
    assert receipt['status'] == 'execution_error_after_ack'
    assert receipt['executed_steps'] == actual.step_id == len(actual.commands)
    assert receipt['ack_rows_missing'] == 1
    assert receipt['returned_ack_rows'] == actual.step_id - 1
    assert sum(a[6] == 0. for a in actual.commands) == 1
    assert max(float(a[0]) for a in actual.commands) <= .01 + 1e-6
    assert actual._motion_action_history[-1]['step_id'] == actual.step_id
    before = len(actual.commands)
    with pytest.raises(ValueError, match='already exists'):
        motion.execute(actual, 'left', [point(.04)], 0, 'i')
    assert len(actual.commands) == before


def test_dual_close_waits_for_slower_fingers(actual):
    actual.finger_trace = lambda n: np.array([[0., 0.], [max(0, 4-n) * .001] * 2])
    actual.update_obs()
    action = np.asarray(actual.obs['states'], np.float32).copy()
    action[[6,13]] = 0.
    receipt = {}
    rows = gripper_hold.hold(actual, action, ['left','right'], {a: np.eye(4) for a in ('left','right')},
        actual.gripper_hold_config, 20, motion.DEFAULTS, 'i', receipt, lambda extra=(): None)
    assert receipt['status'] == 'fingers_stopped_not_grasp_verified'
    assert len(rows) == 6
    assert receipt['samples'][-1]['stable_transitions']['right'] == 2


def test_macro_total_budget_includes_holds_and_remainder(actual):
    actual.metadata['max_episode_steps'] = 5
    result = motion.execute(actual, 'left', [point(.01, gripper='close'), point(.04)], 0, 'i')
    receipt = result['motion_result']
    assert receipt['status'] not in gripper_hold.SUCCESS
    assert len(actual.commands) <= 5 and actual.step_id <= 5
    assert receipt['executed_steps'] == actual.step_id == len(result['steps'])
    assert max([float(a[0]) for a in actual.commands] or [0.]) <= .01 + 1e-6


def test_complete_client_refusal_preserves_no_action_dispatch():
    from .client import RoboDojoTools
    refusal = dict(status='self_collision', error='blocked', executed_steps=0)
    sent = []
    target_value = dict(left=target(), right=target())
    def rpc(op, **payload):
        sent.append((op, payload))
        return dict(correction_result=refusal, step_id=17, steps=[])
    tools = SimpleNamespace(_rpc=rpc, tick=17, history=[], request={'request_id': 'request'})
    rows = RoboDojoTools.execute_correction(tools,
        dict(mode='eef', steps=5, target=target_value))
    assert rows == [] and tools.tick == 17 and tools.correction_result == refusal
    assert sent == [('eef_short_motion', dict(targets=[target_value],
        gripper_intent=dict(left='open', right='open'), minimum_steps=5,
        decision=0, teacher_request_id='request'))]


def test_review_renderer_recognizes_actual_timeout_as_failure(actual):
    from ..tools.render_review_video import failure_details, FAILURE_COLOR, frame_color
    actual.finger_trace = lambda n: np.full((2, 2), n * .001)
    actual.update_obs()
    result = motion.execute(actual, 'left', [point(.01, gripper='close'), point(.04)], 0, 'i')
    row = dict(response=dict(mode='eef', motion=True), motion_result=result['motion_result'])
    failure = failure_details(row)
    assert failure, 'Timed-out hold must be visibly failed in existing renderer'
    assert frame_color(row, failed=bool(failure)) == FAILURE_COLOR


def test_actual_hold_actions_survive_existing_export_and_review_mapping(actual, tmp_path):
    from ..test_motion_sync import archive_fixture
    from ..flywheel.export import export_archive, CAMERAS
    from ..tools.render_review_video import validate_history, failure_details
    from ..io import write_json
    result = motion.execute(actual, 'left', [point(.01, gripper='close'), point(.03)], 0, 'i')
    assert result['motion_result']['status'] == 'waypoints_reached_not_grasp_verified'
    archive = archive_fixture(tmp_path / 'consumer', motion.VERSION)
    raw = archive / 'sim/episode'
    states = [np.r_[np.zeros(6), 1., np.zeros(6), 1.]]
    states.extend(np.asarray(row['obs']['states'], np.float32) for row in result['steps'])
    count = len(actual.commands)
    np.testing.assert_array_equal([r['executed_action'] for r in result['steps']], actual.commands)
    for step, state in enumerate(states):
        np.savez(raw / 'observations' / f'{step:06d}.npz', states=state,
                 **{camera: np.zeros((2, 2, 3), np.uint8) for camera in CAMERAS})
    for step, row in enumerate(result['steps']):
        write_json(raw / f'action_{step:06d}.json', dict(row,
            executed_action=np.asarray(row['executed_action']).tolist(),
            obs=dict(states=np.asarray(row['obs']['states']).tolist())))
    np.savez(archive / 'controller/proposal_000.npz', states=states[0])
    np.savez(archive / 'controller/execution_000.npz', actions=np.asarray(actual.commands))
    record = dict(decision=0, start_tick=0, end_tick=count, executed_steps=count, prediction_id='pi',
        response=dict(mode='eef', motion=True, steps=1, reason='Fixture.', assessment={}),
        motion_result=result['motion_result'])
    write_json(archive / 'controller/history.json', [record])
    write_json(archive / 'sim/summary.json', dict(complete=True, step_id=count,
        terminated=True, truncated=False))
    validate_history([record], count)
    assert failure_details(record) == ''
    output = tmp_path / 'consumer_export'
    exported = export_archive(archive, output, collection_commit='frozen',
                             training_case_id='case', horizon=1, heldout_case_ids=[])
    assert exported['eligible_steps'] == count
    actions = [row['action'] for row in json.loads((output / 'steps.json').read_text())]
    np.testing.assert_array_equal(np.asarray(actions, np.float32), np.asarray(actual.commands))


def test_deployment_snapshot_includes_new_runtime_module_and_its_hash(tmp_path):
    import hashlib
    from pathlib import Path
    from ..deployment import source_files, snapshot
    root = Path(__file__).resolve().parents[3]
    relative = 'hybrid_rollout/robodojo/robodojo_server/gripper_hold.py'
    inventory = {str(rel): path for path, rel in source_files(root)}
    assert relative in inventory
    output = tmp_path / 'source_snapshot'
    snapshot(root, output)
    manifest = json.loads((output / 'source_manifest.json').read_text())
    assert manifest['files'][relative] == hashlib.sha256((output / relative).read_bytes()).hexdigest()
    assert manifest['files'][relative] == hashlib.sha256(inventory[relative].read_bytes()).hexdigest()
    assert not any(__import__('pathlib').Path(name).name.startswith('test_') for name in manifest['files'])


def test_real_session_reset_refuses_missing_finger_feedback_before_policy_ack(tmp_path, monkeypatch):
    from . import session as module
    from ..test_native_evaluation import NativeLikeEnv
    env = NativeLikeEnv()
    native = RoboDojoSession(env, tmp_path, 'build_tower',
        gripper_hold_config=gripper_hold.HoldConfig(3,2,7,.00001))
    monkeypatch.setattr(module, 'DualKinematics', lambda env: SimpleNamespace(
        check=lambda: {'passed': True}, initialize_self_collision=lambda: {'fixture': True}))
    def observe(record=False):
        native.obs = dict(states=np.zeros(14, np.float32), instruction='Build a tower.')
        return native.obs
    monkeypatch.setattr(native, '_observe', observe)
    with pytest.raises(ValueError, match='PhysX measured finger'):
        native.reset(seed=0, source='student', policy_version='test')
    assert env.take_action_cnt == [0] and 'take_action' not in env.events
    assert not getattr(native, '_motion_action_history', [])


def test_short_requested_minimum_includes_closing_hold_ticks(actual):
    result = short_motion.execute(actual,
        [dict(left=target(opening=0.), right=target(opening=1.))], 15, 0, 'i',
        gripper_intent=dict(left='closed', right='keep'))
    receipt = result['correction_result']
    assert receipt['status'] == 'targets_reached_not_grasp_verified'
    assert len(actual.commands) == 15, 'minimum_steps belongs to the entire macro, including the hold'
    assert receipt['events'][0]['executed_steps'] == actual.gripper_hold_config.min_steps
    first = int(np.flatnonzero(np.asarray(actual.commands)[:, 6] == 0.)[0])
    assert first == 15 - actual.gripper_hold_config.min_steps


def test_fast_defaults_stop_at_first_two_post_close_differences(actual):
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    assert actual.gripper_hold_config == gripper_hold.HoldConfig(3, 2, 8, .0001)
    rows, receipt = do_hold(actual)
    assert len(rows) == 3
    assert receipt['status'] == 'fingers_stopped_not_grasp_verified'
    assert [sample['stable_transitions']['left'] for sample in receipt['samples']] == [0, 1, 2]


def test_fast_window_requires_two_consecutive_small_changes_after_reset(actual):
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    trace = [0., 0., .00005, .00025, .00030, .00035]
    actual.finger_trace = lambda n: np.full((2, 2), trace[min(n, len(trace) - 1)])
    actual.update_obs()
    rows, receipt = do_hold(actual)
    assert len(rows) == 5
    assert [sample['stable_transitions']['left'] for sample in receipt['samples']] == [0, 1, 0, 1, 2]


def test_change_equal_to_point_one_mm_does_not_count_as_small(actual):
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    threshold = actual.gripper_hold_config.delta_m
    actual.finger_trace = lambda n: np.full((2, 2), 0. if n < 2 else threshold)
    actual.update_obs()
    rows, receipt = do_hold(actual)
    assert len(rows) == 4
    assert [sample['stable_transitions']['left'] for sample in receipt['samples']] == [0, 0, 1, 2]


def test_fast_default_eight_step_timeout_never_releases_lift(actual):
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    actual.finger_trace = lambda n: np.full((2, 2), n * .001)
    actual.update_obs()
    result = motion.execute(actual, 'left', [point(.01, gripper='close'), point(.03)], 0, 'fast-timeout')
    report = result['motion_result']
    assert report['status'] == 'finger_stability_timeout'
    assert report['events'][0]['executed_steps'] == 8
    assert len([action for action in actual.commands if action[6] == 0.]) == 8
    assert max(action[0] for action in actual.commands) <= .01 + motion.DEFAULTS.ik_position_m
    assert 'delta<0.0001m' in report['error']


def test_fast_budget_reserves_only_three_natural_samples(actual):
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    actual.metadata['max_episode_steps'] = 4
    result = short_motion.execute(actual,
        [dict(left=target(opening=0.), right=target(opening=1.))], 1, 0, 'fast-budget',
        gripper_intent=dict(left='closed', right='keep'))
    receipt = result['correction_result']
    assert receipt['status'] == 'targets_reached_not_grasp_verified'
    assert receipt['segments'][0]['receipt']['executed_steps'] == 1
    assert receipt['events'][0]['executed_steps'] == 3
    assert len(actual.commands) == 4


def test_fast_budget_refuses_when_three_samples_cannot_be_reserved(actual):
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    actual.metadata['max_episode_steps'] = 3
    result = short_motion.execute(actual,
        [dict(left=target(opening=0.), right=target(opening=1.))], 1, 0, 'fast-budget-refusal',
        gripper_intent=dict(left='closed', right='keep'))
    assert result['correction_result']['status'] == 'step_budget'
    assert result['correction_result']['required_hold_minimum_steps'] == 3
    assert not actual.commands


def test_native_recorded_trace_stable_on_eighth_ack_releases_remainder(actual):
    # Frozen prior native session trace; this replays feedback without physics.
    # Source: grasp_hold_20261016/tests/native_180_final/result/native_smoke_result.json
    left_finger_trace = [
        [0.043999698013067245, 0.043999701738357544],
        [0.03508807346224785, 0.035089973360300064],
        [0.02623879909515381, 0.026239633560180664],
        [0.01764174923300743, 0.017642205581068993],
        [0.009507284499704838, 0.009507699869573116],
        [0.0022848027292639017, 0.002285432768985629],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    actual.gripper_hold_config = gripper_hold.DEFAULTS
    actual.finger_trace = lambda n: np.array([
        left_finger_trace[min(n, len(left_finger_trace) - 1)], [.044, .044]])
    actual.update_obs()
    result = motion.execute(actual, 'left', [point(0., gripper='close'), point(.003)], 0, 'native-replay-eight')
    receipt = result['motion_result']
    assert receipt['status'] == 'waypoints_reached_not_grasp_verified'
    assert receipt['events'][0]['status'] == 'fingers_stopped_not_grasp_verified'
    assert receipt['events'][0]['executed_steps'] == 8
    assert receipt['events'][0]['samples'][-1]['stable_transitions']['left'] == 2
    assert len(actual.commands) > 8
    assert all(action[6] == 0. for action in actual.commands)
    assert actual.commands[-1][0] > .0025
