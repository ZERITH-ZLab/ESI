"""Deterministic bootstrap margin, overlapping stability and model hold tests."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from .pi_force_hold import ForceHoldConfig, PiForceHold, SINGLE_FINGER_SCALE_M, configured, margin_plan, BOOTSTRAP_CHECKPOINT_SHA256, validate_tightening_context

ACTOR = '/World/envs/env_0/rigid/block'


def bootstrap_config(**kwargs):
    return ForceHoldConfig(tighten_enabled=True, flywheel_round=1,
        controller_route='auto_infer_v1', checkpoint_sha256=BOOTSTRAP_CHECKPOINT_SHA256, **kwargs)


def action(value=.7, right=.8):
    a = np.zeros(14, np.float32)
    a[[6, 13]] = [value, right]
    return a


def receipt(step, *, actor=ACTOR, force=1., same_direction=False, episode='ep'):
    fingers = {}
    for index, name in enumerate(('link7', 'link8')):
        sign = 1 if index == 0 or same_direction else -1
        fingers[name] = dict(valid=True, objects={actor: dict(
            force_magnitude_mean_N=force, force_world_mean_N=[sign * force, 0., 0.])})
    return dict(episode_id=episode, step_id=step, reports_valid=True,
                arms=dict(left=fingers, right=deepcopy(fingers)))


class Session:
    def __init__(self, *, force=1., moving=False, base=.7, right=.8):
        self.episode_id, self.step_id = 'ep', 0
        self.terminated = self.truncated = False
        self.metadata = dict(max_episode_steps=100, pi_force_hold={})
        self.obs = dict(states=action(base, right), finger_positions_available=True,
            finger_positions_step_id=0, finger_positions_m=np.ones((2, 2)) * .04)
        self.gripper_contact_cache = receipt(0, force=force)
        self.kinematics = SimpleNamespace(manager=SimpleNamespace(
            get_joint=lambda robot: [np.full(6, .25)]), robots=dict(left='L', right='R'),
            check_self_collision=lambda arm, joints: dict(continuous_joint_chords_certified=True))
        self.events, self.transitions, self.commands, self.receipts = [], [], [], []
        self.force, self.moving, self.fail_contact_after = force, moving, None
        self.hold = PiForceHold(bootstrap_config())
        self.hold.allowed_actors = {ACTOR}
        self.hold.last_action = action(base, right)
        self.hold.finger_closing_directions = {arm: np.array([-1., -1.]) for arm in ('left', 'right')}
        self.freeze_model_fingers = False

    def _write_force_hold_event(self, event):
        self.events.append(deepcopy(event))

    def _write_force_hold_transition(self, transition):
        self.transitions.append(deepcopy(transition))

    def original(self, a, index=0, prediction='old-pred'):
        self.step_id += 1
        if self.hold.last_model_action is not None and not self.freeze_model_fingers:
            for i in range(2):
                delta = float(a[i * 7 + 6] - self.hold.last_model_action[i * 7 + 6]) * .054
                self.obs['finger_positions_m'][i] += delta * -self.hold.finger_closing_directions[('left', 'right')[i]]
        self.obs['states'] = a.copy()
        self.obs['finger_positions_step_id'] = self.step_id
        self.gripper_contact_cache = receipt(self.step_id, force=self.force)
        self.hold.observe_ack(self, a, original=True, student_prediction_id=prediction,
                              proposal_action_index=index)
        self.hold.commit_consider(self, ('left', 'right'))
        self.receipts.append(dict(step_id=self.step_id, source='student', executed_action=a.tolist(),
            student_prediction_id=prediction, proposal_action_index=index, obs=deepcopy(self.obs)))

    def _chunk_step_native(self, actions, **kwargs):
        a = np.asarray(actions[0]).copy()
        self.commands.append(a)
        self.step_id += 1
        self.obs['states'] = a.copy()
        self.obs['finger_positions_step_id'] = self.step_id
        if self.moving:
            self.obs['finger_positions_m'] += .0002
        self.gripper_contact_cache = receipt(self.step_id, force=self.force)
        if self.fail_contact_after is not None and len(self.commands) >= self.fail_contact_after:
            self.gripper_contact_cache['reports_valid'] = False
        self.hold.observe_ack(self, a, original=False)
        row = dict(valid=True, source='student', step_id=self.step_id,
            executed_action=a.tolist(), obs=deepcopy(self.obs), **kwargs)
        self.receipts.append(row)
        return dict(steps=[dict(row, executed_action=a)])


def closing_session(**kwargs):
    s = Session(**kwargs)
    s.original(action(.8), index=0)
    s.original(action(.7), index=1)
    return s


def execute(s, candidate=None):
    candidate = action(.7) if candidate is None else candidate
    arms = s.hold.consider(s, candidate)
    assert arms == ['left']
    s.hold.commit_consider(s, arms)
    return s.hold.run_hold(s, candidate, arms, 0, 50, {'student_prediction_id': 'new-pred'})


def test_initial_closed_scalar_does_not_create_intent():
    s = Session(base=0.)
    s.original(action(0.))
    assert not s.hold.arms['left']['closing']
    assert s.hold.consider(s, action(0.)) == []


def test_executed_history_supports_partial_closing_plateau_and_noise():
    s = closing_session()
    s.original(action(.702))
    assert s.hold.consider(s, action(.702)) == ['left']
    assert s.hold.arms['left']['closing']


def test_unexecuted_closing_candidate_cannot_seed_intent():
    s = Session()
    s.original(action(.8))
    assert s.hold.consider(s, action(.2)) == []
    assert not s.hold.arms['left']['closing']


def test_other_arm_opening_candidate_wins_over_left_hold():
    s = closing_session()
    assert s.hold.consider(s, action(.7, .82)) == []
    assert s.hold.arms['left']['closing']


@pytest.mark.parametrize('case', ['stale', 'episode', 'single', 'table', 'different', 'push', 'invalid'])
def test_invalid_contact_cannot_start_tightening(case):
    s = closing_session()
    r = s.gripper_contact_cache
    if case == 'stale': r['step_id'] -= 1
    elif case == 'episode': r['episode_id'] = 'other'
    elif case == 'single': r['arms']['left']['link8']['objects'] = {}
    elif case == 'table': s.gripper_contact_cache = receipt(s.step_id, actor='/World/table')
    elif case == 'different':
        r['arms']['left']['link8']['objects'] = {'/other': next(iter(r['arms']['left']['link8']['objects'].values()))}
    elif case == 'push': s.gripper_contact_cache = receipt(s.step_id, same_direction=True)
    elif case == 'invalid': r['reports_valid'] = False
    assert s.hold.consider(s, action(.7)) == []
    assert s.commands == []


def test_fixed_margin_ramp_and_immutable_model_history():
    s = closing_session(moving=True)
    before_history = deepcopy(s.hold.arms['left']['model_history'])
    rows, event = execute(s)
    assert len(rows) == 8
    assert event['tighten_steps'] == 3
    assert event['end_reason'] == 'contact_tighten_timeout'
    assert s.hold.arms['left']['model_history'] == before_history
    assert event['base_prediction_id'] == 'old-pred'
    assert event['base_proposal_action_index'] == 1
    assert event['raw_target_ref']['prediction_id'] != event['student_prediction_id']
    reductions = [sample['compensation']['left']['actual_cumulative_gap_reduction_m'] for sample in event['samples']]
    assert np.all(np.diff(reductions) >= -1e-10)
    plan = event['margin_plans']['left']
    assert plan['original_margin_m'] == pytest.approx(.0136, abs=1e-7)
    assert reductions[-1] == pytest.approx(.0068, abs=1e-7)
    assert [x['compensation']['left']['increase_count'] for x in event['samples']] == [1,2,3,3,3,3,3,3]
    assert all(np.array_equal(row['executed_action'][[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], np.full(12, .25)) for row in rows)
    assert all(row['executed_action'][13] == pytest.approx(.8) for row in rows)


@pytest.mark.parametrize('force', [1., 5., 21., 31., 100.])
def test_force_does_not_gate_tightening(force):
    s = closing_session(force=force)
    rows, event = execute(s)
    assert len(rows) == 4
    assert event['tighten_steps'] == 3
    assert event['end_reason'] == 'contact_and_finger_stability_confirmed'
    assert [x['stable_intervals']['left'] for x in event['samples']] == [0, 0, 1, 2]


def test_clipping_records_requested_and_actual_margin():
    s = Session(base=.02, moving=True)
    s.original(action(.03))
    s.original(action(.02))
    rows, event = execute(s, action(.02))
    c = event['samples'][-1]['compensation']['left']
    assert c['requested_cumulative_gap_reduction_m'] > .005
    assert c['actual_cumulative_gap_reduction_m'] == pytest.approx(2 * SINGLE_FINGER_SCALE_M * float(np.float32(.02)))
    assert rows[-1]['executed_action'][6] == 0
    assert event['tighten_steps'] == 1
    plan = event['margin_plans']['left']
    assert plan['clipped_final_margin_m'] < plan['requested_margin_m']


def test_contact_loss_preserves_actual_acks_and_reinfers():
    s = closing_session(moving=True)
    s.fail_contact_after = 2
    rows, event = execute(s)
    assert len(rows) == event['actual_count'] == 2
    assert event['end_reason'] == 'contact_lost_or_invalid'
    assert event['reinfer_required']


def test_opening_cancels_and_new_executed_closing_rearms_without_absolute_gate():
    s = closing_session()
    execute(s)
    assert s.hold.arms['left']['used']
    s.original(action(.72))
    assert not s.hold.arms['left']['used'] and not s.hold.arms['left']['closing']
    s.original(action(.68))
    assert s.hold.consider(s, action(.68)) == ['left']


def test_obsolete_v1_config_is_explicitly_rejected():
    with pytest.raises(TypeError): configured({'close_threshold': .2})
    with pytest.raises(TypeError): configured({'departure_m': .002})
    with pytest.raises(ValueError): ForceHoldConfig(max_extra_steps=9)
    with pytest.raises(TypeError): configured({'gap_step_m': .001})
    with pytest.raises(TypeError): configured({'force_target_N': 5})


def test_target_echo_and_contact_plateau_do_not_replace_observed_physical_closing():
    s = Session(force=21.)
    s.freeze_model_fingers = True
    s.original(action(.8))
    s.original(action(.7))
    s.original(action(.7))
    s.original(action(.7))
    assert s.hold.arms['left']['closing']
    assert s.hold.arms['left']['internal_closing_seen']
    assert not s.hold.arms['left']['physical_closing_seen']
    assert s.hold.consider(s, action(.7)) == []


def test_pairwise_stability_does_not_require_first_to_third_stability():
    s = closing_session()
    native = s._chunk_step_native
    def drifting(actions, **kwargs):
        s.obs['finger_positions_m'] += .00008
        return native(actions, **kwargs)
    s._chunk_step_native = drifting
    rows, event = execute(s)
    assert len(rows) == 4
    assert event['end_reason'] == 'contact_and_finger_stability_confirmed'
    assert np.max(np.abs(np.array(event['samples'][3]['post_finger_positions_m'])
                        - np.array(event['samples'][1]['post_finger_positions_m']))) > .0001


def test_unstable_interval_resets_counter_and_each_finger_is_checked():
    s = closing_session()
    native = s._chunk_step_native
    def moving_one_finger(actions, **kwargs):
        if len(s.commands) == 2:
            s.obs['finger_positions_m'][0] += [.0002, -.0002]
        return native(actions, **kwargs)
    s._chunk_step_native = moving_one_finger
    rows, event = execute(s)
    assert len(rows) == 5
    assert [x['stable_intervals']['left'] for x in event['samples']] == [0,0,0,1,2]


@pytest.mark.parametrize('margin,expected', [(.005,.015),(.010,.015),(.01688,.02532)])
def test_margin_formula(margin, expected):
    a = .5
    target = 2*(-.010+.054*a)
    plan = margin_plan(bootstrap_config(), [(target+margin)/2]*2, a)
    assert plan['requested_margin_m'] == pytest.approx(expected)
    assert plan['original_margin_m'] == pytest.approx(margin)


@pytest.mark.parametrize('overrides', [dict(flywheel_round=2),dict(controller_route='subtask_feedforward_v1'),
                                      dict(checkpoint_sha256='trained'),dict(flywheel_round=True)])
def test_bootstrap_guard(overrides):
    options = dict(tighten_enabled=True, flywheel_round=1, controller_route='auto_infer_v1',
                   checkpoint_sha256=BOOTSTRAP_CHECKPOINT_SHA256)
    with pytest.raises(ValueError):
        ForceHoldConfig(**dict(options, **overrides))


def test_actual_runtime_identity_must_match_bootstrap_declaration():
    with pytest.raises(ValueError):
        validate_tightening_context(bootstrap_config(),controller_route='subtask_feedforward_v1')
    with pytest.raises(ValueError):
        validate_tightening_context(bootstrap_config(),controller_route='auto_infer_v1', checkpoint_sha256='trained')
    validate_tightening_context(ForceHoldConfig(),controller_route='subtask_feedforward_v1',checkpoint_sha256='trained')


def stationary_action():
    a = np.full(14,.25,np.float32)
    a[[6,13]]=[.7,.8]
    return a


def test_model_hold_wins_and_no_extra_actions_are_inserted():
    s = closing_session()
    s.hold.config = ForceHoldConfig()
    for index in range(3):
        a = stationary_action()
        arms = s.hold.consider(s,a)
        assert arms == ['left']
        assert s.hold.prefer_model_hold(s,a,arms)
        s.original(a,index=index)
    assert s.hold.arms['left']['used']
    assert s.hold.consider(s,action(.7)) == []
    assert s.commands == []
    assert s.transitions[-1]['reason'] == 'model_hold_stability_confirmed'


def test_partial_model_hold_needs_only_missing_stable_sample():
    s = closing_session()
    s.hold.config = ForceHoldConfig()
    for index in range(2):
        a = stationary_action()
        assert s.hold.prefer_model_hold(s,a,s.hold.consider(s,a))
        s.original(a,index=index)
    assert not s.hold.prefer_model_hold(s,action(.7),s.hold.consider(s,action(.7)))
    rows,event=execute(s)
    assert len(rows)==1 and event['tighten_steps']==0
    assert event['reused_model_hold_samples']=={'left':2}


def test_bootstrap_native_hold_must_also_meet_frozen_tightening_target():
    s = closing_session()
    a = stationary_action()
    arms = s.hold.consider(s,a)
    assert not s.hold.prefer_model_hold(s,a,arms)
    a[6]=s.hold.plans['left']['final_issued_target']
    assert s.hold.prefer_model_hold(s,a,arms)


def test_source_handoff_does_not_multiply_margin_for_same_contact():
    s = closing_session()
    execute(s)
    s.hold.control_switch(s,'teacher')
    s.original(action(.8))
    s.original(action(.7))
    _,event=execute(s)
    assert not event['margin_plans']['left']['tightening_enabled']
    assert event['tighten_steps']==0


def test_budget_exit_retains_last_real_ack():
    s = closing_session()
    s.metadata['max_episode_steps'] = s.step_id + 1
    rows, event = execute(s)
    assert len(rows) == event['actual_count'] == 1
    assert event['end_reason'] == 'step_budget'


def test_signed_opposite_mimic_coordinates_can_prove_net_closing():
    s = Session()
    s.hold.finger_closing_directions['left'] = np.array([-1., 1.])
    s.original(action(.8))
    s.original(action(.7))
    assert s.hold.arms['left']['physical_closing_seen']
    assert s.hold.consider(s, action(.7)) == ['left']


def test_disabled_force_skips_native_monitor_initialization_and_archive(tmp_path, monkeypatch):
    from . import session as module
    from . import gripper_contact
    from ..test_native_evaluation import NativeLikeEnv
    native = module.RoboDojoSession(NativeLikeEnv(), tmp_path, 'build_tower',
                                   gripper_contact_enabled=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled force must not initialize or sample monitor')
    monkeypatch.setattr(gripper_contact, 'for_environment', forbidden)
    monkeypatch.setattr(module, 'DualKinematics', lambda env: SimpleNamespace(
        check=lambda: {'passed': True}, initialize_self_collision=lambda: {'fixture': True}))
    def observe(record=False):
        native.obs = dict(states=np.zeros(14, np.float32), instruction='Build a tower.',
            finger_positions_available=True, finger_positions_step_id=native.step_id,
            finger_positions_m=np.ones((2, 2)) * .04)
        return native.obs
    monkeypatch.setattr(native, '_observe', observe)
    native.reset(seed=0, source='student', policy_version='test')
    assert native.pi_force_hold is None and native.gripper_contact is None
    assert native._record_gripper_contact() is None
    assert native.get_gripper_contact()['reason'] == 'disabled'
    assert native.metadata['gripper_contact']['enabled'] is False
    assert not (native.episode_dir / 'gripper_contact.jsonl').exists()


def test_enabled_tightening_with_disabled_force_is_configuration_error(tmp_path):
    from .session import RoboDojoSession
    with pytest.raises(ValueError, match='requires gripper contact telemetry'):
        RoboDojoSession(SimpleNamespace(), tmp_path, 'build_tower',
                        pi_force_hold_enabled=True, gripper_contact_enabled=False)


def test_new_runtime_archive_validates_and_original_prediction_stays_immutable(tmp_path):
    import hashlib
    import json
    from ..force_hold_archive import validate_hold_mapping, resolve_base_model_targets
    s=closing_session()
    raw=np.tile(action(.7),(50,1));raw[0]=action(.8)
    path=tmp_path/'original.npz';np.savez(path,raw_actions=raw,actions=raw)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    _,event=execute(s)
    (tmp_path/'pi_force_hold_events.jsonl').write_text(json.dumps(event)+'\n')
    _,events=validate_hold_mapping(tmp_path,s.receipts)
    resolved=resolve_base_model_targets(events,{'old-pred':path},s.receipts)
    assert resolved and hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert event['samples'][0]['actual_command'] != event['samples'][2]['actual_command']
    bad=deepcopy(event);bad['margin_plans']['left']['original_margin_m']+=.001
    (tmp_path/'pi_force_hold_events.jsonl').write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='Frozen margin'):
        validate_hold_mapping(tmp_path,s.receipts)


def test_partial_model_hold_archive_keeps_own_original_indices(tmp_path):
    import json
    from ..force_hold_archive import validate_hold_mapping
    s=closing_session();s.hold.config=ForceHoldConfig()
    for index in (2,3):
        a=stationary_action();assert s.hold.prefer_model_hold(s,a,s.hold.consider(s,a))
        s.original(a,index=index)
    _,event=execute(s)
    (tmp_path/'pi_force_hold_events.jsonl').write_text(json.dumps(event)+'\n')
    validate_hold_mapping(tmp_path,s.receipts)
    assert len(event['steps'])==1


def test_transport_with_constant_gripper_is_not_model_hold():
    s=closing_session();s.hold.config=ForceHoldConfig()
    a=stationary_action();a[0]+=.01
    assert not s.hold.prefer_model_hold(s,a,s.hold.consider(s,a))


def test_slip_during_ramp_preserves_executed_prefix_only():
    s=closing_session();s.fail_contact_after=1
    rows,event=execute(s)
    assert len(rows)==len(event['samples'])==1
    assert event['samples'][0]['compensation']['left']['increase_count']==1
    assert event['end_reason']=='contact_lost_or_invalid'
