"""Actual hinge FK and FCL geometric tests of continuous hybrid certification.

The provider fixture is deliberately synthetic, replacing only pinned asset
loading. Asset identity is covered by test_collision_assets.py; real geometry
can be checked with the cooked_collision --asset-root preflight.
"""
import numpy as np
import pytest
import yaml

from hybrid_rollout.robodojo.robodojo_server import cooked_collision
from hybrid_rollout.robodojo.robodojo_server.self_collision import RobotSelfCollision, SelfCollisionError


@pytest.fixture
def guard(tmp_path, monkeypatch):
    import fcl
    urdf = tmp_path / 'arm.urdf'
    urdf.write_text('''<robot name="swing"><link name="base_link"/><link name="tip"/>
      <joint name="hinge" type="revolute"><parent link="base_link"/><child link="tip"/>
      <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
      <limit lower="-4" upper="4" effort="10" velocity="1"/></joint></robot>''')
    config = tmp_path / 'collision.yml'
    config.write_text(yaml.safe_dump(dict(robot_cfg=dict(kinematics=dict(
        base_link='base_link', collision_link_names=['base_link', 'tip'],
        collision_spheres={name:[dict(center=[1., 0., 0.], radius=.1)]
                           for name in ['base_link', 'tip']},
        self_collision_ignore={}, self_collision_buffer={'base_link':-1., 'tip':-1.})))))
    native = cooked_collision.CookedCollision.__new__(cooked_collision.CookedCollision)
    native.fcl = fcl
    native.geometry = {'scope': 'synthetic_FCL_spheres_for_continuous_contract'}
    native.vertices = {name: np.array([[1.1, 0., 0.]]) for name in ['base_link', 'tip']}
    native.distance_request = fcl.DistanceRequest(enable_signed_distance=False)
    native.collision_request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    objects = {name: fcl.CollisionObject(fcl.Sphere(.1)) for name in ['base_link', 'tip']}

    def query(transforms):
        for name, value in transforms.items():
            center = value[:3, :3] @ np.array([1., 0., 0.]) + value[:3, 3]
            objects[name].setTransform(fcl.Transform(center))
        return dict(gap_m=native._distance(objects['base_link'], objects['tip']),
                    link_pair=['base_link', 'tip'], geometry=native.geometry)

    native.query = query
    monkeypatch.setattr(cooked_collision, 'CookedCollision', lambda *args: native)
    return RobotSelfCollision(urdf, config, ['hinge'])


def test_safe_endpoints_do_not_hide_intermediate_collision(guard):
    assert guard.clearance([-np.pi/2])[0] > 1
    assert guard.clearance([np.pi/2])[0] > 1
    with pytest.raises(SelfCollisionError, match='clearance') as caught:
        guard.validate_path([[-np.pi/2], [np.pi/2]])
    assert caught.value.details['reason'] == 'sampled_clearance'
    assert caught.value.details['location']['kind'] == 'joint_chord'


def test_negative_native_buffers_do_not_disable_collision(guard):
    assert guard.clearance([0.])[0] == pytest.approx(-.2)
    with pytest.raises(SelfCollisionError, match='clearance'):
        guard.validate_path([[0.]])


def test_clear_motion_is_certified_without_modifying_commands(guard):
    joints = np.array([[-np.pi], [-np.pi/2]])
    original = joints.copy()
    result = guard.validate_path(joints)
    assert result['continuous_joint_chords_certified']
    assert result['checked_states'] > 2
    np.testing.assert_array_equal(joints, original)


def test_exact_1mm_clearance_is_rejected(guard, monkeypatch):
    monkeypatch.setattr(guard.native, 'query', lambda transforms: dict(
        gap_m=.001, link_pair=['base_link', 'tip'], geometry=guard.native.geometry))
    with pytest.raises(SelfCollisionError) as caught:
        guard.validate_path([[0.]])
    assert caught.value.details['reason'] == 'sampled_clearance'
    assert caught.value.details['gap_m'] == .001


@pytest.mark.parametrize('fault', ['nan', 'exception'])
def test_native_query_fault_fails_closed(guard, monkeypatch, fault):
    def query(transforms):
        if fault == 'exception':
            raise RuntimeError('native fault')
        return dict(gap_m=float('nan'))
    monkeypatch.setattr(guard.native, 'query', query)
    with pytest.raises(SelfCollisionError) as caught:
        guard.validate_path([[0.]])
    assert caught.value.details['reason'] == 'invalid_native_query'
    assert caught.value.details['error_type'] == ('RuntimeError' if fault == 'exception' else 'ValueError')


@pytest.mark.parametrize('budget, reason', [
    ({'max_depth': 0}, 'unresolved_interval'),
    ({'max_checks': 2}, 'resource_budget'),
])
def test_uncertified_interval_budget_is_rejected(guard, budget, reason):
    with pytest.raises(SelfCollisionError) as caught:
        guard.validate_path([[-np.pi], [-np.pi/2]], **budget)
    assert caught.value.details['reason'] == reason
