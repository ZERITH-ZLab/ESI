from .ssh_connection import simulator_target, identity_endpoint


def test_default_and_explicit_login():
    env = dict(ROLLOUT_SIM_HOST='host', ROLLOUT_SIM_SSH_PORT='22')
    assert simulator_target(env) == 'root@host'
    other = dict(env, ROLLOUT_SIM_SSH_USER='robot')
    assert simulator_target(other) == 'robot@host'
    assert identity_endpoint(env) != identity_endpoint(other)


def test_identity_cache_distinguishes_installs():
    env = dict(ROLLOUT_SIM_HOST='host', ROLLOUT_SIM_SSH_PORT='22',
               ROLLOUT_SIM_SSH_USER='robot', ROLLOUT_SIM_ROBODOJO_SOURCE='/opt/RoboDojo')
    assert identity_endpoint(env) != identity_endpoint(dict(env, ROLLOUT_SIM_ROBODOJO_SOURCE='/srv/RoboDojo'))


def test_runtime_fingerprint_uses_configured_login(monkeypatch):
    import pytest
    from . import dual_campaign
    called = []

    def ssh(args, **kwargs):
        called.append(args)
        return 'stop-before-checkpoint-access'

    monkeypatch.setattr(dual_campaign.subprocess, 'check_output', ssh)
    env = dict(ROLLOUT_SIM_HOST='host', ROLLOUT_SIM_SSH_PORT='22',
               ROLLOUT_SIM_SSH_USER='robot', ROLLOUT_SIM_SSH_KEY='/keys/sim',
               ROLLOUT_SIM_ROBODOJO_SOURCE='/opt/RoboDojo', ROLLOUT_SIM_PYTHON='/env/bin/python')
    with pytest.raises(ValueError, match='Invalid remote source fingerprint'):
        dual_campaign.runtime_identity(env)
    assert len(called) == 1
    assert 'robot@host' in called[0]
    assert 'root@host' not in called[0]
