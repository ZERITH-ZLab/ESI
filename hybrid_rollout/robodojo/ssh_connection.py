"""Simulator login identity; resource locks still describe the physical host."""
import re


def simulator_user(env):
    user = env.get('ROLLOUT_SIM_SSH_USER', 'root')
    if not isinstance(user, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\$?', user):
        raise ValueError('Invalid simulator SSH username')
    return user


def simulator_target(env):
    return f"{simulator_user(env)}@{env['ROLLOUT_SIM_HOST']}"


def identity_endpoint(env):
    # Two logins on one host may resolve different installations/permissions.
    return (env['ROLLOUT_SIM_HOST'], env['ROLLOUT_SIM_SSH_PORT'], simulator_user(env),
            env.get('ROLLOUT_SIM_ROBODOJO_SOURCE'), env.get('ROLLOUT_SIM_PYTHON'))
