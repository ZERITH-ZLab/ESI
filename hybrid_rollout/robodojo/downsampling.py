"""Hybrid execution adapter for explicitly versioned v2/v3/v4 selectors.

Selection only sees the GPT-approved original prefix. Every actual ACK, including
GPT corrections, advances the gripper event state. No object/contact truth used.
Release thresholds live in configs/environments/robodojo.json and refer to
RoboDojo's action mapping/control ticks, not universal robot defaults.
"""
import json
import os
import numpy as np

from .flywheel.action_downsampling import ActionDownsampler, PreviewFK, SelectionConfig, algorithm_version, configuration_dict
from .io import write_json

PROMPT = '''
Execution setting: after your approval, the host may thin ONLY the approved
student prefix using robot FK and gripper-command constraints. The displayed
proposal is the original trajectory. Student steps selects its original prefix
(1..15), not a guaranteed physical tick count. No later targets are substituted.
Each retained target costs one native tick; no artificial waits are added.
Use the actual execution receipt and returned observation to assess progress.
GPT corrections and their explicit gripper closure waits are not downsampled.
'''


def configuration(env=None):
    env = os.environ if env is None else env
    mode = env.get('ROLLOUT_ACTION_DOWNSAMPLING', 'off')
    if mode not in ('off', 'error'):
        raise ValueError('ROLLOUT_ACTION_DOWNSAMPLING must be off or error')
    if mode == 'off':
        return None
    if env.get('ROBODOJO_PI_FORCE_HOLD_ENABLED') == '1':
        raise ValueError('Downsampling requires pi_force_hold disabled')
    return SelectionConfig(**json.loads(env.get('ROLLOUT_DOWNSAMPLING_CONFIG') or '{}'))


def identity(env=None):
    config = configuration(env)
    return {} if config is None else dict(action_downsampling=dict(
        mode='error', algorithm_version=algorithm_version(config), configuration=configuration_dict(config)))


class HybridDownsampling:
    def __init__(self, config, rpc):
        self.fk = PreviewFK(rpc)
        self.selector = ActionDownsampler(config, self.fk)
        self.last_tick = None

    def select(self, actions, count, state, tick, path):
        # All physical execution must have been acknowledged through this adapter.
        if self.last_tick is not None and tick != self.last_tick:
            raise RuntimeError('Downsampling ACK state is stale')
        self.fk.cache.clear()
        before_s, before_n = self.fk.seconds, self.fk.calls
        try:
            indices, report = self.selector.select(actions[:count], state, lookahead=actions)
        except Exception as error:
            write_json(path, dict(mode='error', status='failed', start_tick=tick,
                error=f'{type(error).__name__}: {error}'))
            raise
        report.update(start_tick=tick, status='selected',
            fk_wall_seconds=self.fk.seconds-before_s, fk_rpc_calls=self.fk.calls-before_n)
        write_json(path, report)
        return indices, report

    def acknowledge(self, trace, state, tick):
        actions = [r['executed_action'] for r in trace if r['valid']]
        self.selector.acknowledge(np.asarray(actions).reshape(-1, 14), state)
        self.last_tick = tick
