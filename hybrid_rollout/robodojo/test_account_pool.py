"""Offline account scheduling checks: no real logins, model calls or simulators."""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from . import account_pool as ap
from .codex_backend.profiles import account_home, account_lock
from .deployment import source_identity
from .io import write_json
from .test_dual_deployment import configuration


def setup_pool(tmp_path, monkeypatch, names=('codex_a_4', 'codex_a_5'), same=False):
    config = tmp_path/'pool.json'
    write_json(config, dict(profiles=list(names), check_interval_seconds=10, quota_recheck_seconds=10))
    env = dict(ROLLOUT_ACCOUNT_POOL_FILE=str(config), ROLLOUT_AUTH_ROOT=str(tmp_path/'auth'), CODEX_BIN='unused')
    for i, name in enumerate(names):
        path = account_home(name, env['ROLLOUT_AUTH_ROOT'])/'auth.json'
        path.parent.mkdir(parents=True)
        path.parent.parent.chmod(0o700)
        write_json(path, dict(tokens=dict(account_id='same' if same else str(i))))
    monkeypatch.setattr(ap, 'validate_credential', lambda name, shared: account_home(name, shared)/'auth.json')
    return env, config


def available(name):
    return dict(remaining_percent=50, windows=[], reached_type=None)


@pytest.mark.parametrize('message,expected', [
    ('RuntimeError: Codex error: usage limit reached', 'quota_interrupted'),
    ('RuntimeError: {"codexErrorInfo":"usageLimitExceeded"}', 'quota_interrupted'),
    ('RuntimeError: insufficient_quota', 'quota_interrupted'),
    ('RuntimeError: token expired', 'auth_interrupted'),
    ('RuntimeError: 429 too many requests', None),
    ('RuntimeError: Selected model is at capacity', None),
    ('RuntimeError: request timed out', None),
    ('quota variable\nValueError: IK failure', None),
])
def test_terminal_classification(message, expected):
    assert ap.failure_kind(message) == expected


def test_busy_lock_releases_and_is_not_a_permanent_flag(tmp_path, monkeypatch):
    env, _ = setup_pool(tmp_path, monkeypatch)
    pool = ap.AccountPool(env, probe=available)
    with account_lock('codex_a_4', env['ROLLOUT_AUTH_ROOT']):
        with pool.acquire('case') as picked:
            assert picked['auth_profile'] == 'codex_a_5'
    with pool.acquire('next') as picked:
        assert picked['auth_profile'] == 'codex_a_4'


def test_same_account_alias_shares_quota_and_recovers(tmp_path, monkeypatch, capsys):
    env, _ = setup_pool(tmp_path, monkeypatch, same=True)
    clock = [100.]
    probes = []
    def probe(name):
        probes.append(name)
        return available(name)
    pool = ap.AccountPool(env, probe=probe, clock=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0]+s))
    with pool.acquire('first') as picked:
        pool.block(picked['account_fingerprint'], 'quota_interrupted')
    with pool.acquire('retry') as picked:
        assert picked['auth_profile'] == 'codex_a_4'
    assert probes == ['codex_a_4', 'codex_a_4']
    output = capsys.readouterr().out
    assert '任务未运行' in output and 'account_pool_wait' in output
    assert clock[0] == 110


def test_identity_lock_prevents_alias_concurrency(tmp_path, monkeypatch):
    env, _ = setup_pool(tmp_path, monkeypatch, same=True)
    pool = ap.AccountPool(env, probe=available)
    with pool.acquire('one'):
        other = ap.AccountPool(env, probe=available, sleep=lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            with other.acquire('two'):
                pytest.fail('same physical account must not be allocated twice')


def test_hot_reload_empty_list_and_probe_failure(tmp_path, monkeypatch):
    env, config = setup_pool(tmp_path, monkeypatch)
    write_json(config, dict(profiles=[], check_interval_seconds=10, quota_recheck_seconds=10))
    clock = [0]
    def sleep(s):
        clock[0] += s
        if clock[0] == 10:
            write_json(config, dict(profiles=['codex_a_5'], check_interval_seconds=10, quota_recheck_seconds=10))
    attempts = []
    def probe(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise TimeoutError('must not leak secret error bodies')
        return available(name)
    pool = ap.AccountPool(env, probe=probe, clock=lambda: clock[0], sleep=sleep)
    with pool.acquire('case') as picked:
        assert picked['auth_profile'] == 'codex_a_5'
    assert clock[0] == 20 and len(attempts) == 2


def test_reset_time_is_recheck_hint_not_automatic_availability():
    assert ap.next_check(dict(windows=[dict(usedPercent=100, resetsAt=105)]), 100, 600) == 120
    assert ap.next_check(dict(windows=[dict(usedPercent=100, resetsAt=9000)]), 100, 600) == 700


def test_exhausted_worker_waits_without_taking_running_account(tmp_path, monkeypatch):
    env, _ = setup_pool(tmp_path, monkeypatch)
    running = ap.AccountPool(env, probe=available)
    waiting = ap.AccountPool(env, probe=available,
        sleep=lambda _: (_ for _ in ()).throw(InterruptedError('test ends wait')))
    with running.acquire('running') as owner:
        with waiting.acquire('about_to_exhaust') as second:
            assert owner['account_fingerprint'] != second['account_fingerprint']
            waiting.block(second['account_fingerprint'], 'quota_interrupted')
        observations = []
        with pytest.raises(InterruptedError):
            with waiting.acquire('retry', on_wait=observations.append):
                pytest.fail('Must not steal running account or reuse exhausted one')
        assert observations[0]['reasons']['codex_a_4'] == 'busy'
        assert observations[0]['reasons']['codex_a_5'] == 'quota_interrupted'
    with waiting.acquire('after_other_finishes') as owner:
        assert owner['auth_profile'] == 'codex_a_4'


def test_quota_probe_recovers_after_wait(tmp_path, monkeypatch):
    env, _ = setup_pool(tmp_path, monkeypatch, names=('codex_a_4',))
    now = [0]
    def probe(name):
        return dict(remaining_percent=0 if now[0] < 10 else 50, windows=[])
    pool = ap.AccountPool(env, probe=probe, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0]+s))
    with pool.acquire('case'):
        assert now[0] == 10


@pytest.mark.parametrize('failure', ['quota', 'unknown', 'unclean'])
def test_pool_retries_only_account_failure_with_new_attempt(tmp_path, monkeypatch, failure):
    env, _ = configuration(tmp_path)
    from .dual_campaign import configure
    env, cases = configure(env)
    root = tmp_path/'repo'
    (root/'hybrid_rollout/robodojo').mkdir(parents=True)
    (root/'hybrid_rollout/robodojo/sample.py').write_text('# frozen\n')
    runtime = dict(source_sha256=source_identity(root), evaluation_method='pi05_plus_gpt')
    env.update(CODE_ROOT=str(root), ROLLOUT_RESUME_IDENTITY=json.dumps(runtime),
               ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt', ROLLOUT_AUTH_ROOT=str(tmp_path/'auth'))
    class Pool:
        config_path = tmp_path/'pool.json'
        calls = 0
        blocked = []
        @contextmanager
        def acquire(self, job, on_wait=None):
            self.calls += 1
            name = 'codex_a_4' if self.calls == 1 else 'codex_a_5'
            yield dict(auth_profile=name, account_fingerprint=name, credential_revision='rev', fds=(1,2))
        def block(self, identity, reason, revision):
            self.blocked.append((identity, reason))
    pool = Pool()
    attempted = []
    def runner(child, fds):
        attempted.append((child['ROLLOUT_CASE_ID'], child['ROLLOUT_AUTH_PROFILE']))
        n = child['ROLLOUT_ATTEMPT']
        archive = tmp_path/f'archive_{n}'
        (archive/'dual_machine').mkdir(parents=True)
        (archive/'dual_machine/remote_cleanup_status.txt').write_text('3' if failure == 'unclean' else '0')
        receipt = dict(all_owned_processes_exited=True, ports_released=True)
        row = dict(case_id=child['ROLLOUT_CASE_ID'], archive=str(archive), reset=receipt,
                   state='failed' if n=='0' else 'completed',
                   failure_kind='quota_interrupted' if n=='0' and failure != 'unknown' else None)
        path = tmp_path/f'results/test/_replicas/replica_8/attempt_{n}'
        path.mkdir(parents=True)
        write_json(path/'batch.json', dict(final_reset=receipt, episodes=[row], resume_identity=runtime))
        return 1 if n=='0' else 0
    if failure != 'quota':
        with pytest.raises(RuntimeError):
            ap.run_pool(env, cases[:2], pool=pool, runner=runner)
        assert len(attempted) == 1 and not pool.blocked
        return
    assert ap.run_pool(env, cases[:2], pool=pool, runner=runner) == 0
    assert attempted == [(cases[0]['case_id'],'codex_a_4'), (cases[0]['case_id'],'codex_a_5'), (cases[1]['case_id'],'codex_a_5')]
    assert pool.blocked == [('codex_a_4','quota_interrupted')]
    summary = json.loads(next((tmp_path/'results/test/_account_pool').glob('*/pool_run.json')).read_text())
    assert summary['state'] == 'completed'
    assert len(summary['jobs'][0]['attempts']) == 2


def test_cleanup_requires_both_hosts(tmp_path):
    receipt = dict(all_owned_processes_exited=True, ports_released=True)
    row = dict(archive=str(tmp_path), reset=receipt)
    assert not ap.clean_result(dict(final_reset=receipt), row)
    (tmp_path/'dual_machine').mkdir()
    (tmp_path/'dual_machine/remote_cleanup_status.txt').write_text('0')
    assert ap.clean_result(dict(final_reset=receipt), row)
    assert not ap.clean_result(dict(final_reset={}), row)


def grouped_pool(tmp_path, monkeypatch):
    names = ('codex_a_5', 'codex_a_6', 'codex_a_7', 'codex_b')
    env, config = setup_pool(tmp_path, monkeypatch, names=names)
    for name in names:
        write_json(account_home(name, env['ROLLOUT_AUTH_ROOT'])/'auth.json',
                   dict(tokens=dict(account_id='primary' if name != 'codex_b' else 'backup',
                                    refresh_token='session-'+name)))
    value = dict(accounts=[
        dict(name='reserve', profiles=['codex_b'], max_concurrent=1, priority='backup'),
        dict(name='main', profiles=list(names[:3]), max_concurrent=2, priority='primary')],
        check_interval_seconds=10, quota_recheck_seconds=10)
    write_json(config, value)
    return env, config, value


def test_two_slots_cap_shared_quota_and_backup_priority(tmp_path, monkeypatch):
    env, config, value = grouped_pool(tmp_path, monkeypatch)
    pool = ap.AccountPool(env, probe=available)
    with pool.acquire('one') as first:
        with pool.acquire('two') as second:
            assert [first['auth_profile'], second['auth_profile']] == ['codex_a_5', 'codex_a_6']
            assert first['account_fingerprint'] == second['account_fingerprint']
            assert {first['slot'], second['slot']} == {0, 1}
            with pool.acquire('three') as third:
                assert third['auth_profile'] == 'codex_b'  # Third alias cannot create a third lease.
            pool.block(first['account_fingerprint'], 'quota_interrupted')
        with pool.acquire('after_second_finishes') as third:
            assert third['auth_profile'] == 'codex_b'  # Cooldown is account-wide.
    assert not list(pool.root.glob('*.owner.json'))


def test_group_identity_mismatch_and_copied_sessions_rejected(tmp_path, monkeypatch):
    env, config, value = grouped_pool(tmp_path, monkeypatch)
    pool = ap.AccountPool(env, probe=available)
    credential = account_home('codex_a_6', env['ROLLOUT_AUTH_ROOT'])/'auth.json'
    data = json.loads(credential.read_text())
    data['tokens']['account_id'] = 'unexpected'
    write_json(credential, data)
    with pool.acquire('wrong-login') as picked:
        assert picked['auth_profile'] == 'codex_b'
    data['tokens'].update(account_id='primary', refresh_token='session-codex_a_5')
    write_json(credential, data)
    with pool.acquire('copied-login') as picked:
        assert picked['auth_profile'] == 'codex_b'


def test_group_cap_reduction_waits_for_old_slots_and_legacy_leases_block_new(tmp_path, monkeypatch):
    env, config, value = grouped_pool(tmp_path, monkeypatch)
    pool = ap.AccountPool(env, probe=available)
    with pool.acquire('active'):
        value['accounts'][1]['max_concurrent'] = 1
        write_json(config, value)
        with pool.acquire('reduced') as picked:
            assert picked['auth_profile'] == 'codex_b'
    with pool.acquire('serial') as picked:
        assert picked['auth_profile'] == 'codex_a_5'
        value['accounts'][1]['max_concurrent'] = 2
        write_json(config, value)
        with pool.acquire('shared_guard') as other:
            assert other['auth_profile'] == 'codex_b'


def test_invalid_groups_do_not_silently_change_limits(tmp_path, monkeypatch):
    from copy import deepcopy
    _, config, value = grouped_pool(tmp_path, monkeypatch)
    for change in ('duplicate', 'too_high', 'invalid_priority', 'mixed_schema'):
        data = deepcopy(value)
        if change == 'duplicate': data['accounts'][0]['profiles'] = ['codex_a_5']
        elif change == 'too_high': data['accounts'][1]['max_concurrent'] = 5
        elif change == 'invalid_priority': data['accounts'][0]['priority'] = 'typo'
        else: data['profiles'] = ['codex_b']
        write_json(config, data)
        with pytest.raises(ValueError): ap.read_config(config)


def test_account_group_cannot_exceed_eight_even_with_more_profiles(tmp_path):
    path = tmp_path/'pool.json'
    write_json(path, dict(accounts=[dict(name='too_many',
        profiles=[f'codex_a_{i}' for i in range(7, 16)], max_concurrent=9)]))
    with pytest.raises(ValueError, match='1..8'):
        ap.read_config(path)


@pytest.mark.parametrize('group,start,capacity', [('a', 12, 4), ('b', 12, 4), ('b', 8, 8)])
def test_group_slots_prioritized_capped_and_released(tmp_path, monkeypatch, group, start, capacity):
    from contextlib import ExitStack
    names = tuple(f'codex_{group}_{i}' for i in range(start, 16)) + ('codex_b',)
    env, config = setup_pool(tmp_path, monkeypatch, names=names)
    for name in names:
        write_json(account_home(name, env['ROLLOUT_AUTH_ROOT'])/'auth.json',
                   dict(tokens=dict(account_id='new' if name != 'codex_b' else 'old',
                                    refresh_token='independent-'+name)))
    write_json(config, dict(accounts=[
        dict(name='new', profiles=list(names[:capacity]), max_concurrent=capacity, priority='primary'),
        dict(name='old', profiles=['codex_b'], max_concurrent=1, priority='primary')]))
    pool = ap.AccountPool(env, probe=available)
    with ExitStack() as leases:
        accounts = [leases.enter_context(pool.acquire(str(i))) for i in range(capacity)]
        assert [a['auth_profile'] for a in accounts] == list(names[:capacity])
        assert {a['slot'] for a in accounts} == set(range(capacity))
        assert len({a['account_fingerprint'] for a in accounts}) == 1
        with pool.acquire('over-capacity') as extra:
            assert extra['auth_profile'] == 'codex_b'
    with pool.acquire('released') as reused:
        assert reused['auth_profile'] == names[0]
        pool.block(reused['account_fingerprint'], 'quota_interrupted')
    with pool.acquire('shared-cooldown') as fallback:
        assert fallback['auth_profile'] == 'codex_b'
    assert not list(pool.root.glob('*.owner.json'))
