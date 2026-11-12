"""Task-balanced case-weighted sampling, plus the historical single-task hierarchy.

Compile the hierarchy into exact per-window probabilities once. Drawing from
this distribution is equivalent to traversing the tree, with replacement.
No observation/action loading and no changes to training eligibility occur here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from data_processing.subtasks.io import digest, read, require
from data_processing.training_index.build import checksum, validate
from . import VERSION

DEFAULT_CONFIG = dict(gpt_start_ratio=.5, random_ratio=.25, successful_episode_ratio=.7, seed=0)
MULTITASK_RATIO_CONFIG = dict(strategy='task_case_weighted', gpt_start_ratio=.5, random_ratio=.5,
                             successful_case_weight=2., unsuccessful_case_weight=1., seed=0)
MULTITASK_CONFIG = dict(strategy='task_case_control_weighted', gpt_start_weight=4., pi05_start_weight=1.,
                       random_ratio=.5, successful_case_weight=2., unsuccessful_case_weight=1., seed=0)


def config_template(cfg):
    return {'task_round_case_control_weighted': dict(MULTITASK_CONFIG,
                strategy='task_round_case_control_weighted', round_weights={}),
            'task_case_weighted': MULTITASK_RATIO_CONFIG,
            'task_case_control_weighted': MULTITASK_CONFIG}.get(cfg.get('strategy'), DEFAULT_CONFIG)


def load_config(path=None):
    supplied = read(path) if path else {}
    require(isinstance(supplied, dict), 'Sampling configuration must be an object')
    cfg = dict(config_template(supplied))
    require(isinstance(supplied, dict) and not set(supplied)-set(cfg), 'Unknown sampling config keys')
    cfg.update(supplied)
    return checked_config(cfg)


def checked_config(cfg):
    template = config_template(cfg) if isinstance(cfg, dict) else DEFAULT_CONFIG
    require(isinstance(cfg, dict) and set(cfg) == set(template), 'Incomplete/unknown sampling configuration')
    for name, value in cfg.items():
        if name == 'strategy':
            require(value in ('task_case_weighted', 'task_case_control_weighted', 'task_round_case_control_weighted'), 'Invalid strategy')
        elif name == 'round_weights':
            require(isinstance(value, dict) and len(value) >= 2 and all(
                isinstance(k, str) and k and type(v) in (int, float) and math.isfinite(v) and v > 0
                for k, v in value.items()), 'Invalid round_weights')
        elif name.endswith('_weight'):
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, f'Invalid weight: {name}')
        elif name == 'seed':
            require(type(value) is int and value >= 0, 'Invalid seed')
        else:
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f'Invalid probability: {name}')
    return dict(cfg)


class HierarchicalSampler:
    def __init__(self, index_root, config=None):
        self.index_root = Path(index_root).resolve()
        # Hash-check the completed index; raw image/MCAP validation belongs to the loader.
        validate(self.index_root, check_sources=False)
        with sqlite3.connect(f'{(self.index_root/"index.sqlite").as_uri()}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            episodes = {r['episode_key']: dict(r) for r in db.execute('SELECT * FROM episodes')}
            rows = [dict(r) for r in db.execute('''SELECT sample_id,episode_key,start_step,start_control,
                valid_length,gpt_valid_steps,is_decision_start FROM training_samples ORDER BY sample_id''')]
        for episode in episodes.values():
            case = json.loads(episode['metadata_json'])['evaluation_case']
            require(case.get('case_id') and case.get('task') == episode['task']
                    and case.get('variant') == episode['variant'], 'Missing/inconsistent case identity')
            # Preserve task/variant/seed/layout identity even when external names repeat.
            episode['case_key'] = json.dumps([episode['task'], episode['variant'], case['case_id'],
                case.get('eval_seed'), case.get('layout_id')], separators=(',', ':'))
            episode['case_id'] = case['case_id']
        self._initialize(rows, episodes, config or DEFAULT_CONFIG,
                         checksum(self.index_root/'index.sqlite'))

    def _initialize(self, rows, episodes, config, index_sha256):
        self.config = checked_config(config)
        self.rows, self.episodes = rows, episodes
        require(rows and len({r['sample_id'] for r in rows}) == len(rows), 'Empty/duplicate sample index')
        mixed = self.config.get('strategy') == 'task_round_case_control_weighted'
        if not mixed:
            require(len({e['policy_version'] for e in episodes.values()}) == 1 and
                    len({e['collection_round'] for e in episodes.values()}) == 1,
                    'Multiple policies/rounds need an explicit mixing policy; this sampler handles one round')
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
        for i, row in enumerate(rows):
            ep = episodes[row['episode_key']]
            require(row['start_control'] in ('gpt', 'pi05') and ep['variant'] in ('standard', 'random'),
                    'Unsupported control/variant; no silent sample exclusion')
            require(ep['episode_success'] in (0, 1) and row['valid_length'] > 0 and
                    0 <= row['gpt_valid_steps'] <= row['valid_length'], 'Invalid sample metadata')
            outcome = 'success' if ep['episode_success'] else 'failure'
            tree[row['start_control']][ep['variant']][ep['case_key']][outcome][row['episode_key']].append(i)
        self.fallbacks, self.case_availability, self.leaves = [], [], []
        self.probabilities = np.zeros(len(rows), dtype=np.float64)
        cweights = dict(gpt=self.config.get('gpt_start_ratio', .5), pi05=1-self.config.get('gpt_start_ratio', .5))
        vweights = dict(standard=1-self.config['random_ratio'], random=self.config['random_ratio'])
        if mixed:
            self._task_round_case_control_weighted(rows, episodes, index_sha256)
            legacy_controls = {}
        elif self.config.get('strategy') == 'task_case_control_weighted':
            self._task_case_control_weighted(rows, episodes, vweights)
            legacy_controls = {}
        elif self.config.get('strategy') == 'task_case_weighted':
            self._task_case_weighted(rows, episodes, cweights, vweights)
            legacy_controls = {}
        else:
            legacy_controls = self._weights(cweights, tree, 'control', {})
        oweights = dict(success=self.config.get('successful_episode_ratio', .7),
                        failure=1-self.config.get('successful_episode_ratio', .7))
        for control, cp in legacy_controls.items():
            for variant, vp in self._weights(vweights, tree[control], 'variant', dict(control=control)).items():
                cases = tree[control][variant]
                universe = {e['case_key'] for e in episodes.values() if e['variant'] == variant}
                self.case_availability.append(dict(control=control, variant=variant,
                    available_cases=sorted(cases), missing_cases=sorted(universe-set(cases))))
                for case_key, outcomes in sorted(cases.items()):
                    case_probability = cp*vp/len(cases)
                    context = dict(control=control, variant=variant, case_key=case_key)
                    for outcome, op in self._weights(oweights, outcomes, 'episode_outcome', context).items():
                        eps = outcomes[outcome]
                        for key, positions in sorted(eps.items()):
                            probability = case_probability*op/len(eps)
                            self.probabilities[positions] = probability/len(positions)
                            self.leaves.append(dict(**context, case_id=episodes[key]['case_id'],
                                outcome=outcome, episode_key=key, windows=len(positions),
                                probability=probability, per_window_probability=probability/len(positions)))
        require(np.isclose(self.probabilities.sum(), 1, atol=1e-12), 'Probability mass does not sum to one')
        self.probabilities /= self.probabilities.sum()
        self._cdf = np.cumsum(self.probabilities)
        self._cdf[-1] = 1.0
        self.sample_ids = np.asarray([r['sample_id'] for r in rows], dtype=np.int64)
        self.rng = np.random.Generator(np.random.PCG64(self.config['seed']))
        self.draw_count = 0
        code_hash = digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
        self.fingerprint = digest(dict(version=VERSION,index_sha256=index_sha256,config=self.config,
                                       implementation_sha256=code_hash))
        self.index_sha256 = index_sha256

    def _task_case_weighted(self, rows, episodes, cweights, vweights):
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))
        for i, row in enumerate(rows):
            ep = episodes[row['episode_key']]
            outcome = 'success' if ep['episode_success'] else 'failure'
            tree[ep['task']][ep['variant']][row['start_control']][outcome][ep['case_key']][row['episode_key']].append(i)
        tasks = {ep['task'] for ep in episodes.values()}
        require(set(tree) == tasks, f'Tasks without trainable windows: {sorted(tasks-set(tree))}')
        weights = dict(success=self.config['successful_case_weight'], failure=self.config['unsuccessful_case_weight'])
        for task in sorted(tree):
            variants = tree[task]
            for variant, vp in self._weights(vweights, variants, 'variant', dict(task=task)).items():
                for control, cp in self._weights(cweights, variants[variant], 'control', dict(task=task,variant=variant)).items():
                    outcomes = variants[variant][control]
                    denominator = sum(weights[o]*len(cases) for o,cases in outcomes.items())
                    available = {case for cases in outcomes.values() for case in cases}
                    universe = {e['case_key'] for e in episodes.values() if e['task']==task and e['variant']==variant}
                    self.case_availability.append(dict(task=task,variant=variant,control=control,
                        available_cases=sorted(available),missing_cases=sorted(universe-available)))
                    for outcome, cases in sorted(outcomes.items()):
                        for case, eps in sorted(cases.items()):
                            case_mass = vp*cp/len(tasks)*weights[outcome]/denominator
                            for key, positions in sorted(eps.items()):
                                probability = case_mass/len(eps)
                                self.probabilities[positions] = probability/len(positions)
                                self.leaves.append(dict(task=task,variant=variant,control=control,case_key=case,
                                    case_id=episodes[key]['case_id'],outcome=outcome,episode_key=key,
                                    windows=len(positions),probability=probability,
                                    per_window_probability=probability/len(positions)))

    def _task_round_case_control_weighted(self, rows, episodes, index_sha256):
        """Explicit round mixture, with the same uniform task mass in every round."""
        weights = self.config['round_weights']
        require(set(weights) == {e['collection_round'] for e in episodes.values()},
                'round_weights must match all indexed collection rounds exactly')
        tasks = {e['task'] for e in episodes.values()}
        child_config = {k: v for k, v in self.config.items() if k != 'round_weights'}
        child_config['strategy'] = 'task_case_control_weighted'
        for name, weight in sorted(weights.items()):
            eps = {k: e for k, e in episodes.items() if e['collection_round'] == name}
            require({e['task'] for e in eps.values()} == tasks,
                    'Every mixed round must contain every task; no silent task/round reweighting')
            positions = [i for i, row in enumerate(rows) if row['episode_key'] in eps]
            child = object.__new__(HierarchicalSampler)
            child._initialize([rows[i] for i in positions], eps, child_config, index_sha256)
            mass = weight / sum(weights.values())
            self.probabilities[positions] = child.probabilities * mass
            self.fallbacks.extend(dict(x, collection_round=name) for x in child.fallbacks)
            self.case_availability.extend(dict(x, collection_round=name) for x in child.case_availability)
            self.leaves.extend(dict(x, collection_round=name,
                probability=x['probability']*mass, per_window_probability=x['per_window_probability']*mass)
                for x in child.leaves)

    def _task_case_control_weighted(self, rows, episodes, vweights):
        # Task -> variant -> outcome-weighted case -> uniform episode ->
        # weighted window. Control mass is proportional to available windows,
        # so a rare GPT fragment does not inherit a fixed half of its case.
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))
        for i, row in enumerate(rows):
            ep = episodes[row['episode_key']]
            outcome = 'success' if ep['episode_success'] else 'failure'
            tree[ep['task']][ep['variant']][outcome][ep['case_key']][row['episode_key']][row['start_control']].append(i)
        tasks = {ep['task'] for ep in episodes.values()}
        require(set(tree) == tasks, f'Tasks without trainable windows: {sorted(tasks-set(tree))}')
        weights = dict(success=self.config['successful_case_weight'], failure=self.config['unsuccessful_case_weight'])
        controls = dict(gpt=self.config['gpt_start_weight'], pi05=self.config['pi05_start_weight'])
        for task in sorted(tree):
            variants = tree[task]
            for variant, vp in self._weights(vweights, variants, 'variant', dict(task=task)).items():
                outcomes = variants[variant]
                denominator = sum(weights[o]*len(cases) for o,cases in outcomes.items())
                available = {case for cases in outcomes.values() for case in cases}
                universe = {e['case_key'] for e in episodes.values() if e['task']==task and e['variant']==variant}
                self.case_availability.append(dict(task=task,variant=variant,
                    available_cases=sorted(available),missing_cases=sorted(universe-available)))
                for outcome, cases in sorted(outcomes.items()):
                    for case, eps in sorted(cases.items()):
                        case_mass = vp/len(tasks)*weights[outcome]/denominator
                        for key, by_control in sorted(eps.items()):
                            ep_mass = case_mass/len(eps)
                            weighted_count = sum(controls[c]*len(positions) for c,positions in by_control.items())
                            for control, positions in sorted(by_control.items()):
                                per_window = ep_mass*controls[control]/weighted_count
                                self.probabilities[positions] = per_window
                                self.leaves.append(dict(task=task,variant=variant,control=control,case_key=case,
                                    case_id=episodes[key]['case_id'],outcome=outcome,episode_key=key,
                                    windows=len(positions),probability=per_window*len(positions),
                                    per_window_probability=per_window))

    def _weights(self, requested, available, level, context):
        keys = sorted(available)
        require(keys and set(keys) <= set(requested), f'Invalid groups at {level}')
        total = sum(requested[k] for k in keys)
        # Ratios are preferences, not hard filters. If all remaining branches have
        # requested weight zero, use available branches uniformly and report it.
        effective = {k:requested[k]/total if total else 1/len(keys) for k in keys}
        if set(keys) != set(requested) or not total:
            self.fallbacks.append(dict(level=level, context=context, requested=requested,
                effective=effective, missing=sorted(set(requested)-set(keys)),
                reason='available_groups_renormalized' if total else 'all_available_weights_zero_uniform_fallback'))
        return effective

    def draw_indices(self, count):
        """Return zero-based Dataset row indices in ORDER BY sample_id order."""
        require(type(count) is int and count >= 0, 'Draw count must be a nonnegative integer')
        indices = np.searchsorted(self._cdf, self.rng.random(count), side='right')
        self.draw_count += count
        return indices

    def draw_sample_ids(self, count):
        """Persistent SQLite IDs; do NOT confuse with zero-based Dataset indices."""
        return self.sample_ids[self.draw_indices(count)]

    def state_dict(self):
        return dict(version=VERSION, fingerprint=self.fingerprint,
                    draw_count=self.draw_count, rng_state=deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        require(state['version'] == VERSION and state['fingerprint'] == self.fingerprint,
                'Sampler checkpoint index/config/code differs')
        require(type(state['draw_count']) is int and state['draw_count'] >= 0, 'Invalid draw count')
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = deepcopy(state['rng_state'])
        self.rng, self.draw_count = rng, state['draw_count']

    def plan(self):
        expected = dict(control=defaultdict(float),variant=defaultdict(float),episode_outcome=defaultdict(float),task=defaultdict(float),collection_round=defaultdict(float))
        for leaf in self.leaves:
            expected['task'][self.episodes[leaf['episode_key']].get('task','unspecified')] += leaf['probability']
            expected['collection_round'][self.episodes[leaf['episode_key']].get('collection_round','unspecified')] += leaf['probability']
            for axis,field in [('control','control'),('variant','variant'),('episode_outcome','outcome')]:
                expected[axis][leaf[field]] += leaf['probability']
        lengths = np.asarray([r['valid_length'] for r in self.rows])
        gpt = np.asarray([r['gpt_valid_steps'] for r in self.rows])
        return dict(version=VERSION,fingerprint=self.fingerprint,index_sha256=self.index_sha256,
                    config=self.config,sampling='with_replacement',row_order='sample_id ascending, zero-based Dataset indices',
                    window_count=len(self.rows),positive_probability_windows=int(np.count_nonzero(self.probabilities)),
                    effective_probabilities={k:dict(v) for k,v in expected.items()},fallbacks=self.fallbacks,
                    case_availability=self.case_availability,episode_strata=self.leaves,
                    expected_valid_steps_per_draw=float(self.probabilities@lengths),
                    expected_gpt_supervision_fraction=float((self.probabilities@gpt)/(self.probabilities@lengths)))

    def report(self, indices):
        indices = np.asarray(indices)
        require(indices.ndim == 1 and np.issubdtype(indices.dtype,np.integer) and
                np.all((indices >= 0) & (indices < len(self.rows))), 'Invalid sampled Dataset indices')
        frequency = Counter(int(i) for i in indices)
        controls,variants,outcomes,episode_counts,case_counts,tasks = (Counter() for _ in range(6))
        valid_steps=gpt_steps=decision_start_draws=0
        for i,count in frequency.items():
            row=self.rows[i]; ep=self.episodes[row['episode_key']]
            controls[row['start_control']]+=count; variants[ep['variant']]+=count
            tasks[ep.get('task','unspecified')]+=count
            outcomes['success' if ep['episode_success'] else 'failure']+=count
            episode_counts[row['episode_key']]+=count;case_counts[ep['case_key']]+=count
            valid_steps+=row['valid_length']*count;gpt_steps+=row['gpt_valid_steps']*count
            decision_start_draws+=int(row['start_control']=='gpt' and row['is_decision_start'])*count
        total=len(indices)
        return dict(draws=total,unique_windows=len(frequency),repeated_draws=total-len(frequency),
                    max_window_draws=max(frequency.values(),default=0),
                    unique_gpt_start_windows=sum(self.rows[i]['start_control']=='gpt' for i in frequency),
                    unique_gpt_decision_starts=sum(self.rows[i]['start_control']=='gpt' and self.rows[i]['is_decision_start'] for i in frequency),
                    gpt_decision_start_draws=decision_start_draws,
                    window_counts=dict(control=dict(controls),variant=dict(variants),episode_outcome=dict(outcomes),task=dict(tasks)),
                    fractions={axis:{k:v/total for k,v in counts.items()} if total else {} for axis,counts in
                               [('control',controls),('variant',variants),('episode_outcome',outcomes),('task',tasks)]},
                    supervised_steps=dict(total=valid_steps,gpt=gpt_steps,pi05=valid_steps-gpt_steps),
                    episode_draws=dict(sorted(episode_counts.items())),case_draws=dict(sorted(case_counts.items())))

    def exposure(self, total_draws):
        """Expected repetitions, separate from a particular random draw realization."""
        require(type(total_draws) is int and total_draws > 0, 'Invalid total draws')
        expected = self.probabilities*total_draws
        episode_mass = defaultdict(float)
        for leaf in self.leaves:
            episode_mass[leaf['episode_key']] += leaf['probability']
        top = []
        for i in np.argsort(expected)[-20:][::-1]:
            row = self.rows[int(i)]; ep = self.episodes[row['episode_key']]
            top.append(dict(sample_id=row['sample_id'],episode_key=row['episode_key'],
                task=ep.get('task'),case_id=ep['case_id'],start_step=row['start_step'],
                start_control=row['start_control'],expected_draws=float(expected[i])))
        return dict(total_draws=total_draws,effective_window_count=float(1/(self.probabilities@self.probabilities)),
            expected_window_draws=dict(zip(('min','median','p95','p99','max'),
                np.quantile(expected,[0,.5,.95,.99,1]).tolist())),
            episode_expected_draws={k:v*total_draws for k,v in sorted(episode_mass.items())},
            highest_exposure_windows=top)
