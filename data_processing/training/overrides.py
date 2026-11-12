"""Typed launch overrides, resolved before any JAX import."""
import json

from data_processing.subtasks.io import require


# All other fields can also be overridden through TRAIN_OVERRIDES_JSON.
ENV_FIELDS = {
    'TRAIN_MODE': 'conditioning.mode',
    'TRAIN_OUTPUT': 'output',
    'TRAIN_INDEX_ROOT': 'index_root',
    'TRAIN_BASE_CHECKPOINT': 'base_checkpoint',
    'TRAIN_NORM_STATS_DIR': 'norm_stats_dir',
    'TRAIN_OPENPI_SOURCE': 'openpi_source',
    'TRAIN_NUM_STEPS': 'training.num_steps',
    'TRAIN_BATCH_SIZE': 'training.batch_size',
    'TRAIN_MICRO_BATCH_SIZE': 'training.micro_batch_size',
    'TRAIN_NUM_WORKERS': 'training.num_workers',
    'TRAIN_CACHE_EPISODES': 'training.cache_episodes',
    'TRAIN_SEED': 'training.seed',
    'TRAIN_SAMPLING_SEED': 'sampling.seed',
    'TRAIN_GPT_START_RATIO': 'sampling.gpt_start_ratio',
    'TRAIN_GPT_START_WEIGHT': 'sampling.gpt_start_weight',
    'TRAIN_PI05_START_WEIGHT': 'sampling.pi05_start_weight',
    'TRAIN_RANDOM_RATIO': 'sampling.random_ratio',
    'TRAIN_SUCCESSFUL_EPISODE_RATIO': 'sampling.successful_episode_ratio',
    'TRAIN_SUCCESSFUL_CASE_WEIGHT': 'sampling.successful_case_weight',
    'TRAIN_UNSUCCESSFUL_CASE_WEIGHT': 'sampling.unsuccessful_case_weight',
    'TRAIN_LEARNING_RATE': 'optimizer.learning_rate',
    'TRAIN_END_LEARNING_RATE': 'optimizer.end_learning_rate',
    'TRAIN_WARMUP_STEPS': 'optimizer.warmup_steps',
    'TRAIN_WEIGHT_DECAY': 'optimizer.weight_decay',
    'TRAIN_CLIP_GRADIENT_NORM': 'optimizer.clip_gradient_norm',
    'TRAIN_B1': 'optimizer.b1', 'TRAIN_B2': 'optimizer.b2', 'TRAIN_EPS': 'optimizer.eps',
    'TRAIN_SAVE_INTERVAL': 'checkpoint.save_interval',
    'TRAIN_MAX_TO_KEEP': 'checkpoint.max_to_keep',
    'TRAIN_NUM_DEVICES': 'parallelism.num_devices',
    'TRAIN_FSDP_DEVICES': 'parallelism.fsdp_devices',
    'TRAIN_JAX_PREALLOCATE': 'runtime.jax_preallocate',
    'TRAIN_REQUIRE_GPU': 'runtime.require_gpu',
}
BATCH_ALIASES = {'TRAIN_PER_DEVICE_BATCH_SIZE', 'TRAIN_GRADIENT_ACCUMULATION_STEPS'}


def apply_overrides(cfg, env, mode=None):
    allowed = set(ENV_FIELDS) | BATCH_ALIASES | {'TRAIN_CONFIG', 'TRAIN_PYTHON', 'TRAIN_OVERRIDES_JSON'}
    require(not {k for k in env if k.startswith('TRAIN_')} - allowed,
            f'Unknown training environment variables: {sorted({k for k in env if k.startswith("TRAIN_")} - allowed)}')
    explicit = set()

    def merge(dst, src, prefix=''):
        require(isinstance(src, dict), 'TRAIN_OVERRIDES_JSON must contain nested configuration objects')
        for key, value in src.items():
            path = f'{prefix}.{key}' if prefix else key
            require(key in dst, f'Unknown override: {path}')
            if isinstance(dst[key], dict):
                merge(dst[key], value, path)
            else:
                dst[key] = value
                explicit.add(path)

    try:
        merge(cfg, json.loads(env.get('TRAIN_OVERRIDES_JSON', '{}')))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid TRAIN_OVERRIDES_JSON: {exc}') from exc
    for name, path in ENV_FIELDS.items():
        if name not in env:
            continue
        parts = path.split('.')
        dst = cfg
        for key in parts[:-1]:
            dst = dst[key]
        require(parts[-1] in dst, f'{name} does not apply to the selected configuration')
        # Strings only occur in paths and mode. Other values use strict JSON
        # scalars, so "false" is false and malformed numbers are never ignored.
        value = env[name]
        if path not in ('output', 'index_root', 'base_checkpoint', 'norm_stats_dir',
                        'openpi_source', 'conditioning.mode'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{name}: expected a JSON number or boolean') from exc
        dst[parts[-1]] = value
        explicit.add(path)
    if mode is not None:
        cfg['conditioning']['mode'] = mode

    if any(name in env for name in BATCH_ALIASES):
        tr = cfg['training']
        devices = cfg['parallelism']['num_devices']
        for name, value in [('num_devices', devices), ('batch_size', tr['batch_size']),
                            ('micro_batch_size', tr['micro_batch_size'])]:
            require(type(value) is int and value > 0, f'Invalid {name}')

        def positive(name, default):
            if name not in env:
                require(default is not None, f'Set {name} explicitly for this device/batch configuration')
                return default
            try:
                value = json.loads(env[name])
            except json.JSONDecodeError as exc:
                raise ValueError(f'{name}: expected positive integer') from exc
            require(type(value) is int and value > 0, f'{name}: expected positive integer')
            return value

        per_device = positive('TRAIN_PER_DEVICE_BATCH_SIZE',
                              tr['micro_batch_size']//devices if tr['micro_batch_size'] % devices == 0 else None)
        ga = positive('TRAIN_GRADIENT_ACCUMULATION_STEPS',
                      tr['batch_size']//tr['micro_batch_size'] if tr['batch_size'] % tr['micro_batch_size'] == 0 else None)
        values = dict(micro_batch_size=devices*per_device, batch_size=devices*per_device*ga)
        for key, value in values.items():
            require(f'training.{key}' not in explicit or tr[key] == value,
                    f'Explicit training.{key} conflicts with per-device batch / gradient accumulation')
            tr[key] = value
    return cfg
