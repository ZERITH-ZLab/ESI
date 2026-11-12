"""One strict configuration for data, sampling, optimization and checkpoints."""
from pathlib import Path
import math
import os
from data_processing.sampling.sampler import checked_config
from data_processing.subtasks.io import read, require


def load_config(path, *, mode=None, environ=None):
    cfg = read(path)
    expected = read(Path(__file__).with_name('config.example.json'))
    # Old single-device configs remain readable; resume still checks code/config fingerprints.
    cfg.setdefault('parallelism', {'num_devices': 1, 'fsdp_devices': 1})
    cfg.setdefault('conditioning', {'mode': 'task'})
    from .overrides import apply_overrides
    cfg = apply_overrides(cfg, os.environ if environ is None else environ, mode)
    def keys(value, template, name):
        require(isinstance(value,dict) and set(value)==set(template), f'{name}: missing/unknown config keys')
        for k, v in template.items():
            if k == 'sampling' and name == 'config':
                checked_config(value[k])
            elif isinstance(v,dict): keys(value[k],v,f'{name}.{k}')
    keys(cfg,expected,'config')
    require(cfg['conditioning']['mode'] in ('task', 'subtask'), 'conditioning.mode must be task or subtask')
    require(isinstance(cfg['output'], str), 'output: path required')
    cfg['output'] = cfg['output'].replace('{mode}', cfg['conditioning']['mode'])
    for k in ('index_root','base_checkpoint','norm_stats_dir','openpi_source','output'):
        require(isinstance(cfg[k],str) and cfg[k].strip(), f'{k}: path required')
        cfg[k]=str(Path(cfg[k]).expanduser().resolve())
    checked_config(cfg['sampling'])
    tr,opt,ck = cfg['training'],cfg['optimizer'],cfg['checkpoint']
    for k,v in tr.items():
        require(type(v) is int and v >= (0 if k in ('seed','num_workers') else 1),f'Invalid training.{k}')
    require(tr['batch_size'] % tr['micro_batch_size']==0,'batch_size must be divisible by micro_batch_size')
    par=cfg['parallelism']
    require(all(type(v) is int and v>0 for v in par.values()),'Invalid parallelism settings')
    require(par['num_devices'] % par['fsdp_devices']==0,'num_devices must be divisible by fsdp_devices')
    require(tr['micro_batch_size'] % par['num_devices']==0,
            'Global micro_batch_size must be divisible by num_devices (at least one window per device)')
    require(type(opt['warmup_steps']) is int and 0 <= opt['warmup_steps'] < tr['num_steps'],'Invalid warmup_steps')
    for k,v in opt.items():
        require(type(v) in (int,float) and math.isfinite(v) and v>=0,f'Invalid optimizer.{k}')
    require(opt['learning_rate']>0 and opt['eps']>0 and opt['clip_gradient_norm']>0 and
            opt['end_learning_rate']<=opt['learning_rate'] and 0<=opt['b1']<1 and 0<=opt['b2']<1,
            'Invalid optimizer parameters')
    require(all(type(v) is int and v>0 for v in ck.values()),'Invalid checkpoint settings')
    require(all(type(v) is bool for v in cfg['runtime'].values()),'Runtime flags must be bool')
    out=Path(cfg['output'])
    for source in [Path(cfg[k]) for k in ('index_root','base_checkpoint','norm_stats_dir','openpi_source')]:
        require(not out.is_relative_to(source) and not source.is_relative_to(out),'Output overlaps input data/code')
    require(not out.is_relative_to(Path(__file__).resolve().parents[2]),'Training output must be outside repository')
    return cfg
