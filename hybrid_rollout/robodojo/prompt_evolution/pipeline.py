"""Persistent lead analysis and independent review; publish only a next-round bundle."""
import json
import os
from pathlib import Path
from ..io import write_json
from .bundle import seed,load,digest
from .validation import support_check
from .config import settings
from .data import inventory,read

HERE=Path(__file__).parent



def resolved_env(cfg,env):
    # Account connection settings come exclusively from annotation_config.
    return dict(env)


def run(suite, env=None, *, overrides=None, prepare_only=False, backend=None, output=None,
        historical_source=None, experiment_only=False):
    suite=Path(suite).resolve();cfg=read(suite/'config.json')
    if cfg.get('schema')!='robodojo.collection_suite.v1':raise ValueError('Only training collection suites are allowed')
    opts=settings(dict(cfg.get('prompt_evolution',{}),**(overrides or {})))
    if historical_source is not None and not experiment_only:
        raise ValueError('Historical prompt rollback is allowed only in isolated experiments')
    if experiment_only:
        if output is None or Path(output).resolve().is_relative_to(suite):
            raise ValueError('Experiment output must be separate from the collection suite')
        opts['publish']='candidate_only'
    original=seed(historical_source) if historical_source is not None else None
    base=original if original is not None else load(cfg['prompt_bundle']) if cfg.get('prompt_bundle') else seed()
    env=resolved_env(cfg,os.environ if env is None else env)
    manifest=inventory(suite,cfg)
    identity=dict(config=opts,base_sha256=base['sha256'],inventory=manifest,
                  experiment_only=experiment_only,historical_source=str(historical_source) if historical_source else None,
                  code_sha256=digest({p.name:p.read_text() for p in HERE.iterdir() if p.suffix in ('.py','.md')}))
    key=digest(identity)[:16];root=Path(output) if output else suite/'prompt_evolution'/('run_'+key)
    from .conversation import run_conversation
    return run_conversation(suite,cfg,opts,manifest,base,identity,root,
                            prepare_only=prepare_only,backend=backend)


def after_collection(cfg,env,suite,code):
    """Best effort hook. Never changes collection exit status or live prompt files."""
    value=env.get('ROLLOUT_PROMPT_EVOLUTION')
    opts=settings(cfg.get('prompt_evolution'))
    if value is not None:
        if value not in ('0','1'):raise ValueError('ROLLOUT_PROMPT_EVOLUTION must be 0 or 1')
        opts['enabled']=value=='1'
    if not opts['enabled']:return None
    status=read(Path(suite)/'status.json')
    if code not in (0,124) or status.get('state') not in ('completed','completed_with_timeouts'):
        result=dict(state='deferred',reason='Collection not fully completed',collection_state=status.get('state'))
    else:
        try:result=run(suite,env,overrides=opts)
        except Exception as exc:result=dict(state='failed',error=f'{type(exc).__name__}: {exc}')
    write_json(Path(suite)/'prompt_evolution_status.json',result)
    print(json.dumps(dict(event='prompt_evolution_finished',**result)),flush=True)
    return result
