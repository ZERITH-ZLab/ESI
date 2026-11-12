"""Resolve one stage, inspect it, then delegate to an existing stage implementation."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .config import ROOT, load, recursive_round_weights


def annotation_settings(cfg, online=False):
    accounts = cfg['deployment']['annotation']
    return dict(cfg['llm']['annotation'], backend='codex', profiles=accounts['profiles'],
        workers=accounts['workers'], shared_root=cfg['deployment']['paths']['auth_root'],
        codex_bin=cfg['deployment']['paths']['codex_bin'], proxy_file=accounts.get('proxy_file'),
        base_url='https://api.openai.com/v1', api_mode='responses', api_key_env='SUBTASK_API_KEY',
        timeout_seconds=1800 if online else 600, max_output_tokens=48000 if online else 24000,
        max_input_chars=900000 if online else 700000, http_retries=2, max_repairs=3)


def environment(cfg):
    dep = cfg['deployment']; paths = dep['paths']
    env = dict(os.environ)
    # Do not accidentally inherit research rollout flags, case filters or prompt overrides.
    for key in list(env):
        if key.startswith(('ROLLOUT_', 'ROBODOJO_', 'POLICY_', 'SIM_', 'CODEX_IMAGE_', 'CODEX_MAX_')) or key in ('CHECKPOINT','MAX_DECISIONS'):
            env.pop(key)
    env.update(CODE_ROOT=str(ROOT), PYTHONPATH=str(ROOT), ROBODOJO_SOURCE=paths['robodojo_source'],
        ROBODOJO_PYTHON=paths['robodojo_python'], OPENPI_SOURCE=paths['openpi_source'],
        OPENPI_PYTHON=paths['openpi_python'], CODEX_BIN=paths['codex_bin'],
        ROLLOUT_SHARED_ROOT=paths['output_root'], ROLLOUT_AUTH_ROOT=paths['auth_root'],
        ROLLOUT_ACCOUNT_POOL_FILE=paths['account_pool_file'],
        ROLLOUT_SIM_SSH_USER=paths.get('sim_ssh_user', 'root'))
    for key,value in cfg.get('environment', {}).items():
        env[key] = json.dumps(value) if isinstance(value,(dict,list,bool)) else str(value)
    if cfg['stage'] in ('collection','eval'):
        slots = dep['collection']['slots']
        if not slots: raise ValueError('Configure at least one deployment.collection.slots entry')
        env.update({k:str(v) for k,v in slots[0].items() if k != 'ROLLOUT_SIM_SSH_USER'})
        env['ROLLOUT_WORKER_SLOTS_JSON'] = json.dumps(slots)
        env['ROLLOUT_PARALLEL_WORKERS'] = str(dep['collection']['workers'])
        env['CHECKPOINT'] = cfg['run']['checkpoint']
        env['ROLLOUT_AUTH_PROFILE'] = dep['collection']['auth_profile']
        from hybrid_rollout.robodojo.parallel_collection import explicit_slots
        explicit_slots(env)  # Only checks explicit capacity/ports, no GPUs, SSH or accounts.
    return env


def plan(cfg, directory, *, inspect=False, training_mode=None):
    stage = cfg['stage']; run = cfg['run']; dep = cfg['deployment']; paths = dep['paths']
    if not isinstance(run.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+',run['name']):
        raise ValueError('Set run.name to a new explicit experiment name')
    def required(key):
        value = run.get(key)
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f'Set run.{key} to a nonempty value')
        return value
    if stage in ('collection','eval','training'): required('checkpoint')
    env = environment(cfg); native = deepcopy(cfg.get('config', {})); files = {}
    python = paths['robodojo_python']; native_path = str(directory/'stage_config.json')
    cmd = []
    if stage in ('collection','eval'):
        native.update(name=run['name'],workers=dep['collection']['workers'],account_pool_file=paths['account_pool_file'])
        teacher = cfg['llm']['rollout']
        env.update(ROLLOUT_MODEL=teacher['model'],ROLLOUT_REASONING_EFFORT=teacher['effort'],
            CODEX_IMAGE_MAX_EDGE=str(teacher['image_max_edge']),
            ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT=str(teacher['auto_compact_token_limit']),
            ROLLOUT_MAX_SECONDS=str(cfg['limits']['episode_seconds']))
        if stage=='collection':
            native.update(tasks=cfg['tasks'],checkpoint=run['checkpoint'],model=teacher['model'],reasoning_effort=teacher['effort'],
                max_seconds=cfg['limits']['episode_seconds'], auto_compact_token_limit=teacher['auto_compact_token_limit'],
                image_max_edge=teacher['image_max_edge'])
            hold = native.get('pi_force_hold',{})
            # RoboDojo/X5 execution adaptation, not a general ESI requirement.
            # A disabled master switch must not demand tightening credentials.
            if hold and not hold.get('enabled'):
                hold['config'] = {}
            if hold.get('enabled') and hold['config'].get('tighten_enabled'):
                hold['config'].update(flywheel_round=run['round'],
                    controller_route=native['controller_route'])
                # Bind to this run's actual checkpoint, never the author's weight hash.
                if inspect: hold['config']['checkpoint_sha256']='<computed on start>'
                else:
                    from hybrid_rollout.robodojo.pi05_server.checkpoint import checkpoint_identity
                    hold['config']['checkpoint_sha256']=checkpoint_identity(run['checkpoint'])['checkpoint_sha256']
            if 'prompt_evolution' in native:
                files['annotation_accounts.json'] = annotation_settings(cfg)
                native['prompt_evolution']['annotation_config'] = str(directory/'annotation_accounts.json')
            cmd=['bash',str(ROOT/'hybrid_rollout/robodojo/launch_collection_suite.sh'),'--config',native_path]
        else:
            if cfg.get('robodojo', {}).get('pi_force_hold', {}).get('enabled'):
                raise ValueError('Release eval disables automatic pi_force_hold; this switch is for collection')
            native['teacher']=teacher
            native['models']=[dict(cfg['model'],task=task,experiment=run['name']+'__'+task,
                checkpoint=run['checkpoint'],cases=cfg['cases'][task]) for task in cfg['tasks']]
            if 'expected_runtime' in native or 'execution_source' in native:
                raise ValueError('Release eval records actual runtime; historical source/runtime pins belong in reference files')
            cmd=['bash',str(ROOT/'hybrid_rollout/robodojo/launch_dual.sh'),'--eval-retry','--config',native_path,
                 '--max-restarts',str(cfg['limits']['queue_restarts'])]
    elif stage=='annotation':
        required('batch')
        route=cfg['route']; online=route=='online'
        if route not in ('offline','online'):raise ValueError('Unknown annotation route')
        native = dict(annotation_settings(cfg,online),**native)
        module='data_processing.online_subtask_outcomes' if online else 'data_processing.subtasks'
        cmd=[python,'-u','-m',module,run['action']]
        if run['action']=='prepare':
            if not run['inputs']:raise ValueError('Set run.inputs to collection directories')
            cmd += [*run['inputs'],'--output',run['batch']]
        elif run['action']=='run':cmd += [run['batch'],'--config',native_path]
        elif run['action'] in ('validate','catalog'):cmd += [run['batch']]
        else:raise ValueError('Unsupported annotation action')
    elif stage=='training':
        for key in ('index','norm','output'): required(key)
        if run['action'] not in ('run','check-data','show-config'): raise ValueError('Unsupported training action')
        if run['mode'] not in ('task','subtask'): raise ValueError('Unknown conditioning mode')
        recursive_round_weights(run['rounds'])  # Validate even the single-round case.
        native['conditioning']['mode']=run['mode']
        native.update(index_root=run['index'],base_checkpoint=run['checkpoint'],norm_stats_dir=run['norm'],
                      output=run['output'],openpi_source=paths['openpi_source'])
        rounds=run['rounds']
        native['sampling'].pop('round_weights',None)
        if len(rounds)>1:
            native['sampling'].update(strategy='task_round_case_control_weighted',round_weights=recursive_round_weights(rounds))
        elif len(rounds)==1:native['sampling']['strategy']='task_case_control_weighted'
        else:raise ValueError('Set run.rounds oldest to newest')
        hw=dep['training'];devices=hw['num_devices'];per=hw['per_device_batch_size'];ga=hw['gradient_accumulation_steps']
        if any(type(v) is not int or v<1 for v in (devices,per,ga)):raise ValueError('Positive device/batch/GA values required')
        if type(hw['num_nodes']) is not int or hw['num_nodes'] < 1 or devices % hw['num_nodes']:
            raise ValueError('num_devices must divide evenly across the explicitly configured nodes')
        native['parallelism']=dict(num_devices=devices,fsdp_devices=hw['fsdp_devices'])
        native['training'].update(batch_size=devices*per*ga,micro_batch_size=devices*per,num_workers=hw['num_workers'])
        from data_processing.training.overrides import apply_overrides
        native = apply_overrides(native,env,mode=training_mode)
        if native['parallelism']['num_devices'] % hw['num_nodes']:
            raise ValueError('Overridden device count does not divide across nodes')
        env.setdefault('XLA_CLIENT_MEM_FRACTION',str(hw['memory_fraction']))
        env.pop('XLA_PYTHON_CLIENT_MEM_FRACTION',None)
        env.update(JAX_PLATFORMS='cuda',OPENBLAS_NUM_THREADS='1')
        declared_count=env.get('JAX_PROCESS_COUNT')
        if declared_count is not None and int(declared_count)!=hw['num_nodes']:
            raise ValueError('JAX_PROCESS_COUNT differs from deployment.training.num_nodes')
        env['JAX_PROCESS_COUNT']=str(hw['num_nodes'])
        if hw['num_nodes']==1: env['JAX_PROCESS_INDEX']='0'
        if hw['num_nodes']>1:
            if not inspect:
                rank=env.get('JAX_PROCESS_INDEX',env.get('RANK'))
                if rank is None:raise ValueError('Multinode training requires node RANK or JAX_PROCESS_INDEX (one process per node)')
                if not 0 <= int(rank) < hw['num_nodes']: raise ValueError('Invalid node rank')
                env['JAX_PROCESS_INDEX']=rank
            # Same behavior as the existing multinode launcher.
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','NO_PROXY','no_proxy'):env.pop(key,None)
        cmd=[paths['openpi_python'],'-u','-m','data_processing.training',run['action'],'--config',native_path]
    elif stage=='data':
        action=run['action']
        required('output')
        for key in {'export':('annotation_run',),'index':('annotation_run','policy_version','collection_round'),
                    'merge':(),'normalize':('index','checkpoint')}.get(action,()): required(key)
        list_key={'index':'exports','merge':'indexes'}.get(action)
        if list_key and (not isinstance(run[list_key],list) or not run[list_key] or
                         not all(isinstance(v,str) and v for v in run[list_key])):
            raise ValueError(f'Set run.{list_key} to nonempty paths')
        if action=='export':cmd=[python,'-u','-m','data_processing.exports','--annotation-run',run['annotation_run'],'--output',run['output'],'--workers',str(cfg['workers'])]
        elif action=='index':cmd=[python,'-u','-m','data_processing.training_index','build','--annotation-run',run['annotation_run'],'--exports',*run['exports'],'--output',run['output'],'--policy-version',run['policy_version'],'--collection-round',run['collection_round'],'--horizon',str(cfg['horizon']),'--min-valid',str(cfg['min_valid'])]
        elif action=='merge':cmd=[python,'-u','-m','data_processing.training_index','merge','--indexes',*run['indexes'],'--output',run['output']]
        elif action=='normalize':cmd=[paths['openpi_python'],'-u','-m','data_processing.normalization','--index',run['index'],'--base-checkpoint',run['checkpoint'],'--output',run['output']];env['JAX_PLATFORMS']='cpu'
        else:raise ValueError('Unsupported data action')
    elif stage=='evolution':
        required('suite'); required('output')
        files['annotation_accounts.json']=annotation_settings(cfg)
        native['annotation_config']=str(directory/'annotation_accounts.json')
        cmd=[python,'-u','-m','hybrid_rollout.robodojo.prompt_evolution','--suite-root',run['suite'],'--settings',native_path,'--output',run['output']]
    else:raise ValueError(f'Unknown stage: {stage}')
    files['stage_config.json']=native
    return cmd,env,files


def freeze(directory, cfg, files):
    directory.mkdir(parents=True,exist_ok=True)
    # Short local lock allows every training node to read the same manifest.
    with (directory/'config.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        payloads={'resolved_config.json':cfg,**files}
        for name,value in payloads.items():
            path=directory/name
            if path.exists() and json.loads(path.read_text())!=value:
                raise ValueError(f'Configuration changed: {path}. Use a new experiment name; do not overwrite an existing run.')
        for name,value in payloads.items():
            path=directory/name
            if not path.exists():
                with tempfile.NamedTemporaryFile('w',dir=directory,delete=False) as tmp:
                    json.dump(value,tmp,ensure_ascii=False,indent=2);tmp.write('\n');temp=Path(tmp.name)
                temp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--deployment',default=str(ROOT/'configs/deployment.local.json'))
    p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    p.add_argument('--show-config',action='store_true',help='Resolve and print only; no files, models, SSH or account calls')
    p.add_argument('args',nargs=argparse.REMAINDER,help='Arguments after -- are passed to the existing stage CLI')
    a=p.parse_args(argv);cfg=load(a.config,a.deployment,a.set)
    if not isinstance(cfg.get('run',{}).get('name'),str) or not re.fullmatch(r'[A-Za-z0-9_-]+',cfg['run']['name']):
        raise ValueError('Set run.name to a new explicit experiment name')
    directory=Path(cfg['deployment']['paths']['output_root'])/'release_runs'/cfg['run']['name']/cfg['stage']
    if cfg['stage'] in ('annotation','data','training'):
        directory=directory/cfg['run']['action']
    if directory.resolve().is_relative_to(ROOT):raise ValueError('Outputs must be outside the source repository')
    extra=a.args[1:] if a.args[:1]==['--'] else a.args
    training_mode=None
    if cfg['stage']=='training':
        for i,value in enumerate(extra):
            if value=='--mode' and i+1<len(extra): training_mode=extra[i+1]
            elif value.startswith('--mode='): training_mode=value.split('=',1)[1]
        if training_mode is not None: cfg['run']['mode']=training_mode
    cmd,env,files=plan(cfg,directory,inspect=a.show_config,training_mode=training_mode)
    if a.show_config:
        print(json.dumps(dict(config=cfg,generated_configs=files,command=cmd+extra,manifest_dir=str(directory)),ensure_ascii=False,indent=2));return 0
    freeze(directory,cfg,files)
    print(json.dumps(dict(event='release_stage_start',stage=cfg['stage'],manifest_dir=str(directory),command=cmd+extra),ensure_ascii=False),flush=True)
    return subprocess.call(cmd+extra,env=env,cwd=ROOT)


if __name__=='__main__':raise SystemExit(main())
