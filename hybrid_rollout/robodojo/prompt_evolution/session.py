"""Persistent, leased Codex conversations. Independent reviewer has its own directory/thread."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import jsonschema

from ..io import write_json
from .bundle import digest
from .client import annotation_accounts, annotation_lease


def utc():
    return datetime.now(timezone.utc).isoformat()


def events(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_text(errors='replace').splitlines():
        try: result.append(json.loads(line))
        except ValueError: pass
    return result


class Conversation:
    def __init__(self, root, cfg, *, backend=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.cfg=cfg;self.backend=backend
        self.path=self.root/'session.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(
            thread_id=None,profile=None,generation=0,started_at=utc(),needs_anchor=False)

    def save(self):
        write_json(self.path,self.state)

    def command(self, cfg, workspace, schema_path, final_path):
        command=[cfg['codex_bin'],'exec','--ignore-user-config','--ignore-rules',
                 '--skip-git-repo-check','--sandbox','read-only','--cd',str(self.root)]
        overrides=dict(model_provider='openai',forced_login_method='chatgpt',
            cli_auth_credentials_store='file',approval_policy='never',web_search='disabled',
            model_reasoning_effort=self.cfg['reasoning_effort'],
            model_auto_compact_token_limit=self.cfg['auto_compact_token_limit'],
            model_context_window=self.cfg['model_context_window'],
            sqlite_home=str(self.root/'runtime_db'),log_dir=str(workspace/'runtime_logs'))
        for key in ('shell_tool','unified_exec','apply_patch_freeform','view_image','multi_agent',
                    'apps','skill_search','hooks','image_generation'):
            overrides['features.'+key]=False
        overrides['features.skip_host_skill_discovery']=True
        for k,v in overrides.items():command+=['-c',f'{k}={json.dumps(v)}']
        if self.state['thread_id']:command+=['resume',self.state['thread_id']]
        command+=['--model',self.cfg['model'],'--output-schema',str(schema_path),
                  '--output-last-message',str(final_path),'--json']
        return command

    def recover(self, folder):
        ev=events(folder/'events.jsonl')
        for e in ev:
            if e.get('type')=='thread.started':self.state['thread_id']=e['thread_id']
            kind=str(e.get('type',''))+' '+str((e.get('item') or {}).get('type',''))
            if 'compact' in kind.lower():self.state['needs_anchor']=True
        if ev:self.save()
        return ev

    def ask(self, turn, prompt, payload, schema, checkpoint, *, images=()):
        folder=self.root/'turns'/f'{turn:05d}';folder.mkdir(parents=True,exist_ok=True)
        request=dict(prompt=prompt,payload=payload,schema=schema,images=[str(p) for p in images],
                     checkpoint_sha256=digest(checkpoint))
        path=folder/'request.json'
        if path.exists() and json.loads(path.read_text())!=request:
            raise ValueError('Turn input differs from persisted checkpoint')
        write_json(path,request)
        # Save evidence inventory BEFORE any possible native context compaction/network interruption.
        write_json(folder/'checkpoint_before.json',checkpoint)
        accepted=folder/'accepted.json'
        if accepted.exists():return json.loads(accepted.read_text())
        attempts=sorted(folder.glob('attempt_*'))
        for previous in attempts:
            if (previous/'reconstruction.json').exists():
                self.state.update(thread_id=None,needs_anchor=True)
                self.save()
                continue
            ev=self.recover(previous);final=previous/'last_message.json'
            if final.exists() and any(e.get('type')=='turn.completed' for e in ev):
                try:
                    value=json.loads(final.read_text());jsonschema.validate(value,schema)
                except (ValueError,jsonschema.ValidationError):continue
                write_json(accepted,value);return value
        for attempt in range(self.cfg['max_attempts']):
            work=folder/f'attempt_{len(attempts)+attempt:03d}';work.mkdir()
            started=time.monotonic();meta=dict(started_at=utc(),state='running',thread_before=self.state['thread_id'])
            write_json(work/'timing.json',meta)
            try:
                if self.backend:
                    value=self.backend(self,prompt,payload,schema,checkpoint,images)
                    if not self.state['thread_id']:self.state['thread_id']='test-thread-'+self.root.name
                    self.save()
                else:value=self.execute(work,prompt,payload,schema,checkpoint,images)
                write_json(work/'response.json',value)
                jsonschema.validate(value,schema)
                write_json(accepted,value);meta['state']='completed'
                return value
            except (ValueError,OSError,TimeoutError,subprocess.TimeoutExpired,jsonschema.ValidationError) as exc:
                meta.update(state='failed',error=str(exc)[:1600])
                # A lost server thread is reconstructible, but quota/auth/transport errors must not
                # silently replace the ongoing conversation or switch accounts.
                text=str(exc).lower()
                if any(s in text for s in ('thread not found','session not found','no saved session found')):
                    meta['reconstruct']=True
                    write_json(work/'reconstruction.json',dict(reason=str(exc),checkpoint=str(folder/'checkpoint_before.json')))
                if attempt+1==self.cfg['max_attempts']:raise
            except BaseException as exc:
                meta.update(state='interrupted',error=type(exc).__name__)
                raise
            finally:
                ev=self.recover(work)
                if meta.get('reconstruct'):
                    self.state.update(thread_id=None,generation=self.state['generation']+1,needs_anchor=True)
                    self.save()
                usages=[e['usage'] for e in ev if e.get('type')=='turn.completed' and e.get('usage')]
                meta.update(ended_at=utc(),duration_seconds=time.monotonic()-started,
                            usage=usages or None,thread_after=self.state['thread_id'])
                write_json(work/'timing.json',meta)

    def execute(self, work, prompt, payload, schema, checkpoint, images):
        from ..codex_backend.profiles import account_home,clean_login_env
        from data_processing.subtasks.client import codex_failure_detail
        accounts=annotation_accounts(self.cfg['annotation_config'])
        if self.state['profile']:
            if self.state['profile'] not in accounts['profiles']:
                raise ValueError('Pinned conversation profile removed from annotation pool; restore it to resume')
            accounts=dict(accounts,profiles=[self.state['profile']])
        with annotation_lease(accounts,self.cfg['account_wait_seconds']) as (name,fd):
            self.state['profile']=name;self.save()
            home=account_home(name,accounts['shared_root'])
            # Verify identity without persisting credentials. Session ownership must not change on relogin.
            auth=json.loads((home/'auth.json').read_text());account=digest(auth['tokens']['account_id'])
            if self.state.get('account_sha256',account)!=account:
                raise ValueError('Pinned profile now belongs to a different account')
            self.state['account_sha256']=account;self.save()
            schema_path=work/'schema.json';final=work/'last_message.json';write_json(schema_path,schema)
            first=not self.state['thread_id'];anchor=first or self.state['needs_anchor']
            content=(prompt+'\n\n' if anchor else '')+json.dumps(dict(
                incremental_evidence=payload,**({'durable_checkpoint':checkpoint} if anchor else {})),ensure_ascii=False)
            if len(content)>self.cfg['max_input_chars']:
                raise ValueError('Increment/checkpoint exceeds input budget; reduce evidence size, never truncate')
            write_json(work/'input.json',dict(text=content,images=[str(p) for p in images]))
            command=self.command(accounts,work,schema_path,final)
            for p in images:command+=['--image',str(Path(p).resolve())]
            command+=['-']
            with (work/'events.jsonl').open('w') as out,(work/'stderr.log').open('w') as err:
                proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,text=True,
                    env=clean_login_env(home,accounts['proxy_file']),start_new_session=True,pass_fds=(fd,))
                try:proc.communicate(content,timeout=self.cfg['timeout_seconds'])
                except BaseException:
                    if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait();raise
            self.state['needs_anchor']=False;ev=self.recover(work)
            if proc.returncode or not final.exists():raise ValueError(codex_failure_detail(work))
            if not self.state['thread_id']:raise ValueError('Codex did not return a persistent thread ID')
            self.save()
            return json.loads(final.read_text())


def usage_summary(root):
    timings=[];totals={};missing=0
    for path in Path(root).glob('*/turns/*/attempt_*/timing.json'):
        t=json.loads(path.read_text());timings.append(t)
        if not t.get('usage'):missing+=1
        for u in t.get('usage') or []:
            for k,v in u.items():
                if isinstance(v,(int,float)):totals[k]=totals.get(k,0)+v
    return dict(attempts=len(timings),attempts_without_usage=missing,usage=totals,
        cumulative_call_seconds=sum(t.get('duration_seconds',0) for t in timings),
        note='Cached input is a subset of input; reasoning output is not added again. Failed calls without usage are unknown.')
