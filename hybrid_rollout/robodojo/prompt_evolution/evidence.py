"""Read-only evidence index and deterministic coverage/problem accounting."""
from collections import Counter,defaultdict
from pathlib import Path
import re

from ..io import write_json
from .data import ledger,read,load_images
from .bundle import digest


class Evidence:
    def __init__(self,root,rows,cfg):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.rows={r['key']:r for r in rows};self.cfg=cfg;self.cache={}

    def packet(self,key):
        if key not in self.rows:raise ValueError('Unknown case_key')
        if key not in self.cache:
            path=self.root/(key+'.json')
            if path.exists():
                saved=read(path)
                if saved['source_sha256']!=self.rows[key]['history_sha256']:
                    raise ValueError('Evidence cache history identity changed')
                value=saved['packet']
                if digest(value)!=saved['packet_sha256']:raise ValueError('Evidence cache modified')
            else:
                value=ledger(self.rows[key]);write_json(path,dict(source_sha256=self.rows[key]['history_sha256'],
                    packet=value,packet_sha256=digest(value)))
            self.cache[key]=value
        return self.cache[key]

    def index(self):
        result=[]
        for key,row in self.rows.items():
            p=self.packet(key);records=p['records'];streak=longest=0;teacher=steps=0
            for r in records:
                n=r['executed_steps'] or 0;steps+=n
                if r['controller'] in ('eef','edit'):teacher+=n;streak+=n;longest=max(longest,streak)
                else:streak=0
            result.append(dict(case_key=key,task=row['task'],case_id=row['case_id'],split=row['split'],
                result=p['result'],failure_kind=p['failure_kind'],decisions=len(records),
                first_decision=records[0]['decision'] if records else None,
                last_decision=records[-1]['decision'] if records else None,
                executed_steps=steps,teacher_steps=teacher,longest_teacher_streak=longest,
                instruction=(p['instruction'] or '')[:400],
                last_assessment_excerpt=str((records[-1].get('assessment') or {}).get('execution_evidence',''))[:240] if records else '',
                excerpt_is_not_local_outcome=True))
        return result

    def fetch(self,reads,images,knowledge,destination,*,text_budget=None):
        if len(reads)>self.cfg['conversation_max_reads_per_turn']:
            raise ValueError('Too many text ranges in a turn')
        if len(images)>self.cfg['max_images_per_call']:raise ValueError('Too many image ranges')
        output=[];attachments=[];denied=[];used=0
        for req in reads:
            key=req['case_key'];packet=self.packet(key);start=req['start_decision'];end=req['end_decision']
            ids={r['decision'] for r in packet['records']}
            if not req['question'].strip() or start not in ids or end not in ids or start>end:
                raise ValueError('Invalid evidence range/question')
            records=[r for r in packet['records'] if start<=r['decision']<=end]
            if len(records)>self.cfg['conversation_max_read_decisions']:raise ValueError('Text range too long')
            item=dict(kind='records',case_key=key,question=req['question'],instruction=packet['instruction'],records=records)
            if not knowledge['read'].get(key):
                archive=Path(self.rows[key]['archive'])/'controller/codex_workspace'
                item['guidance_at_collection']={name:(archive/name).read_text() for name in
                    ('manipulation_guidance.md','task_success_conditions.md') if (archive/name).is_file()}
                item['guidance_note']='Archived instructions may differ from the baseline being revised; do not confuse the two.'
            import json
            size=len(json.dumps(item,ensure_ascii=False))
            if used+size>(text_budget if text_budget is not None else self.cfg['max_input_chars']//2):
                denied.append(dict(request=req,reason='Incremental text budget reached; request smaller ranges'));continue
            used+=size;output.append(item)
            known=set(knowledge['read'].get(key,[]));known.update(r['decision'] for r in records)
            knowledge['read'][key]=sorted(known)
        for req in images:
            key=req['case_key'];packet=self.packet(key);row=self.rows[key]
            if len(attachments)>=self.cfg['max_images_per_call']:
                denied.append(dict(request=req,reason='Per-turn image budget reached'));continue
            paths,manifest,rejections=load_images(row,[{k:v for k,v in req.items() if k!='case_key'}],self.cfg,
                Path(destination)/key,set(),self.cfg['max_images_per_call']-len(attachments),{r['decision'] for r in packet['records']})
            denied.extend(rejections);known=set(knowledge['images'].get(key,[]))
            for path,meta in zip(paths,manifest):
                if meta['image_id'] not in known and len(known)>=self.cfg['max_images_per_episode']:
                    denied.append(dict(case_key=key,image_id=meta['image_id'],reason='Unique image budget reached'));continue
                known.add(meta['image_id']);attachments.append(path)
                output.append(dict(kind='image',case_key=key,**meta))
            knowledge['images'][key]=sorted(known)
        return dict(evidence=output,denied=denied),attachments

    def update(self,updates,state):
        from .validation import validate_report
        if len(updates)>self.cfg['conversation_max_updates_per_turn']:raise ValueError('Too many issue updates')
        for update in updates:
            key=update['case_key'];packet=self.packet(key);row=self.rows[key]
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',update['id']):raise ValueError('Invalid issue ID')
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',update['group_id']):raise ValueError('Invalid group ID')
            known=set(state['knowledge']['read'].get(key,[]))
            start,end=update['start_decision'],update['end_decision']
            span=[r for r in packet['records'] if start<=r['decision']<=end]
            if not span or not {r['decision'] for r in span}<=known:
                raise ValueError('Read the complete recovery span, including its outcome, before recording it')
            issue={k:v for k,v in update.items() if k not in ('case_key','group_id')}
            validate_report(dict(issues=[issue]),[r for r in packet['records'] if r['decision'] in known],
                self.cfg,set(state['knowledge']['images'].get(key,[])))
            if row.get('failure_kind') or row['receipt_state']!='completed':
                if issue['verified']:raise ValueError('Interrupted/infrastructure case cannot supply verified strategy evidence')
            ref=key+':'+issue['id']
            state['issues'][ref]=dict(issue,ref=ref,group_id=update['group_id'],episode_key=key,
                task=row['task'],case_id=row['case_id'],source_archive=row['archive'],
                history_sha256=row['history_sha256'],actual_steps=sum(r['executed_steps'] or 0 for r in span),
                teacher_steps=sum(r['executed_steps'] or 0 for r in span if r['controller'] in ('eef','edit')))

    def coverage(self,knowledge):
        by=defaultdict(lambda:dict(total_cases=0,read_cases=0,fully_read_cases=0,read_decisions=0,total_decisions=0))
        for key,row in self.rows.items():
            records=self.packet(key)['records'];ids={r['decision'] for r in records};seen=set(knowledge['read'].get(key,[]))
            d=by[row['task']];d['total_cases']+=1;d['read_cases']+=bool(seen);d['fully_read_cases']+=bool(ids) and ids<=seen
            d['read_decisions']+=len(ids&seen);d['total_decisions']+=len(ids)
        return dict(by_task=dict(by),note='Read coverage only; uninspected cases are not negatives. No population incidence estimate.')

    def problem_table(self,issues):
        groups=defaultdict(list)
        for x in issues.values():groups[x['group_id']].append(x)
        result=[]
        for group,items in groups.items():
            cases={(i['task'],i['case_id']) for i in items};outcomes={}
            for field in ('recovery','category'):
                outcomes[field]={v:len({(i['task'],i['case_id']) for i in items if i[field]==v}) for v in sorted({i[field] for i in items})}
            # Union overlapping occurrences per case: duplicated windows cannot inflate step burden.
            observed=defaultdict(set)
            for i in items:
                observed[i['episode_key']].update(range(i['start_decision'],i['end_decision']+1))
            steps=teacher=0
            for key,ids in observed.items():
                for r in self.packet(key)['records']:
                    if r['decision'] in ids:
                        n=r['executed_steps'] or 0;steps+=n;teacher+=n if r['controller'] in ('eef','edit') else 0
            result.append(dict(group_id=group,distinct_cases=len(cases),tasks=sorted({t for t,c in cases}),
                observed_case_outcomes=outcomes,executed_steps_union=steps,teacher_steps_union=teacher,
                refs=[i['ref'] for i in items],denominator='Observed cases with recorded evidence; outcomes may overlap within a case'))
        return result
