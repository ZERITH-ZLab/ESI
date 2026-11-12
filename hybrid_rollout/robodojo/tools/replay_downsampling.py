"""Paired open-loop selector replay; never calls a model, simulator, or account.

Frozen proposals, approvals, observations and GPT corrections are inputs, not
predictions of a new physical rollout. Baseline selection must match every saved
selection exactly. Native URDF FK is checked against every recorded H50 preview.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from ..flywheel.action_downsampling import ActionDownsampler,SelectionConfig,configuration_dict
from ..robodojo_server.kinematics import ArmFK


class OfflineFK:
    def __init__(self,urdf):
        self.arm=ArmFK(urdf,[f'joint{i}' for i in range(1,7)])
        self.cache={}
    def matrices(self,qs):
        qs=np.asarray(qs,float)
        t=np.broadcast_to(np.eye(4),(len(qs),4,4)).copy()
        for name,kind,origin,axis in self.arm.chain:
            m=np.broadcast_to(np.eye(4),(len(qs),4,4)).copy()
            if kind in ('revolute','continuous'):
                m[:,:3,:3]=Rotation.from_rotvec(qs[:,self.arm.names.index(name),None]*axis).as_matrix()
            elif kind!='fixed':raise ValueError('Unsupported URDF joint')
            t=t@origin@m
        return t
    def __call__(self,rows):
        rows=np.asarray(rows,float)
        missing={row.tobytes():row for row in rows if row.tobytes() not in self.cache}
        if missing:
            values=np.asarray(list(missing.values()))
            fk=np.stack([self.matrices(values[:,:6]),self.matrices(values[:,7:13])],axis=1)
            self.cache.update(zip(missing,fk))
        return np.asarray([self.cache[row.tobytes()] for row in rows])
    def verify(self,actions,preview):
        local=self(actions);error=0.
        for i,arm in enumerate(('left','right')):
            recorded=[]
            for row in preview:
                p=row[arm];q=p['quaternion_wxyz'];t=np.eye(4)
                t[:3,:3]=Rotation.from_quat([*q[1:],q[0]]).as_matrix();t[:3,3]=p['position']
                recorded.append(t)
            if len(recorded)!=len(actions):raise ValueError('Incomplete saved FK preview')
            # Rigid base transforms cancel in Euclidean/rotation errors. Recover
            # the base transform only to verify the archived full trajectory.
            root=recorded[0]@np.linalg.inv(local[0,i])
            error=max(error,float(np.max(np.abs(root@local[:,i]-recorded))))
        if error>1e-7:raise ValueError(f'URDF FK does not match saved preview: {error}')
        return error


def frozen_module(archive):
    path=archive/'source_snapshot/hybrid_rollout/robodojo/flywheel/action_downsampling.py'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    name='_frozen_downsampler_'+digest
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name],digest


def replay_episode(item,urdf,stream,candidate_config=None):
    archive=Path(item['archive']);controller=archive/'controller'
    result=json.loads((controller/'result.json').read_text())
    history=json.loads((controller/'history.json').read_text())
    fk=OfflineFK(urdf);module,digest=frozen_module(archive)
    cfg=result['action_downsampling']['configuration']
    if cfg['gripper_epsilon']!=.005:raise ValueError('Expected baseline epsilon .005')
    baseline=module.ActionDownsampler(module.SelectionConfig(**cfg),fk)
    newcfg=(replace(SelectionConfig(**cfg),closing_hold_epsilon=.02)
            if candidate_config is None else candidate_config)
    matched=ActionDownsampler(newcfg,fk);clock=ActionDownsampler(newcfg,fk)
    totals=Counter(episodes=1);max_fk=0.
    for row in history:
        decision=row['decision'];exec_path=controller/f'execution_{decision:03d}.npz'
        if not exec_path.exists():
            if row['executed_steps']:raise ValueError('Missing executed actions')
            continue
        with np.load(exec_path) as d:
            actions=d['actions'].copy()
            if len(actions)!=row['executed_steps']:raise ValueError('ACK count changed')
            if 'original_window' not in d:
                request=json.loads((controller/f'request_{decision:03d}.json').read_text())
                initial=np.asarray(request['current_state'])
                for selector in (baseline,matched,clock):selector.acknowledge(actions,initial)
                totals['gpt_steps']+=len(actions)
                continue
            initial=d['initial_measured_state'].copy();window=d['original_window'].copy()
            actual_indices=d['proposal_action_index'].tolist()
        if not len(actions):continue
        saved=json.loads((controller/f'selection_{decision:03d}.json').read_text())
        with np.load(controller/f'proposal_{decision:03d}.npz') as d:proposal=d['actions'].copy()
        request=json.loads((controller/f'request_{decision:03d}.json').read_text())
        np.testing.assert_array_equal(proposal[:len(window)],window)
        np.testing.assert_array_equal(proposal[actual_indices],actions)
        fk.cache.clear();max_fk=max(max_fk,fk.verify(proposal,request['student_eef_trajectory']))
        old,old_report=baseline.select(window,initial,lookahead=proposal)
        if old!=saved['retained_indices']:
            raise ValueError(f'Baseline does not reproduce saved selection: {item["case_id"]} {decision}: {old} != {saved["retained_indices"]}')
        if old_report['protected_gripper_indices']!=saved['protected_gripper_indices']:
            raise ValueError('Baseline gripper state differs')
        same,new_same=matched.select(window,initial,lookahead=proposal)
        counterfactual,new_clock=clock.select(window,initial,lookahead=proposal)
        reach=actual_indices[-1]+1
        actual_same=[i for i in same if i<reach];actual_clock=[i for i in counterfactual if i<reach]
        totals.update(windows=1,original=len(window),old_selected=len(old),
            new_matched_selected=len(same),new_clock_selected=len(counterfactual),
            reached_original=reach,old_ack=len(actions),new_matched_reached=len(actual_same),
            new_clock_reached=len(actual_clock),partial_windows=int(len(actual_indices)!=len(old)),
            old_protected=len(saved['protected_gripper_indices']),
            new_matched_protected=len(new_same['protected_gripper_indices']),
            new_clock_protected=len(new_clock['protected_gripper_indices']))
        if len(actual_indices)==len(old):
            totals.update(full_original=len(window),full_old_selected=len(old),
                          full_new_matched_selected=len(same),full_new_clock_selected=len(counterfactual))
        stream.write(json.dumps(dict(task=item['task'],case_id=item['case_id'],decision=decision,
            original_count=len(window),reached_original=reach,old_indices=old,
            new_matched_indices=same,new_clock_indices=counterfactual,
            old_actual_indices=actual_indices,
            new_clock_hold_before=new_clock['closing_hold_remaining_before']))+'\n')
        # Matched history isolates rule changes. The counterfactual clock counts
        # candidate retained targets, avoiding an artificially fast hold expiry.
        # For terminal partial windows, stop at the original reached horizon.
        baseline.acknowledge(actions,initial);matched.acknowledge(actions,initial)
        clock.acknowledge(window[actual_clock],initial)
    if totals['old_ack']!=result['student_steps']:raise ValueError('Student action coverage mismatch')
    if totals['gpt_steps']!=result['edited_steps']+result['recovery_steps']:
        raise ValueError('GPT action coverage mismatch')
    return dict(item,**totals,max_fk_matrix_error=max_fk,frozen_algorithm_sha256=digest,
                candidate_configuration=configuration_dict(newcfg))


def summarize(rows):
    c=Counter()
    for r in rows:
        for k,v in r.items():
            if type(v) is int:c[k]+=v
    for label,key in [('old','old_selected'),('new_matched','new_matched_selected'),('new_clock','new_clock_selected')]:
        c[label+'_removed']=c['original']-c[key]
        c[label+'_removed_fraction']=1-c[key]/c['original']
    c['old_reached_removed_fraction']=1-c['old_ack']/c['reached_original']
    c['new_clock_reached_removed_fraction']=1-c['new_clock_reached']/c['reached_original']
    return dict(c)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-config',type=Path,
                   help='Explicit selector JSON; omitted preserves the original v3 replay')
    p.add_argument('--urdf',type=Path,default=Path('/opt/RoboDojo/Assets/Robots/x5/X5A.urdf'))
    args=p.parse_args();cohort=json.loads(args.cohort.read_text())
    candidate=None if args.candidate_config is None else SelectionConfig(**json.loads(args.candidate_config.read_text()))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cohort.json').write_text(json.dumps(cohort,indent=2))
    rows=[]
    with (args.output/'windows.jsonl').open('w') as stream:
        for item in cohort['episodes']:
            row=replay_episode(item,args.urdf,stream,candidate);rows.append(row);stream.flush()
            (args.output/'episodes.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(dict(event='openloop_episode_done',case=item['case_id'],
                windows=row['windows'],old=row['old_selected'],new=row['new_clock_selected'])),flush=True)
    tasks=defaultdict(list)
    for row in rows:tasks[row['task']].append(row)
    summary=dict(scope='frozen_inputs_open_loop_not_physical_success_or_speed_validation',
        metrics='approved_prefix_counts; reached counts clip both candidates at historical terminal horizon',
        history_modes=dict(matched='candidate protection advanced by original ACK history',
            clock='candidate protection advanced by its retained commands; observations/proposals/GPT remain frozen'),
        total=summarize(rows),previous24=summarize([r for r in rows if r.get('previous24')]),
        tasks={k:summarize(v) for k,v in tasks.items()},
        previous24_tasks={k:summarize([r for r in v if r.get('previous24')]) for k,v in tasks.items()},
        baseline_exact_match_windows=sum(r['windows'] for r in rows),
        max_fk_matrix_error=max(r['max_fk_matrix_error'] for r in rows),
        urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        candidate_source_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath('flywheel/action_downsampling.py').read_bytes()).hexdigest())
    summary['candidate_configuration']=rows[0]['candidate_configuration'] if rows else None
    summary['candidate_helper_sha256']={name:hashlib.sha256(Path(__file__).parents[1].joinpath('flywheel',name).read_bytes()).hexdigest()
        for name in ('closing_phase.py','significant_closing.py')}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(event='openloop_finished',summary=str(args.output/'summary.json'),total=summary['total'])),flush=True)


if __name__=='__main__':main()
