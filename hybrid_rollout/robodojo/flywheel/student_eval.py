"""Pure student inference and native stepping; no teacher process or decisions."""
import argparse
import time
import traceback
from pathlib import Path
import numpy as np
from ..io import write_json
from ..pi05_server.client import Pi05Client
from ..robodojo_server.protocol import RPCClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sim-port',type=int,required=True)
    parser.add_argument('--student-port',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--evaluation-method',choices=['pi05_only'],required=True)
    parser.add_argument('--codex')
    parser.add_argument('--max-decisions',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    student=sim=None
    episode=None; tick=0; history=[]; began=time.monotonic()
    try:
        student=Pi05Client(args.student_port,args.checkpoint)
        sim=RPCClient('127.0.0.1',args.sim_port,timeout=600)
        meta=sim.request('metadata')
        if meta['task']!=args.task or args.max_decisions!=0:
            raise ValueError('Require the exact task and native episode horizon')
        reset=sim.request('reset',seed=args.seed,source='student',policy_version=student.metadata['checkpoint_sha256'])
        episode=reset['episode_id']; tick=reset['step_id']
        def rpc(op,**fields):
            return sim.request(op,episode_id=episode,step_id=tick,**fields)
        if 'pi_force_hold' in reset.get('metadata',{}):
            meta['pi_force_hold']=reset['metadata']['pi_force_hold']
        force_hold_enabled=bool(meta.get('pi_force_hold',{}).get('enabled',False))
        force_hold_intervened=False
        proposal_steps=hold_steps=tighten_steps=0
        hold_run=({'pi_force_hold':meta['pi_force_hold'],
            'force_hold_enabled':True,'control_class':'pi05_force_assisted',
            'student_steps_semantics':'actual student-source ACKs including force hold; student_proposal_steps excludes hold'}
            if force_hold_enabled else {})
        write_json(args.output/'run.json',dict(evaluation_method='pi05_only',teacher_off=True,
            student_identity_sha256=student.metadata['checkpoint_sha256'],seed=args.seed,
            evaluation_case=reset['metadata']['evaluation_case'],execution_chunk=15,
            control_dt=meta['control_dt'],max_episode_steps=meta['max_episode_steps'],**hold_run))
        terminal=False
        while not terminal:
            obs=rpc('teacher_observation')  # Cached RGB/state; model input is allowlisted by Pi05Client.
            index=len(history)
            actions,prediction=student.infer(obs,args.output/f'proposal_{index:03d}.npz')
            hold_args={'original_action_count':len(actions)} if force_hold_enabled else {}
            result=rpc('chunk_step',actions=actions[:15],student_prediction_id=prediction['prediction_id'],**hold_args)
            zero_ack_reinfer=bool(force_hold_enabled and not result['steps'] and result.get('reinfer_required')
                and result.get('force_hold_events'))
            if (not result['steps'] and not zero_ack_reinfer) or not all(r['valid'] for r in result['steps']):
                raise RuntimeError('Invalid native control receipt')
            tick=result['step_id']; last=result['steps'][-1] if result['steps'] else None
            terminal=bool(last and (last['terminated'] or last['truncated']))
            extra={}
            if force_hold_enabled:
                consumed=result['proposal_steps_consumed']; held=result['force_hold_steps']
                if consumed+held!=len(result['steps']) or not 0<=consumed<=min(15,len(actions)):
                    raise RuntimeError('Force-hold counts disagree with actual ACKs')
                proposal_steps+=consumed; hold_steps+=held
                force_hold_intervened|=bool(result['force_hold_events']) or any(
                    r.get('gripper_override') for r in result['steps'])
                extra=dict(decision=index,start_tick=tick-len(result['steps']),end_tick=tick,
                    response=dict(mode='student',steps=min(15,len(actions))),source='student',
                    selected_student_steps=min(15,len(actions)),
                    student_proposal_steps=consumed,force_hold_steps=held,
                    discarded_student_steps=len(actions)-consumed,
                    force_hold_events=result['force_hold_events'],reinfer_required=result.get('reinfer_required',False),
                    force_hold_state_transitions=result.get('force_hold_state_transitions',[]))
                if meta['pi_force_hold'].get('version') in ('contact_tighten_v2', 'contact_margin_v3', 'contact_zero_v4'):
                    count=result['tighten_steps']
                    if type(count) is not int or not 0<=count<=held:
                        raise RuntimeError('Tighten count is not a subset of extra ACKs')
                    tighten_steps+=count
                    extra['tighten_steps']=count
                np.savez_compressed(args.output/f'execution_{index:03d}.npz',
                    actions=np.asarray([r['executed_action'] for r in result['steps']],np.float32).reshape(-1,14),
                    states=np.asarray([r['obs']['states'] for r in result['steps']],np.float32).reshape(-1,14),
                    proposal_action_index=np.asarray([r.get('proposal_action_index') if r.get('proposal_action_index') is not None else -1 for r in result['steps']],np.int64),
                    trigger_proposal_action_index=np.asarray([r.get('trigger_proposal_action_index') if r.get('trigger_proposal_action_index') is not None else -1 for r in result['steps']],np.int64),
                    hold_event_id=np.asarray([r.get('hold_event_id') or '' for r in result['steps']],dtype=str))
            history.append(dict(prediction_id=prediction['prediction_id'],end_step=tick,executed_steps=len(result['steps']),**extra))
            write_json(args.output/'history.json',history)
            write_json(args.output/'progress.json',dict(step_id=tick,predictions=len(history),teacher_off=True))
        final=rpc('finish_pilot',reason='terminal')
        final.update(complete=True,evaluation_method='pi05_only',teacher_off=True,
            student_steps=tick,edited_steps=0,recovery_steps=0,predictions=len(history),
            wall_seconds=time.monotonic()-began)
        if force_hold_enabled:
            final.update(student_proposal_steps=proposal_steps,force_hold_steps=hold_steps,
                pi_force_hold=meta['pi_force_hold'],force_hold_enabled=True,
                force_hold_intervened=force_hold_intervened,control_class='pi05_force_assisted')
            if meta['pi_force_hold'].get('version') in ('contact_tighten_v2', 'contact_margin_v3', 'contact_zero_v4'):
                final['tighten_steps']=tighten_steps
        write_json(args.output/'result.json',final)
    except BaseException:
        write_json(args.output/'failure.json',dict(error=traceback.format_exc(),step_id=tick,episode_id=episode))
        raise
    finally:
        if sim is not None: sim.close()
        if student is not None: student.close()


if __name__=='__main__':
    main()
