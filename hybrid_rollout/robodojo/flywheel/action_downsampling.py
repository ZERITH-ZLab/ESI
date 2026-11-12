"""Error-bounded joint-target thinning; never synthesizes executed actions.

Defaults are engineering starting points, not calibrated task guarantees.
An ACK remains one native control tick; thinning does not prove speedup.
"""
ALGORITHM_VERSION = "joint_fk_rdp_significant_gripper_v2"
SPLIT_GRIPPER_VERSION = "joint_fk_rdp_split_gripper_v3"
PHASE_GRIPPER_VERSION = "joint_fk_rdp_closing_phase_v4"
FINITE_GRIPPER_VERSION = "joint_fk_rdp_significant_closing_v5"

from dataclasses import dataclass, asdict
from copy import deepcopy
import time
import numpy as np
from .closing_phase import ClosingPhases
from .significant_closing import SignificantClosingPhases

@dataclass(frozen=True)
class SelectionConfig:
    position_error_m: float = .002
    rotation_error_rad: float = .02
    max_skipped: int = 3
    max_joint_jump_rad: float = .15
    gripper_epsilon: float = 1e-6
    gripper_guard: int = 1
    closing_hold_ticks: int = 15
    gripper_reconstruction_error: float = .002
    max_gripper_jump: float = .01
    # None preserves the v2 protocol/identity. New runs explicitly choose .02.
    closing_hold_epsilon: float | None = None
    closing_phase_enabled: bool = False
    closing_onset_epsilon: float = .002
    closing_settle_ticks: int = 2
    closing_refresh_epsilon: float | None = None

    def __post_init__(self):
        for key in ('position_error_m', 'rotation_error_rad', 'max_joint_jump_rad', 'gripper_epsilon', 'gripper_reconstruction_error', 'max_gripper_jump'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('max_skipped','gripper_guard','closing_hold_ticks'):
            if type(getattr(self,key)) is not int or getattr(self,key)<0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.closing_hold_epsilon is not None and (
                not np.isfinite(self.closing_hold_epsilon) or
                self.closing_hold_epsilon < self.gripper_epsilon):
            raise ValueError('closing_hold_epsilon must be finite and >= gripper_epsilon')
        if type(self.closing_phase_enabled) is not bool:
            raise ValueError('closing_phase_enabled must be boolean')
        if not np.isfinite(self.closing_onset_epsilon) or self.closing_onset_epsilon <= 0:
            raise ValueError('closing_onset_epsilon must be finite and positive')
        if type(self.closing_settle_ticks) is not int or self.closing_settle_ticks < 1:
            raise ValueError('closing_settle_ticks must be a positive integer')
        if self.closing_phase_enabled and (self.closing_hold_epsilon is None or
                self.closing_onset_epsilon >= self.closing_hold_epsilon or
                self.closing_hold_ticks < self.closing_settle_ticks):
            raise ValueError('Closing phases require progress < closing threshold and hold >= settle ticks')
        if self.closing_refresh_epsilon is not None and (
                not self.closing_phase_enabled or not np.isfinite(self.closing_refresh_epsilon) or
                not 0 < self.closing_refresh_epsilon <= self.closing_hold_epsilon):
            raise ValueError('Finite closing requires phases and 0 < refresh epsilon <= closing threshold')


def algorithm_version(config):
    if config.closing_refresh_epsilon is not None:
        return FINITE_GRIPPER_VERSION
    if config.closing_phase_enabled:
        return PHASE_GRIPPER_VERSION
    return ALGORITHM_VERSION if config.closing_hold_epsilon is None else SPLIT_GRIPPER_VERSION


def configuration_dict(config):
    value = asdict(config)
    if value['closing_hold_epsilon'] is None:
        value.pop('closing_hold_epsilon')  # Existing frozen v2 identities still match.
    if not config.closing_phase_enabled:
        for key in ('closing_phase_enabled', 'closing_onset_epsilon', 'closing_settle_ticks'):
            value.pop(key)
    if config.closing_refresh_epsilon is None:
        value.pop('closing_refresh_epsilon')
    else:
        value.pop('closing_onset_epsilon')  # v5 has no .002 pending-phase onset.
    return value

class PreviewFK:
    """Reuse native robot-only FK RPC in H50 batches with exact target cache."""
    def __init__(self, rpc):
        self.rpc=rpc; self.cache={}; self.calls=0; self.seconds=0.
    def __call__(self, rows):
        rows=np.asarray(rows,float)
        missing={row.tobytes():row for row in rows if row.tobytes() not in self.cache}
        keys=list(missing)
        for start in range(0,len(keys),50):
            part=keys[start:start+50]; batch=np.asarray([missing[k] for k in part])
            padded=np.concatenate([batch,np.repeat(batch[-1:],50-len(batch),axis=0)])
            began=time.monotonic(); result=self.rpc('fk_preview',actions=padded)
            self.seconds+=time.monotonic()-began; self.calls+=1
            if result.get('physical_steps') != 0 or not result['measured_fk_check']['passed']:
                raise RuntimeError('FK preview failed measured validation or advanced physics')
            if len(result['trajectory']) != 50:
                raise ValueError('FK preview must return H50 trajectory')
            for key, item in zip(part,result['trajectory']):
                matrices=[]
                for arm in ('left','right'):
                    pose=item[arm]; q=np.asarray(pose['quaternion_wxyz'],float)
                    position=np.asarray(pose['position'],float)
                    if q.shape!=(4,) or position.shape!=(3,) or not np.isfinite(np.r_[q,position]).all() or abs(np.linalg.norm(q)-1)>1e-4:
                        raise ValueError('Invalid native FK pose')
                    t=np.eye(4);t[:3,3]=pose['position']
                    w,x,y,z=q/np.linalg.norm(q)
                    t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
                    matrices.append(t)
                self.cache[key]=np.asarray(matrices)
        return np.asarray([self.cache[row.tobytes()] for row in rows])

class ActionDownsampler:
    def __init__(self, config, fk):
        self.config=config; self.fk=fk
        self.previous_gripper=None; self.gripper_event_anchor=None; self.hold_remaining=np.zeros(2,dtype=int)
        self.closing_event_anchor=None
        self.closing_phases=None

    def _new_phases(self, opening):
        cls = SignificantClosingPhases if self.config.closing_refresh_epsilon is not None else ClosingPhases
        return cls(opening)

    def _events(self, g, local_anchor, hold_anchor):
        local_delta=g-local_anchor
        change=np.abs(local_delta)>self.config.gripper_epsilon
        threshold=self.config.closing_hold_epsilon
        threshold=self.config.gripper_epsilon if threshold is None else threshold
        hold_delta=g-hold_anchor
        hold_change=np.abs(hold_delta)>threshold
        closing=hold_delta < -threshold
        return (change, local_delta < -self.config.gripper_epsilon, closing,
                np.where(change,g,local_anchor), np.where(hold_change,g,hold_anchor))

    def acknowledge(self, executed_actions, initial_state):
        """Commit only actual native ACK targets; closing tick starts configured FOLLOWING ticks."""
        if self.previous_gripper is None:
            self.previous_gripper=np.asarray(initial_state,float)[[6,13]].copy()
        if self.gripper_event_anchor is None:
            self.gripper_event_anchor=np.asarray(initial_state,float)[[6,13]].copy()
        if self.closing_event_anchor is None:
            self.closing_event_anchor=np.asarray(initial_state,float)[[6,13]].copy()
        if self.config.closing_phase_enabled and self.closing_phases is None:
            self.closing_phases=self._new_phases(np.asarray(initial_state,float)[[6,13]])
        for action in np.asarray(executed_actions,float).reshape(-1,14):
            g=action[[6,13]]
            _,_,closing,self.gripper_event_anchor,self.closing_event_anchor=self._events(
                g,self.gripper_event_anchor,self.closing_event_anchor)
            if self.config.closing_phase_enabled:
                self.closing_phases.advance(g,self.config)
                self.hold_remaining=np.asarray([arm.remaining for arm in self.closing_phases.arms])
            else:
                self.hold_remaining=np.where(closing,self.config.closing_hold_ticks,
                                              np.maximum(self.hold_remaining-1,0))
            self.previous_gripper=g.copy()
    def select(self, window, initial_state, lookahead=None):
        began=time.monotonic(); c=self.config
        a=np.asarray(window,float);initial=np.asarray(initial_state,float)
        if a.ndim!=2 or a.shape[1]!=14 or not 1<=len(a)<=15 or initial.shape!=(14,) or not np.isfinite(np.r_[a.ravel(),initial]).all():
            raise ValueError('Expected finite original execution window <=15 x 14 and measured state14')
        n=len(a); required={0,n-1}; protected=set(); events=[]; hold_events=[]
        # Selection is pure: no state changes before acknowledged execution.
        prev=(initial[[6,13]] if self.gripper_event_anchor is None else self.gripper_event_anchor).copy()
        hold_anchor=(initial[[6,13]] if self.closing_event_anchor is None else self.closing_event_anchor).copy()
        pending=self.hold_remaining.copy()
        phases=deepcopy(self.closing_phases) if self.closing_phases is not None else self._new_phases(initial[[6,13]])
        phase_before=phases.snapshot();phase_events=[];phase_protected=set()
        close_start=np.zeros(2,dtype=int)
        proposal=a if lookahead is None else np.asarray(lookahead,float)
        if proposal.ndim!=2 or proposal.shape[1]!=14 or len(proposal)<n or not np.array_equal(proposal[:n],a) or not np.isfinite(proposal).all():
            raise ValueError('Lookahead must begin with the exact original window')
        for i,row in enumerate(proposal[:n+c.gripper_guard+1]):
            g=row[[6,13]]
            change,local_closing,closing,prev,hold_anchor=self._events(g,prev,hold_anchor)
            if np.any(change):
                protected.update(range(max(0,i-1-c.gripper_guard),min(n,i+c.gripper_guard+1)))
                events.append(dict(index=i,changed_arms=np.flatnonzero(change).tolist(),closing_arms=np.flatnonzero(local_closing).tolist()))
            if i<n:
                if c.closing_phase_enabled:
                    if c.closing_refresh_epsilon is not None:
                        for arm, state in enumerate(phases.arms):
                            if state.mode=='ready' and g[arm] >= state.peak-1e-12:
                                close_start[arm]=i
                    keep,changes=phases.advance(g,c)
                    phase_events.extend(dict(index=i,**event) for event in changes)
                    if any(keep):phase_protected.add(i)
                    if any(e['event'] in ('closing_candidate','renewed_closing') for e in changes):
                        phase_protected.add(max(0,i-1))
                    if c.closing_refresh_epsilon is not None:
                        for event in changes:
                            arm=event['arm']
                            if event['event']=='rearmed_on_opening':close_start[arm]=i
                            if event['event']=='closing_confirmed':
                                # Retrospective protection is restricted to this
                                # approved prefix. Unconfirmed small drift gets
                                # no blanket retention, nor a speculative tail.
                                phase_protected.update(range(int(close_start[arm]),i+1))
                else:
                    if np.any(closing):hold_events.append(dict(index=i,closing_arms=np.flatnonzero(closing).tolist()))
                    if np.any(pending>0) or np.any(closing): protected.add(i)
                    pending=np.where(closing,c.closing_hold_ticks,np.maximum(pending-1,0))
        protected.update(phase_protected)
        required.update(protected)
        def evaluate(rows):
            result=np.asarray(self.fk(rows),float)
            if result.shape!=(len(rows),2,4,4) or not np.isfinite(result).all():
                raise ValueError('FK must return finite N x 2 x 4 x 4 poses')
            return result
        joints=np.r_[0:6,7:13]; q=np.vstack([initial,a]);poses=evaluate(q)
        def segment(lo,hi):
            interior=np.arange(lo+1,hi)
            jump=float(np.max(np.abs(q[hi+1,joints]-q[lo+1,joints])))
            gjump=float(np.max(np.abs(q[hi+1,[6,13]]-q[lo+1,[6,13]])))
            if not len(interior):return interior,np.empty(0),0.,0.,jump,0.,gjump
            alpha=(interior-lo)/(hi-lo)
            interp=q[lo+1]+alpha[:,None]*(q[hi+1]-q[lo+1])
            predicted=evaluate(interp);truth=poses[interior+1]
            pos=np.linalg.norm(predicted[:,:,:3,3]-truth[:,:,:3,3],axis=-1)
            relative=predicted[:,:,:3,:3]@np.swapaxes(truth[:,:,:3,:3],-1,-2)
            rot=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1.,1.))
            gerror=np.abs(interp[:,[6,13]]-q[interior+1][:,[6,13]])
            score=np.maximum(np.maximum(pos/c.position_error_m,rot/c.rotation_error_rad).max(axis=1),
                             (gerror/c.gripper_reconstruction_error).max(axis=1))
            return interior,score,float(pos.max()),float(rot.max()),jump,float(gerror.max()),gjump
        retained=set(required);unmet=[]
        def split(lo,hi):
            interior,score,_,_,jump,_,gjump=segment(lo,hi)
            if hi-lo<=1:
                if jump>c.max_joint_jump_rad+1e-12:
                    unmet.append(dict(from_index=lo,to_index=hi,reason='original_adjacent_joint_jump',joint_jump_rad=jump))
                if gjump>c.max_gripper_jump+1e-12:
                    unmet.append(dict(from_index=lo,to_index=hi,reason='original_adjacent_gripper_jump',gripper_jump=gjump))
                return
            geometric=bool(np.any(score>1+1e-9))
            if geometric or hi-lo-1>c.max_skipped or jump>c.max_joint_jump_rad+1e-12 or gjump>c.max_gripper_jump+1e-12:
                middle=int(interior[np.argmax(score)]) if geometric else (lo+hi)//2
                retained.add(middle);split(lo,middle);split(middle,hi)
        anchors=[-1,*sorted(required)]
        for lo,hi in zip(anchors,anchors[1:]):split(lo,hi)
        indices=sorted(retained);maxp=maxr=maxscore=maxjump=maxgerror=maxgjump=0.
        for lo,hi in zip([-1,*indices[:-1]],indices):
            _,s,p,r,j,ge,gj=segment(lo,hi);maxp=max(maxp,p);maxr=max(maxr,r)
            maxgerror=max(maxgerror,ge);maxgjump=max(maxgjump,gj)
            maxscore=max(maxscore,float(s.max()) if len(s) else 0.);maxjump=max(maxjump,j)
        report=dict(mode='error',algorithm_version=algorithm_version(c),configuration=configuration_dict(c),threshold_status='uncalibrated_engineering_start',
            original_window_count=n,original_window_indices=list(range(n)),retained_indices=indices,
            retained_count=len(indices),retained_fraction=len(indices)/n,compression_ratio=n/len(indices),
            removed_fraction=1-len(indices)/n,protected_gripper_indices=sorted(protected),gripper_events=events,
            closing_hold_remaining_before=self.hold_remaining.tolist(),
            closing_hold_semantics=f'{c.closing_hold_ticks} following actual native ACK ticks, reset on each significant closing event',
            geometric_bound_scope='original action sample indices only; not continuous-path or measured-dynamics guarantee',
            joint_jump_scope='command endpoints only; unavoidable original adjacent jumps are reported, not certified safe',
            virtual_anchor_index=-1,virtual_anchor='current_measured_joint_state',
            max_reconstruction_position_error_m=maxp,max_reconstruction_rotation_error_rad=maxr,
            max_gripper_reconstruction_error=maxgerror,max_retained_gripper_jump=maxgjump,
            gripper_event_semantics='signed displacement from last significant ACK-committed value, not accumulated absolute motion',
            gripper_event_anchor_before=None if self.gripper_event_anchor is None else self.gripper_event_anchor.tolist(),
            max_normalized_combined_error=maxscore,max_normalized_fk_error=max(maxp/c.position_error_m,maxr/c.rotation_error_rad),max_retained_joint_jump_rad=maxjump,
            constraints_satisfied=not unmet and maxscore<=1+1e-9,unmet_constraints=unmet,
            selector_wall_seconds=time.monotonic()-began,
            timing_semantics='each retained target uses one native tick; frame reduction is not demonstrated speedup')
        if c.closing_hold_epsilon is not None:
            report.update(closing_hold_events=hold_events,
                closing_event_anchor_before=None if self.closing_event_anchor is None else self.closing_event_anchor.tolist(),
                closing_hold_semantics=f'{c.closing_hold_ticks} following actual native ACK ticks; reset only on signed closing displacement > {c.closing_hold_epsilon} from independent significant-opening/closing anchor; local jitter does not refresh it')
        if c.closing_phase_enabled:
            report.pop('closing_hold_events',None)
            report.pop('closing_event_anchor_before',None)
            report.update(closing_phase_before=phase_before,
                closing_phase_after_proposal=phases.snapshot(),
                closing_phase_events=phase_events,
                closing_phase_protected_indices=sorted(phase_protected),
                closing_hold_semantics=f'Protect pending/confirmed command closure; {c.closing_hold_ticks} following actual ACK ticks after last effective new-minimum closing progress; {c.closing_settle_ticks} settling ticks count inside tail; latched oscillations do not refresh; not contact detection')
        if c.closing_refresh_epsilon is not None:
            report['closing_hold_semantics']=(f'Net closing > {c.closing_hold_epsilon} starts '
                f'{c.closing_hold_ticks} following actual ACK ticks; only further new-low '
                f'closing > {c.closing_refresh_epsilon} renews; no pending/unbounded phase; '
                'confirmed onset backfilled inside approved prefix only; not contact detection')
        return indices,report
