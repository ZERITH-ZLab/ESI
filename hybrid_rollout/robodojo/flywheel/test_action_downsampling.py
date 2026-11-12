"""Offline selector tests: no model, simulator, teacher, or physical success claims."""
import unittest
import numpy as np
from .action_downsampling import SelectionConfig, ActionDownsampler, PreviewFK

def toy_fk(rows):
    rows=np.asarray(rows);t=np.tile(np.eye(4),(len(rows),2,1,1))
    for arm,o in ((0,0),(1,7)):
        q=rows[:,o:o+6];t[:,arm,0,3]=np.cos(q[:,0]);t[:,arm,1,3]=np.sin(q[:,0]);t[:,arm,2,3]=q[:,1]
        c,s=np.cos(q[:,2]),np.sin(q[:,2]);t[:,arm,0,0]=c;t[:,arm,0,1]=-s;t[:,arm,1,0]=s;t[:,arm,1,1]=c
    return t

def sample():
    a=np.zeros((15,14));a[:,[6,13]]=.6;return a

def selector(**kw):
    return ActionDownsampler(SelectionConfig(max_skipped=99,max_joint_jump_rad=10,**kw),toy_fk)

class SelectorTests(unittest.TestCase):
    def test_joint_interpolation_not_cartesian_chord(self):
        a=sample();a[:,0]=np.linspace(-1,1,15)
        idx,r=selector().select(a,a[0]);self.assertEqual(idx,[0,14]);self.assertLess(r['max_normalized_fk_error'],1)
    def test_bend_is_retained(self):
        a=sample();a[:,1]=np.r_[np.linspace(0,.1,8),np.linspace(.1,0,8)[1:]]
        idx,r=selector().select(a,a[0]);self.assertIn(7,idx);self.assertTrue(r['constraints_satisfied'])
    def test_right_arm_alone_enforces_common_selection(self):
        a=sample();a[7,8]=.1
        idx,r=selector().select(a,a[0]);self.assertIn(7,idx);self.assertEqual(r['max_reconstruction_position_error_m'],0.)
    def test_rotation_only_bend(self):
        a=sample();a[7,9]=.2
        idx,r=selector().select(a,a[0]);self.assertIn(7,idx);self.assertTrue(r['constraints_satisfied'])
    def test_max_skipped_boundary(self):
        a=sample();idx,r=ActionDownsampler(SelectionConfig(),toy_fk).select(a,a[0])
        self.assertTrue(all(j-i-1<=3 for i,j in zip([-1,*idx[:-1]],idx)))
        self.assertEqual((idx[0],idx[-1]),(0,14))
    def test_unavoidable_entry_jump_is_reported(self):
        a=sample();a[:,0]=.3;initial=a[0].copy();initial[0]=0
        idx,r=ActionDownsampler(SelectionConfig(),toy_fk).select(a,initial)
        self.assertEqual(idx[0],0);self.assertFalse(r['constraints_satisfied'])
        self.assertEqual(r['unmet_constraints'][0]['from_index'],-1)
    def test_joint_jump_splits_even_with_zero_fk_error(self):
        a=sample();a[:,3]=np.linspace(0,1,15)
        idx,r=ActionDownsampler(SelectionConfig(max_skipped=99),toy_fk).select(a,a[0])
        self.assertGreater(len(idx),2);self.assertLessEqual(r['max_retained_joint_jump_rad'],.15)
    def test_continuous_gripper_changes_preserve_density(self):
        a=sample();a[:,6]=np.linspace(.6,.59,15)
        idx,r=selector().select(a,sample()[0]);self.assertEqual(idx,list(range(15)))
    def test_opening_guard_preserves_both_sides(self):
        a=sample();a[7:,13]=.7
        idx,r=selector().select(a,a[0]);self.assertTrue(set(range(5,9))<=set(idx))
    def test_lookahead_closure_preserves_previous_guard(self):
        a=sample();future=np.vstack([a,a[:2]]);future[15:,6]=.4
        idx,r=selector().select(a,a[0],future);self.assertIn(13,idx);self.assertIn(14,idx)
    def test_close_hold_uses_acks_and_survives_chunk(self):
        a=sample();a[13:,6]=.4;s=selector();idx,r=s.select(a,a[0])
        self.assertIsNone(s.previous_gripper);self.assertEqual(s.hold_remaining.tolist(),[0,0])
        s.acknowledge(a[idx],a[0]);self.assertEqual(s.hold_remaining.tolist(),[14,0])
        b=sample();b[:,6]=.4;idx,r=s.select(b,b[0]);self.assertTrue(set(range(14))<=set(idx))
        s.acknowledge(b[:1],b[0]);self.assertEqual(s.hold_remaining.tolist(),[13,0])
    def test_unexecuted_closure_does_not_commit(self):
        a=sample();a[14,6]=.4;s=selector();idx,r=s.select(a,a[0])
        s.acknowledge(a[[i for i in idx if i<14]],a[0]);self.assertEqual(s.hold_remaining.tolist(),[0,0])
        self.assertAlmostEqual(s.previous_gripper[0],.6)
    def test_zero_ack_does_not_consume_hold(self):
        s=selector();a=sample();close=a[0].copy();close[6]=.4
        s.acknowledge([close],a[0]);s.acknowledge([],close);self.assertEqual(s.hold_remaining.tolist(),[15,0])
    def test_invalid_fk_rejected(self):
        a=sample()
        for bad in (lambda rows: np.full((len(rows),2,4,4),np.nan), lambda rows: np.zeros((len(rows),4,4))):
            with self.assertRaisesRegex(ValueError,'FK must return'):ActionDownsampler(SelectionConfig(),bad).select(a,a[0])
    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):SelectionConfig(position_error_m=0)
        with self.assertRaises(ValueError):SelectionConfig(max_skipped=-1)
    def test_single_frame_window(self):
        a=sample()[:1];idx,r=selector().select(a,a[0]);self.assertEqual(idx,[0])
    def test_unchanged_original_targets(self):
        a=sample();before=a.copy();idx,r=selector().select(a,a[0]);np.testing.assert_array_equal(a,before)
        np.testing.assert_array_equal(a[idx],before[idx])
    def test_fk_preview_batches_cache_and_zero_physics(self):
        calls=[]
        def rpc(op,actions):
            calls.append(actions.copy());return dict(physical_steps=0,measured_fk_check=dict(passed=True),trajectory=[
                {arm:dict(position=[row[o],0,0],quaternion_wxyz=[1,0,0,0]) for arm,o in [('left',0),('right',7)]} for row in actions])
        fk=PreviewFK(rpc);a=sample();fk(a);fk(a);self.assertEqual(len(calls),1);self.assertEqual(calls[0].shape,(50,14))

if __name__=='__main__':unittest.main()
