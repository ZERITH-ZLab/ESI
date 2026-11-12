import unittest
import numpy as np
from .action_downsampling import ActionDownsampler,SelectionConfig
from .test_action_downsampling import sample,toy_fk

def sampler(**kwargs):
    config=dict(gripper_epsilon=.005,max_skipped=99,max_joint_jump_rad=10)
    config.update(kwargs)
    return ActionDownsampler(SelectionConfig(**config),toy_fk)

class SignificantGripperTests(unittest.TestCase):
    def test_signed_accumulated_closing_across_chunks(self):
        s=sampler();initial=sample()[0];a=initial.copy();a[6]=.598
        s.acknowledge([a],initial);self.assertAlmostEqual(s.gripper_event_anchor[0],.6);self.assertEqual(s.hold_remaining[0],0)
        b=sample();b[:,6]=.594;idx,r=s.select(b,a)
        self.assertEqual(idx,list(range(15)));self.assertEqual(r['gripper_events'][0]['closing_arms'],[0])
        self.assertAlmostEqual(s.gripper_event_anchor[0],.6)
        s.acknowledge(b[:1],a);self.assertAlmostEqual(s.gripper_event_anchor[0],.594);self.assertEqual(s.hold_remaining[0],15)
    def test_bounded_oscillation_does_not_accumulate_absolute_motion(self):
        s=sampler();initial=sample()[0];a=sample();a[:,6]=.6+np.tile([.002,-.002,0],5)
        s.acknowledge(a,initial);self.assertAlmostEqual(s.gripper_event_anchor[0],.6);self.assertEqual(s.hold_remaining[0],0)
    def test_gripper_pulse_retained_despite_equal_endpoints(self):
        s=sampler(gripper_epsilon=1.,gripper_guard=0,closing_hold_ticks=0);a=sample();a[7,13]=.58
        idx,r=s.select(a,a[0]);self.assertIn(7,idx);self.assertLessEqual(r['max_gripper_reconstruction_error'],.002)
    def test_slow_linear_close_constrained_by_gripper_jump(self):
        s=sampler(gripper_epsilon=1.,gripper_guard=0,closing_hold_ticks=0);a=sample();a[:,6]=np.linspace(.6,.56,15)
        idx,r=s.select(a,a[0]);self.assertGreater(len(idx),2);self.assertLessEqual(r['max_retained_gripper_jump'],.01+1e-12)
        self.assertLess(r['max_gripper_reconstruction_error'],1e-12)
    def test_unavoidable_original_gripper_jump_reported(self):
        s=sampler(gripper_epsilon=1.);a=sample();a[7:,6]=.5
        idx,r=s.select(a,a[0]);self.assertIn(7,idx);self.assertFalse(r['constraints_satisfied'])
        self.assertTrue(any(u['reason']=='original_adjacent_gripper_jump' for u in r['unmet_constraints']))
    def test_measured_entry_gripper_jump_reported(self):
        s=sampler(gripper_epsilon=1.);a=sample();initial=a[0].copy();initial[13]=.8
        idx,r=s.select(a,initial);self.assertEqual(idx[0],0)
        self.assertTrue(any(u['from_index']==-1 and u['reason']=='original_adjacent_gripper_jump' for u in r['unmet_constraints']))
    def test_partial_and_zero_ack_never_commit_future_significant_event(self):
        s=sampler();a=sample();a[:,6]=.598;a[14,6]=.59;initial=sample()[0]
        idx,r=s.select(a,initial);s.acknowledge(a[:1],initial);s.acknowledge([],initial)
        self.assertAlmostEqual(s.gripper_event_anchor[0],.6);self.assertEqual(s.hold_remaining[0],0)
    def test_subepsilon_deleted_points_preserve_significant_anchor(self):
        s=sampler(gripper_guard=0,closing_hold_ticks=15,max_gripper_jump=.1,gripper_reconstruction_error=.01)
        a=sample();a[:,6]=np.linspace(.6,.594,15)
        idx,r=s.select(a,a[0]);self.assertLess(len(idx),15)
        full=sampler(gripper_guard=0,closing_hold_ticks=15,max_gripper_jump=.1,gripper_reconstruction_error=.01)
        full.acknowledge(a,a[0]);s.acknowledge(a[idx],a[0])
        np.testing.assert_array_equal(s.gripper_event_anchor,full.gripper_event_anchor)
        self.assertEqual(s.hold_remaining.tolist(),full.hold_remaining.tolist())
        self.assertEqual(s.hold_remaining[0],13)
        following=sample();following[:,6]=.594
        next_idx,_=s.select(following,following[0]);self.assertTrue(set(range(13))<=set(next_idx))
    def test_original_positional_config_compatibility(self):
        c=SelectionConfig(.002,.02,3,.15,1e-6,1,15)
        self.assertEqual((c.gripper_guard,c.closing_hold_ticks,c.gripper_reconstruction_error,c.max_gripper_jump),(1,15,.002,.01))
    def test_combined_score_distinguishes_gripper_from_fk_error(self):
        s=sampler(gripper_epsilon=1.,gripper_guard=0,closing_hold_ticks=0);a=sample();a[7,6]=.601
        idx,r=s.select(a,a[0]);self.assertEqual(idx,[0,14])
        self.assertAlmostEqual(r['max_normalized_combined_error'],.5)
        self.assertEqual(r['max_normalized_fk_error'],0.)
    def test_selected_original_continuous_values_are_never_quantized(self):
        s=sampler(gripper_epsilon=1.);a=sample();a[:,6]=np.linspace(.6,.599,15);before=a.copy()
        idx,r=s.select(a,a[0]);np.testing.assert_array_equal(a,before);np.testing.assert_array_equal(a[idx],before[idx])

if __name__=='__main__':unittest.main()
