"""Deterministic unit and timing contracts for passive contact telemetry."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('gripper_contact', Path(__file__).with_name('gripper_contact.py'))
contact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contact)


def report(actor0, actor1, impulse=(.004, 0, 0)):
    return ([SimpleNamespace(actor0=actor0, actor1=actor1, collider0=actor0+'/shape',
        collider1=actor1+'/shape', contact_data_offset=0, num_contact_data=1, type='PERSISTS')],
        [SimpleNamespace(impulse=impulse, normal=(1, 0, 0), position=(0, 0, 0), separation=0)])


class ContactWindowTest(unittest.TestCase):
    def setUp(self):
        self.window = contact.ContactWindow({arm: {name: '/'+arm+'/'+name
            for name in contact.FINGERS} for arm in contact.ARMS}, .004, .04)

    def complete(self, reports=None):
        self.window.begin(1)
        for index in range(10):
            self.window.record_reports(*(reports[index] if reports else ([], [])))
        return self.window.finish()

    def test_backend_is_checked_before_native_import_or_mutation(self):
        contact.require_supported_backend('cpu')
        for device in ('cuda:0', 'cuda', 'gpu', None, ''):
            with self.assertRaisesRegex(RuntimeError, 'unsupported_physics_backend'):
                contact.NativeContactMonitor(SimpleNamespace(device=device), {}, .04)

    def test_unknown_not_zero(self):
        self.assertFalse(self.window.query()['valid'])
        self.assertNotIn('gripper_contact_force_world_mean_N', self.window.observation())

    def test_zero_is_valid_after_ten_steps(self):
        result = self.complete()
        self.assertTrue(result['valid'])
        self.assertEqual(result['sample_count'], 10)
        self.assertEqual(result['arms']['left']['link7']['force_magnitude_peak_N'], 0)

    def test_impulse_units_zero_inclusive_average_and_pair_sign(self):
        reports = [report('/left/link7', '/object')] + [([], [])] * 9
        result = self.complete(reports)
        finger = result['arms']['left']['link7']
        self.assertAlmostEqual(finger['force_world_mean_N'][0], .1)
        self.assertAlmostEqual(finger['force_magnitude_peak_N'], 1.)
        self.assertAlmostEqual(finger['contact_ratio'], .1)
        result = self.complete([report('/object', '/left/link7')] * 10)
        self.assertAlmostEqual(result['arms']['left']['link7']['force_world_mean_N'][0], -1.)

    def test_object_filter_recomputes_and_does_not_mutate_cache(self):
        self.complete([report('/left/link7', '/object')] * 5 +
                      [report('/left/link7', '/table', (.008, 0, 0))] * 5)
        value = self.window.query(arm='left', object_id='/object')
        self.assertEqual(set(value['arms']), {'left'})
        self.assertAlmostEqual(value['arms']['left']['link7']['force_world_mean_N'][0], .5)
        self.assertAlmostEqual(self.window.query()['arms']['left']['link7']['force_world_mean_N'][0], 1.5)
        value['arms'].clear()
        self.assertTrue(self.window.query()['arms'])
        missing = self.window.query(object_id='/missing')
        self.assertEqual(missing['arms']['left']['link7']['force_magnitude_peak_N'], 0)

    def test_partial_and_query_error_invalid(self):
        self.window.begin(3)
        self.window.record_reports([], [])
        self.assertFalse(self.window.finish()['valid'])
        self.window.begin(4)
        for _ in range(10):
            self.window.record_reports([], [])
        self.window.record_error(RuntimeError('sensor failure'))
        self.assertFalse(self.window.finish()['valid'])
        self.assertNotIn('gripper_contact_force_world_mean_N', self.window.observation())

    def test_reset_clears_and_no_capture_outside_window(self):
        self.complete([report('/left/link7', '/object')] * 10)
        self.window.clear()
        self.window.record_reports(*report('/left/link7', '/object'))
        self.assertFalse(self.window.query()['valid'])
        self.assertEqual(self.window.query()['samples'], [])

    def test_sleeping_finger_unknown_does_not_hide_awake_arm(self):
        self.window.begin(1)
        for _ in range(10):
            self.window.record_reports(*report('/left/link7', '/object'),
                sleeping={'/right/link7': True, '/right/link8': True})
        result = self.window.finish()
        self.assertFalse(result['valid'])
        self.assertTrue(result['reports_valid'])
        self.assertFalse(result['arms']['right']['link7']['valid'])
        self.assertNotIn('force_world_mean_N', result['arms']['right']['link7'])
        self.assertTrue(self.window.query(arm='left')['valid'])
        self.assertFalse(self.window.query(arm='right')['valid'])
        obs = self.window.observation()
        np.testing.assert_array_equal(obs['gripper_contact_valid_mask'], [[True, True], [False, False]])
        self.assertIn('gripper_contact_left_link7_force_world_mean_N', obs)
        self.assertNotIn('gripper_contact_right_link7_force_world_mean_N', obs)
        self.assertTrue(self.window.query(arm='left', object_id='/object')['valid'])

    def test_malformed_and_nan_rejected(self):
        self.window.begin(1)
        with self.assertRaises(ValueError):
            self.window.record_reports(*report('/left/link7', '/object', (float('nan'), 0, 0)))
        headers, data = report('/left/link7', '/object')
        headers[0].num_contact_data = 2
        with self.assertRaises(ValueError):
            self.window.record_reports(headers, data)


if __name__ == '__main__':
    unittest.main()
