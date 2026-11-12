"""Opt-in teacher task constraint; never rewrites VLA inputs or gates actions."""
import os

from .io import require

ENV_KEY = 'ROLLOUT_PLACEMENT_ORDER'
ORDERS = ('1234', '4321', '2314')
VERSION = 'placement_order_task_constraint_v1'


def validate(order):
    require(isinstance(order, str) and order in ORDERS, 'Placement order must be 1234, 4321 or 2314')
    return order


def configuration(env=None, *, task=None):
    env = os.environ if env is None else env
    order = env.get(ENV_KEY, '')
    if not order:
        return None
    validate(order)
    require((task or env.get('ROBODOJO_TASK')) == 'arrange_largest_number',
            'Placement order requires arrange_largest_number')
    require(env.get('ROLLOUT_STRICT_EVAL') == '1' and
            env.get('ROLLOUT_CONTROLLER_ROUTE') == 'subtask_feedforward_v1',
            'Placement order requires strict feedforward evaluation')
    return dict(version=VERSION, order=order, pad_numbers=[int(x) for x in order],
                numbering='left_to_right_in_initial_main_camera',
                constraint='final_pad_completion_order', teacher_only=True,
                native_success_unchanged=True, order_review='manual_video')


def identity(env, panel):
    value = configuration(env, task=panel.get('task'))
    if value is None:
        return {}
    require(panel.get('task') == 'arrange_largest_number' and len(panel['cases']) == 1 and
            all(c['variant'] == 'standard' and c['layout_id'] == 0 and c['eval_seed'] == 0
                and c['policy_rng_seed'] == 0 for c in panel['cases']),
            'Placement order experiment requires standard layout0, eval_seed0, policy_seed0')
    return dict(placement_order=value)


def render(value):
    order = ' -> '.join(validate(value['order']))
    return f'''\n## Episode-specific placement order

In the INITIAL main-camera view, number the four destination pads 1, 2, 3, 4
from left to right. Keep these pad identities fixed throughout this episode.
The final arrangement is unchanged: digits descend from left to right, largest
on pad 1. The required order of filling those pads is {order}.

Process the digit belonging to the current pad; visibly place and release it on
that final pad before beginning to acquire the digit for the next pad. These
numbers identify destination pads, NOT digit values or a new final arrangement.
Regrasping, temporary staging, handover and correction of the current digit are
allowed; temporary staging does not complete its final placement. Repair accidental
disturbances as needed without deliberately advancing to a later target out of order.
Keep the current pad/object in current_intent and task_progress. Do not change
this order merely to follow a different student proposal. Retain this requirement
after compaction and use same-episode history if progress is unclear.

Usual safety, observations, action tools, native budget and termination remain
unchanged. This instruction supplies no object coordinates or grasping solution.
'''
