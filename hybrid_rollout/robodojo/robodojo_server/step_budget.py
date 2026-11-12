"""Explicit training-only native step budget override; no reward changes."""
from ..collection import SCHEMA, validate
from ..io import require


def apply(env, requested=0, panel=None):
    native = int(env.step_lim)
    from ..regression_eval import SCHEMA as EVAL_SCHEMA, NATIVE_BUDGET
    if panel is not None and panel.get('schema') == EVAL_SCHEMA:
        require(requested == 0, 'Eval cannot override the native step budget')
        expected = panel.get('native_step_limit', NATIVE_BUDGET)
        require(native == expected,
                f'Native task budget changed: expected {expected}, read {native}; review eval protocol')
    if requested:
        require(panel is not None and panel.get('schema') == SCHEMA, 'Step override requires collection panel')
        validate(panel)
        limits = panel['collection']
        require(requested == limits['step_limit'] and native == limits['native_step_limit'],
                'Unexpected native or collection step limit')
        env.step_lim = requested
    elif panel is not None and panel.get('schema') == SCHEMA:
        raise ValueError('Collection simulator did not receive its step override')
    return dict(original_native_step_limit=native, effective_step_limit=int(env.step_lim),
                step_limit_overridden=bool(requested), budget_protocol='collection' if requested else 'native')
