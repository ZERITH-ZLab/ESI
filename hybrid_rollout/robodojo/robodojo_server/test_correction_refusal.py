"""A refused complete-goal RPC returns without issuing client-side actions."""
from types import SimpleNamespace
from .client import RoboDojoTools


def test_self_collision_refusal_returns_without_dispatching_action():
    refusal = dict(status="self_collision", error="blocked", executed_steps=0)
    sent = []
    target = dict(left={"position": [0, 0, 0], "gripper_closed": True},
                  right={"position": [0, 0, 0], "gripper_closed": False})
    def rpc(op, **payload):
        sent.append((op, payload))
        return dict(correction_result=refusal, step_id=17, steps=[])
    tools = SimpleNamespace(_rpc=rpc, tick=17, history=[], request={"request_id": "request"})
    rows = RoboDojoTools.execute_correction(tools, dict(mode="eef", steps=5, target=target))
    assert rows == [] and tools.tick == 17
    assert tools.correction_result == refusal
    assert sent == [("eef_short_motion", dict(targets=[target], gripper_intent=dict(left="closed", right="open"), minimum_steps=5,
                    decision=0, teacher_request_id="request"))]
