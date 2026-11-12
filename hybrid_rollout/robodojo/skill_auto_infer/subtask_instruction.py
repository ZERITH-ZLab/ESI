"""Online instruction rendering matching subtasks.v9; parity-tested against training.

Kept inside the rollout snapshot, which does not distribute data_processing.
"""
import re
from ..io import InputError
from ..skill.schema import _obj

SKILLS = ('approach', 'align', 'grasp', 'transport', 'place', 'push', 'release',
          'rotate', 'handover', 'retreat', 'return_home', 'custom')
FIELDS = dict(skill_type=SKILLS, arm=('left','right','both'),
    object_ref=None, destination_ref=None, target_relation=('none','to','on','in','near'),
    target_orientation_ref=None, custom_description=None,
    giver_arm=('left','right','unknown'), receiver_arm=('left','right','unknown'))


def online_schema():
    from ..prompt_profile import freeform_subtask
    if freeform_subtask():
        return {'anyOf': [{'type':'string', 'minLength':1, 'maxLength':240}, {'type':'null'}]}
    return schema()


def render_online(value):
    from ..prompt_profile import freeform_subtask
    if not freeform_subtask():
        return render_instruction(value)
    if value is None:
        return None
    if (not isinstance(value, str) or not value.strip() or len(value) > 240
            or '\n' in value or '\r' in value):
        raise InputError('subtask must be null or one nonempty line of at most 240 characters')
    return value.strip()


def schema():
    properties = {k: ({'type':'string','enum':list(v)} if v else {'type':['string','null']})
                  for k,v in FIELDS.items()}
    return {'anyOf': [_obj(properties), {'type':'null'}]}


def canonical(i):
    """Only code writes training text. Missing slots produce no instruction."""
    skill, arm, target = i["skill_type"], i["arm"], i["object_ref"]
    if skill == "unknown" or arm == "unknown":
        return None
    label = lambda x: x.replace("_", " ") if x else None
    target, destination = label(target), label(i["destination_ref"])
    prefix = {"left": "Left arm", "right": "Right arm", "both": "Both arms"}[arm]
    orientation = label(i["target_orientation_ref"])

    def oriented(text):
        # The field always describes the OBJECT, never an inferred wrist pose.
        if orientation:
            if not target:
                return None
            return text + f" Target orientation of {target}: {orientation}."
        return text

    if skill == "custom":
        description = i["custom_description"]
        if not description or not description.strip():
            return None
        return oriented(f"{prefix}: {description.strip().rstrip('.')}.")
    if skill == "return_home":
        return oriented(f"{prefix}: Return home.")
    if not target:
        return None
    if skill == "handover":
        giver, receiver = i["giver_arm"], i["receiver_arm"]
        if arm != "both" or giver == "unknown" or receiver == "unknown" or giver == receiver:
            return None
        return oriented(f"Transfer {target} from the {giver} gripper to the {receiver} gripper.")
    if skill in ("transport", "place", "push", "reposition"):
        if not destination or i["target_relation"] == "none":
            return None
        return oriented(f"{prefix}: {skill.capitalize()} {target} {i['target_relation']} {destination}.")
    if skill == "rotate":
        return f"{prefix}: Rotate {target} to {orientation}." if orientation else None
    verb = {"approach": "Approach", "align": "Align with", "grasp": "Grasp",
            "retreat": "Retreat from", "release": "Release"}[skill]
    return oriented(f"{prefix}: {verb} {target}.")


def render_instruction(value):
    if value is None:
        return None
    if not isinstance(value,dict) or set(value) != set(FIELDS):
        raise InputError('subtask must be null or exactly the declared skill fields')
    for key, choices in FIELDS.items():
        item=value[key]
        if choices:
            if not isinstance(item,str) or item not in choices:
                raise InputError(f'Invalid subtask.{key}')
        elif item is not None and (not isinstance(item,str) or not item.strip()):
            raise InputError(f'subtask.{key} must be null or nonempty text')
    for key in ('object_ref','destination_ref','target_orientation_ref'):
        if value[key] is not None and not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*',value[key]):
            raise InputError(f'subtask.{key} must be a stable snake_case entity')
    if value['skill_type'] != 'custom' and value['custom_description'] is not None:
        raise InputError('custom_description is only for custom skills')
    if value['skill_type'] != 'handover' and (value['giver_arm'] != 'unknown' or value['receiver_arm'] != 'unknown'):
        raise InputError('giver_arm/receiver_arm must be unknown except for handover')
    if value['skill_type'] not in ('transport','place','push','custom') and (
            value['destination_ref'] is not None or value['target_relation'] != 'none'):
        raise InputError('Unused destination_ref/target_relation must be null/none')
    if value['custom_description'] is not None and (any(c in value['custom_description'] for c in ('\n', '\r')) or len(value['custom_description'])>240):
        raise InputError('custom_description must be a concise single-line instruction, at most 240 characters')
    text=canonical(value)
    if text is None:
        raise InputError('Subtask lacks required object/arm/destination/orientation; use null if unresolved')
    return text


def format_prompt(task, subtask):
    if not isinstance(task,str) or not task.strip():
        raise InputError('Episode instruction must not be empty')
    return task if subtask is None else f'{task}\nCurrent subtask: {subtask.strip()}'
