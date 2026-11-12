"""Immutable prompt bundles, editable strategies separated from fixed task semantics."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ENV = 'ROLLOUT_GUIDANCE_BUNDLE_JSON'
# Reviewed paragraph allowlist. All unspecified task text stays immutable.
# Conservative: mixed goal/strategy paragraphs are fixed, not LLM-classified.
EDITABLE = {
    'arrange_largest_number': {1, 2, 3, 4, 5},
    'pack_objects_into_box': {1, 2, 3},
    'sort_nesting_dolls_by_size': {1, 2, 3, 4},
    'fill_pen_holder': {2, 3, 4},
    'sweep_blocks': {1, 2},
}
TASKS = ('pack_objects_into_box', 'arrange_largest_number', 'sort_nesting_dolls_by_size',
         'classify_objects', 'fill_pen_holder', 'organize_table', 'imitate_sorting_sequence',
         'stack_blocks_by_language', 'classify_objects_by_language', 'sweep_blocks')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def seal(bundle):
    result = {k: v for k, v in bundle.items() if k != 'sha256'}
    return dict(result, sha256=digest(result))


def seed(source=None):
    source = ROOT if source is None else Path(source)
    catalog = (source/'task_completion_briefs.md').read_text()
    shared = (source/'manipulation_guidance.md').read_text()
    chunks = re.split(r'^## (.+)\n', shared, flags=re.M)
    rules = [dict(id=f'shared_{i//2:02}', title=chunks[i], text=chunks[i+1].strip())
             for i in range(1, len(chunks), 2)]
    sections = re.split(r'^## ([A-Za-z0-9_]+)\n', catalog, flags=re.M)
    tasks = {}
    for i in range(1, len(sections), 2):
        name = sections[i]; base = name.removesuffix('_random')
        if base not in TASKS:
            continue
        tasks[name] = [dict(id=f'p{n}', editable=n in EDITABLE.get(base, set()), text=text)
                       for n, text in enumerate(sections[i+1].strip().split('\n\n'))]
    fixed = dict(shared_prefix=chunks[0], catalog=catalog,
                 task_fixed={t: [(p['id'], p['text']) for p in parts if not p['editable']]
                             for t, parts in tasks.items()})
    return seal(dict(schema='robodojo.guidance_bundle.v1', seed_sha256=digest(fixed),
                     parent_sha256=None, shared_prefix=chunks[0], shared_rules=rules,
                     catalog=catalog, tasks=tasks))


def validate(value, *, original=None):
    original = seed() if original is None else original
    if (value.get('schema') != original['schema'] or seal(value)['sha256'] != value.get('sha256')
            or value.get('seed_sha256') != original['seed_sha256']
            or value.get('shared_prefix') != original['shared_prefix']
            or value.get('catalog') != original['catalog'] or set(value.get('tasks', {})) != set(original['tasks'])):
        raise ValueError('Guidance bundle hash/immutable seed mismatch')
    for task, parts in value['tasks'].items():
        fixed = [(p['id'], p['text']) for p in parts if not p['editable']]
        expected = [(p['id'], p['text']) for p in original['tasks'][task] if not p['editable']]
        if fixed != expected or len({p['id'] for p in parts}) != len(parts):
            raise ValueError('Immutable task semantics changed or duplicate rule ID')
    if len({r['id'] for r in value['shared_rules']}) != len(value['shared_rules']):
        raise ValueError('Duplicate shared rule ID')
    return value


def load(path):
    return validate(json.loads(Path(path).read_text()))


def from_env(env=None):
    env = os.environ if env is None else env
    text = env.get(ENV)
    return validate(json.loads(text)) if text else None


def render(value, task, *, original=None):
    validate(value, original=original)
    if task not in value['tasks']:
        raise ValueError(f'Evolved bundle has no task {task}')
    shared = value['shared_prefix'].rstrip() + '\n\n' + '\n\n'.join(
        f"## {r['title']}\n{r['text']}" for r in value['shared_rules']) + '\n'
    section = f'## {task}\n\n' + '\n\n'.join(p['text'] for p in value['tasks'][task]) + '\n'
    return shared, section


def editable_text(value, scope):
    if scope == 'shared':
        return value['shared_rules']
    return {t: [p for p in ps if p['editable']] for t, ps in value['tasks'].items()
            if t.removesuffix('_random') == scope}


def apply_patch(value, patch, cfg, *, original=None):
    """Only a single strategy rule may change; fixed criteria cannot be addressed."""
    result = deepcopy(value); scope = patch['scope']; ident = patch['rule_id']
    text = patch['text'].strip(); op = patch['operation']
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', ident):
        raise ValueError('Invalid rule ID')
    if op not in ('replace', 'add', 'delete'):
        raise ValueError('Invalid patch operation')
    if op != 'delete' and (not text or len(text.split()) > cfg['max_rule_words'] or '\n#' in '\n'+text):
        raise ValueError('Rule must be short prose without nested prompt headings')
    if op == 'delete' and text:
        raise ValueError('Deletion must have empty text')
    if scope == 'shared' and op == 'add' and (not patch['title'].strip()
            or len(patch['title'].split()) > 12 or '\n' in patch['title']):
        raise ValueError('Shared rule title must be a short single line')
    if scope == 'shared':
        groups = [result['shared_rules']]
    else:
        groups = [v for k, v in result['tasks'].items() if k.removesuffix('_random') == scope]
        if not groups:
            raise ValueError('Unknown task scope')
    for group in groups:
        matches = [r for r in group if r['id'] == ident]
        if op == 'add':
            if matches or not ident.startswith('learned_'):
                raise ValueError('New rule needs a unique learned_ ID')
            group.append(dict(id=ident, text=text, **(
                dict(title=patch['title']) if scope == 'shared' else dict(editable=True))))
        else:
            if len(matches) != 1 or matches[0].get('editable', True) is not True:
                raise ValueError('Only existing editable strategies may change')
            if op == 'delete':group.remove(matches[0])
            else:matches[0]['text'] = text
    result['parent_sha256'] = value['sha256']
    return validate(seal(result), original=original)


def word_count(value, scope):
    # Count each runtime variant independently to enforce the per-task budget.
    if scope == 'shared':return sum(len(r['text'].split()) for r in value['shared_rules'])
    return max((sum(len(p['text'].split()) for p in ps if p['editable'])
                for t, ps in value['tasks'].items() if t.removesuffix('_random') == scope), default=0)
