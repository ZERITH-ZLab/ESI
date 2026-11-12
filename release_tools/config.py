"""JSON composition and typed references. No shell expansion or environment probing."""
from copy import deepcopy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REF = re.compile(r'\$\{([A-Za-z0-9_.]+)\}')


def merge(left, right):
    out = deepcopy(left)
    for key, value in right.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else deepcopy(value)
    return out


def read(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f'Configuration include cycle: {path}')
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        parents = data.pop('extends', [])
        if isinstance(parents, str): parents = [parents]
        base = {}
        for parent in parents: base = merge(base, read(path.parent/parent, (*stack, path)))
        data = merge(base, data)
    def include(value):
        if isinstance(value, dict) and set(value) == {'$file'}:
            return read(path.parent/value['$file'], (*stack, path))
        if isinstance(value, dict): return {k:include(v) for k,v in value.items()}
        if isinstance(value, list): return [include(v) for v in value]
        return value
    return include(data)


def override(data, expression):
    if '=' not in expression: raise ValueError('--set expects dotted.key=JSON-value')
    key, raw = expression.split('=', 1)
    parts = key.split('.'); target = data
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f'Unknown config key: {key}')
        target = target[part]
    if parts[-1] not in target: raise ValueError(f'Unknown config key: {key}')
    try: value = json.loads(raw)
    except json.JSONDecodeError: value = raw
    target[parts[-1]] = value


def resolve(data):
    def lookup(key, trail):
        if key in trail: raise ValueError(f'Configuration reference cycle: {key}')
        value = data
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value: raise ValueError(f'Unknown reference: {key}')
            value = value[part]
        if value is None: raise ValueError(f'Set required configuration value: {key}')
        return visit(value, (*trail, key))
    def visit(value, trail=()):
        if isinstance(value, dict): return {k:visit(v, trail) for k,v in value.items()}
        if isinstance(value, list): return [visit(v, trail) for v in value]
        if not isinstance(value, str): return value
        match = REF.fullmatch(value)
        if match: return lookup(match[1], trail)
        def replace(m):
            v = lookup(m[1], trail)
            if isinstance(v, (dict, list)): raise ValueError('Object reference must occupy the whole value')
            return str(v)
        return REF.sub(replace, value)
    return visit(data)


def load(recipe, deployment, overrides=()):
    data = read(recipe)
    if 'deployment' in data: raise ValueError('Deployment belongs in the local deployment file, not the recipe')
    data['deployment'] = read(deployment)
    data['repo_root'] = str(ROOT)
    for value in overrides: override(data, value)
    return resolve(data)


def recursive_round_weights(rounds, latest=0.6):
    if not isinstance(rounds, list) or not rounds or len(rounds) != len(set(rounds)) or not all(isinstance(r,str) and r for r in rounds):
        raise ValueError('rounds must be distinct nonempty names ordered oldest to newest')
    if not 0 < latest < 1: raise ValueError('latest round probability must be in (0,1)')
    result = {}; remaining = 1.0
    for name in reversed(rounds[1:]):
        result[name] = remaining*latest
        remaining *= 1-latest
    result[rounds[0]] = remaining
    return {name:result[name] for name in rounds}
