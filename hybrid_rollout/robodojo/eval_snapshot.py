"""One immutable execution source per eval queue, independent of checkout path."""
import json
from pathlib import Path
import tempfile

from .deployment import RUNTIME_SOURCE_SCOPE, snapshot, source_identity, source_scope, verify_snapshot
from .io import require, write_json


def requested_source(cfg, env):
    root = Path(cfg.get('execution_source', env['CODE_ROOT'])).resolve()
    require((root/'hybrid_rollout/robodojo/robodojo_server/client.py').is_file(),
            f'Missing episode source: {root}')
    if cfg.get('execution_source'):
        verify_snapshot(root)
    return root


def experiment_source(cfg, env, queue_root, *, dry_run=False, temporary=None):
    """Caller owns queue.lock for writes. Dry runs use temporary snapshots only.

    Existing experiments never inspect the current checkout or original pin.
    Missing/incomplete snapshots are errors, not permission to change policy.
    """
    queue_root = Path(queue_root)
    target = queue_root/'source_snapshot'
    if target.exists():
        require(not target.is_symlink(), 'Experiment snapshot must not be a symlink')
        verify_snapshot(target)
        require((target/'hybrid_rollout/robodojo/robodojo_server/client.py').is_file(),
                'Experiment snapshot is missing the episode implementation')
        return target
    require(not (queue_root/'queue_config.json').exists(),
            'Existing eval has no experiment source_snapshot. Refusing to resume with new code; '
            'restore its original snapshot or use a new experiment name.')
    root = requested_source(cfg, env)
    scope = source_scope(root) if cfg.get('execution_source') else RUNTIME_SOURCE_SCOPE
    if dry_run:
        require(temporary is not None, 'Dry run requires a temporary snapshot directory')
        destination = Path(temporary)/'source_snapshot'
        freeze_source(root, destination, scope)
        return destination
    queue_root.mkdir(parents=True, exist_ok=True)
    # Publish only a completely copied and verified directory. A failed copy
    # leaves neither a reusable partial snapshot nor a queue config.
    with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=queue_root) as staging:
        destination = Path(staging)/'source_snapshot'
        freeze_source(root, destination, scope)
        destination.rename(target)
    return target


def freeze_source(root, destination, scope):
    before = source_identity(root, scope=scope)
    copied = snapshot(root, destination, scope=scope)
    require(before == copied == source_identity(root, scope=scope) == verify_snapshot(destination),
            'Execution source changed while freezing; retry with a stable checkout')


def bind_queue_config(queue_root, output, frozen, source, *, dry_run=False, retry=False):
    """Bind config AND source digest; retries must share their parent's identity."""
    frozen = dict(frozen, execution_source_sha256=verify_snapshot(source))
    parent = Path(queue_root)/'queue_config.json'
    if retry:
        require(parent.is_file(), 'Retry requires an initialized parent eval queue')
        base = {k:v for k,v in frozen.items() if k != 'retry_selection'}
        require(json.loads(parent.read_text()) == base, 'Parent eval queue config or source changed')
    path = Path(output)/'queue_config.json'
    if path.exists():
        require(json.loads(path.read_text()) == frozen, 'Eval queue config or source changed')
    elif not dry_run:
        write_json(path, frozen)
    return frozen
