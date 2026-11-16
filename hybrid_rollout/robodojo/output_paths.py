"""Output directory validation and a bidirectional shared-storage probe."""
import argparse
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile


def validate_output_root(value, *, source_root=None, create=True):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError('Output directory must be a nonempty absolute path')
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f'Output directory must be an absolute path: {value}')
    path = path.resolve()
    source = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    if path == Path(path.anchor) or path.is_relative_to(source):
        raise ValueError('Outputs must be outside the source repository and not the filesystem root')
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Probe actual I/O rather than assuming a mount name implies access.
            with os.scandir(path):
                pass
            with tempfile.TemporaryFile(dir=path) as probe:
                probe.write(b'esi-storage-check')
                probe.flush()
                os.fsync(probe.fileno())
                probe.seek(0)
                if probe.read() != b'esi-storage-check':
                    raise OSError('Output read-back mismatch')
        except OSError as exc:
            raise ValueError(f'Output directory must be creatable, readable and writable: {path}: {exc}') from exc
    return path


REMOTE_PROBE = '''import os, sys
from pathlib import Path
p = Path(sys.argv[1])
if p.read_text() != sys.argv[2]:
    raise RuntimeError("Shared storage probe content differs")
with p.open("w") as f:
    f.write(sys.argv[3])
    f.flush()
    os.fsync(f.fileno())
'''


def verify_shared_storage(root, ssh_command, remote_python, *, timeout=60):
    """Require remote read/write visibility at the exact configured path."""
    root = validate_output_root(root)
    sent, reply = secrets.token_hex(32), secrets.token_hex(32)
    path = root/f'.esi-shared-probe-{secrets.token_hex(16)}'
    # Use normal output-file permissions (including umask/default ACLs), so
    # different local/remote users can use a group-writable shared directory.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(sent)
            stream.flush()
            os.fsync(stream.fileno())
        command = shlex.join([remote_python, '-c', REMOTE_PROBE, str(path), sent, reply])
        result = subprocess.run([*ssh_command, command], capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise ValueError(f'Shared storage probe failed: {result.stderr.strip()}')
        if path.read_text() != reply:
            raise ValueError('Remote write is not visible on the controller')
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise ValueError(f'Controller and simulator must read/write the same shared storage at '
                         f'the same absolute path ({root}): {exc}') from exc
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--source-root')
    parser.add_argument('--remote-python')
    parser.add_argument('--ssh-command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = validate_output_root(args.path, source_root=args.source_root)
    if args.ssh_command:
        if not args.remote_python:
            parser.error('--ssh-command requires --remote-python')
        verify_shared_storage(path, args.ssh_command, args.remote_python)


if __name__ == '__main__':
    main()
