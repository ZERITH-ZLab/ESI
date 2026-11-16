"""Storage portability checks without GPUs, accounts or an SSH server."""
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from . import output_paths


def test_non_mnt_directory_and_preview(tmp_path):
    root = tmp_path/'user data'/'outputs'
    assert output_paths.validate_output_root(root, create=False) == root
    assert not root.exists()
    assert output_paths.validate_output_root(root) == root
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('value', ['', 'relative/output', '/'])
def test_reject_invalid_roots(value):
    with pytest.raises(ValueError):
        output_paths.validate_output_root(value)


def test_reject_source_and_symlink_into_source(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    alias = tmp_path/'alias'
    alias.symlink_to(source, target_is_directory=True)
    for path in (source, source/'outputs', alias/'outputs'):
        with pytest.raises(ValueError, match='outside the source'):
            output_paths.validate_output_root(path, source_root=source)
    assert list(source.iterdir()) == []


def test_file_or_unwritable_storage_rejected(tmp_path, monkeypatch):
    file = tmp_path/'file'
    file.write_text('keep')
    with pytest.raises(ValueError, match='readable and writable'):
        output_paths.validate_output_root(file)
    def denied(**kwargs):
        raise PermissionError('read-only storage')
    monkeypatch.setattr(output_paths.tempfile, 'TemporaryFile', denied)
    with pytest.raises(ValueError, match='read-only storage'):
        output_paths.validate_output_root(tmp_path/'outputs')
    assert file.read_text() == 'keep'


@pytest.mark.parametrize('mode', ['shared', 'separate', 'remote_error', 'timeout'])
def test_bidirectional_probe(tmp_path, monkeypatch, mode):
    root = tmp_path/"shared folder's outputs"
    real_run = subprocess.run
    def fake_ssh(args, **kwargs):
        assert args[:2] == ['ssh', 'user@simulator']
        assert kwargs['timeout'] == 60
        command = shlex.split(args[-1])
        if mode == 'timeout':
            raise subprocess.TimeoutExpired(args, 60)
        if mode == 'remote_error':
            return subprocess.CompletedProcess(args, 1, '', 'Permission denied')
        if mode == 'separate':
            # Same initial contents on a different filesystem must not pass.
            other = tmp_path/'other-storage'
            other.write_text(Path(command[3]).read_text())
            command[3] = str(other)
        return real_run(command, **kwargs)
    monkeypatch.setattr(output_paths.subprocess, 'run', fake_ssh)
    if mode == 'shared':
        output_paths.verify_shared_storage(root, ['ssh', 'user@simulator'], sys.executable)
    else:
        with pytest.raises(ValueError, match='same shared storage'):
            output_paths.verify_shared_storage(root, ['ssh', 'user@simulator'], sys.executable)
    assert list(root.iterdir()) == []


def test_storage_cli_non_mnt(tmp_path):
    root = tmp_path/'outputs'
    subprocess.run([sys.executable, '-m', 'hybrid_rollout.robodojo.output_paths',
                    str(root)], check=True)
    assert root.is_dir() and not list(root.iterdir())
