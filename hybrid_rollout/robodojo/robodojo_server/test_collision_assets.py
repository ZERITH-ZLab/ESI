"""Portable asset identity must still reject changes to collision semantics."""
import json
import shutil

import pytest
import yaml

from .cooked_collision import CONFIG_IDENTITY, config_identity, sha256, verify_assets


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / 'robot'
    root.mkdir()
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    for name in ('robot.urdf', 'robot.usd', 'mesh.usd'):
        (root / name).write_text(name)
    config = dict(robot_cfg=dict(kinematics=dict(
        asset_root_path=str(root), urdf_path=str(root / 'robot.urdf'),
        collision_spheres={'tip': [dict(center=[0., 0., 0.], radius=.03)]},
        self_collision_buffer={'tip': -.01},
        self_collision_ignore={}, load_meshes=False)), planner=dict(frame_bias=[0., 0., 0.]))
    (root / 'curobo.yml').write_text(yaml.safe_dump(config))
    (frozen / 'cooked_shapes_cpu_v6.json').write_text('{}')
    manifest = dict(sha256={k: sha256(p) for k, p in dict(
        urdf=root/'robot.urdf', usd=root/'robot.usd', config=root/'curobo.yml',
        cooked=frozen/'cooked_shapes_cpu_v6.json').items()},
        usd_dependencies={'mesh.usd': sha256(root/'mesh.usd')},
        config_identity=dict(algorithm=CONFIG_IDENTITY, sha256=config_identity(root/'curobo.yml')))
    (frozen/'manifest.json').write_text(json.dumps(manifest))
    return root, frozen


def verify(root, frozen):
    return verify_assets(root/'robot.urdf', root/'curobo.yml', root/'robot.usd', frozen)


def test_relocated_formatted_yaml_passes(assets, tmp_path):
    root, frozen = assets
    moved = tmp_path/'different installation'
    shutil.copytree(root, moved)
    cfg = yaml.safe_load((moved/'curobo.yml').read_text())
    cfg['robot_cfg']['kinematics'].update(asset_root_path=str(moved), urdf_path=str(moved/'robot.urdf'))
    (moved/'curobo.yml').write_text('# new installation\n'+yaml.safe_dump(cfg, sort_keys=False))
    assert sha256(root/'curobo.yml') != sha256(moved/'curobo.yml')
    assert verify(root, frozen) == verify(moved, frozen)


@pytest.mark.parametrize('field,value', [
    ('collision_spheres', {'tip': [dict(center=[0., 0., 0.], radius=.04)]}),
    ('self_collision_ignore', {'tip': ['base']}), ('load_meshes', True),
    ('external_asset_path', '/different/asset'), ('self_collision_buffer', {'tip': -.1}),
])
def test_semantic_changes_rejected(assets, field, value):
    root, frozen = assets
    cfg = yaml.safe_load((root/'curobo.yml').read_text())
    cfg['robot_cfg']['kinematics'][field] = value
    (root/'curobo.yml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='config semantics'):
        verify(root, frozen)


@pytest.mark.parametrize('name', ['robot.urdf', 'robot.usd', 'mesh.usd', 'cooked_shapes_cpu_v6.json'])
def test_geometry_changes_rejected(assets, name):
    root, frozen = assets
    target = frozen if name.startswith('cooked') else root
    (target/name).write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify(root, frozen)


def test_missing_geometry_rejected(assets):
    root, frozen = assets
    (root/'mesh.usd').unlink()
    with pytest.raises(FileNotFoundError):
        verify(root, frozen)


def test_malformed_path_rejected(assets):
    root, frozen = assets
    cfg = yaml.safe_load((root/'curobo.yml').read_text())
    cfg['robot_cfg']['kinematics']['urdf_path'] = None
    (root/'curobo.yml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='configuration path'):
        verify(root, frozen)


def test_unknown_identity_version_rejected(assets):
    root, frozen = assets
    m = json.loads((frozen/'manifest.json').read_text())
    m['config_identity']['algorithm'] = 'future'
    (frozen/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Unsupported'):
        verify(root, frozen)


def test_legacy_manifest_keeps_byte_check(assets):
    root, frozen = assets
    m = json.loads((frozen/'manifest.json').read_text())
    del m['config_identity']
    (frozen/'manifest.json').write_text(json.dumps(m))
    verify(root, frozen)
    with (root/'curobo.yml').open('a') as f:
        f.write('\n# different bytes\n')
    with pytest.raises(ValueError, match='legacy byte hash'):
        verify(root, frozen)
