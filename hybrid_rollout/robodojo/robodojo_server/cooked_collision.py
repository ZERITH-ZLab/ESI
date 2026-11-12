"""CPU distances between frozen PhysX cooked convexes through external FCL.

This checks static robot geometry on the configured pair list. It does not
query runtime contacts, scene objects, or simulator contact offsets.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ASSET_DIR = Path(__file__).with_name('collision_assets')
FCL_VERSION = '0.7.0.10'
CONFIG_IDENTITY = 'curobo_yaml_without_install_paths_v1'


def config_identity(path):
    """Hash parsed configuration, normalizing only two installation locations.

    The actual runtime URDF/USD and cooked geometry are hashed independently.
    All other configuration values (including other asset references) remain
    part of the identity. This is not permission to change robot geometry.
    """
    config = yaml.safe_load(Path(path).read_text())
    try:
        kinematics = config['robot_cfg']['kinematics']
        for name in ('asset_root_path', 'urdf_path'):
            if not isinstance(kinematics[name], str) or not kinematics[name].strip():
                raise ValueError(f'Invalid collision configuration path: {name}')
            kinematics[name] = f'<{name}>'
        payload = json.dumps(config, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False)
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid cuRobo collision configuration structure') from exc
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def sha256(path):
    """Hash an existing file; missing or unreadable files fail initialization."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets(urdf, config, usd, asset_dir=ASSET_DIR):
    """Validate frozen geometry and its actual runtime source files."""
    if usd is None:
        raise ValueError('Actual articulation USD path is required')
    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir/'manifest.json').read_text())
    identity = manifest.get('config_identity')
    if identity is None:
        # Explicit backwards compatibility for previously frozen manifests.
        if sha256(config) != manifest['sha256']['config']:
            raise ValueError('Collision asset hash mismatch: config (legacy byte hash)')
    else:
        if identity['algorithm'] != CONFIG_IDENTITY:
            raise ValueError('Unsupported collision configuration identity algorithm')
        if config_identity(config) != identity['sha256']:
            raise ValueError('Collision asset hash mismatch: config semantics')
    for name, path in [('urdf', urdf), ('usd', usd),
                       ('cooked', asset_dir/'cooked_shapes_cpu_v6.json')]:
        actual = sha256(path)
        if actual != manifest['sha256'][name]:
            raise ValueError(f'Collision asset hash mismatch: {name}')
    for relative, expected in manifest['usd_dependencies'].items():
        path = Path(usd).parent/relative
        if sha256(path) != expected:
            raise ValueError(f'Collision USD dependency hash mismatch: {relative}')
    return manifest


class CookedCollision:
    """Load pinned convex pieces into CPU FCL and query all configured pairs."""

    def __init__(self, links, pairs, urdf, config, usd, asset_dir=ASSET_DIR):
        import fcl
        if fcl.__version__ != FCL_VERSION:
            raise ValueError(f'Expected python-fcl {FCL_VERSION}, got {fcl.__version__}')
        manifest = verify_assets(urdf, config, usd, asset_dir)
        self.fcl, self.pairs = fcl, list(pairs)
        if [list(pair) for pair in pairs] != manifest['pairs']:
            raise ValueError('Cooked and configured collision pairs differ')
        shapes = json.loads((Path(asset_dir)/'cooked_shapes_cpu_v6.json').read_text())
        if shapes['asset_sha256'] != manifest['sha256']['usd']:
            raise ValueError('Cooked source USD identity differs')
        records = {r['body'].rsplit('/', 1)[-1]: r for r in shapes['records']}
        base_world = self._matrix(records[manifest['base_link']]['body_world'])
        self.objects, self.vertices, self.geometries = {}, {}, {}
        self.distance_request = fcl.DistanceRequest(enable_nearest_points=False,
                                                   enable_signed_distance=False)
        self.collision_request = fcl.CollisionRequest(num_max_contacts=1,
                                                     enable_contact=False)
        for link, (fk, indices, _, _) in links.items():
            record = records[link]
            if not record['result'].endswith('RESULT_VALID'):
                raise ValueError(f'Invalid cooked result: {link}')
            if len(record['convexes']) != manifest['piece_counts'][link]:
                raise ValueError(f'Cooked piece count differs: {link}')
            body_world = self._matrix(record['body_world'])
            local = np.linalg.inv(body_world) @ self._matrix(record['world'])
            if np.max(np.abs(np.linalg.inv(base_world) @ body_world -
                             fk.matrix(np.zeros(len(indices))))) >= 2e-5:
                raise ValueError(f'Cooked USD and source FK frames differ: {link}')
            pieces, points, geometries = [], [], []
            for piece in record['convexes']:
                vertices = np.asarray(piece['vertices'], dtype=np.float64)
                if (vertices.ndim != 2 or vertices.shape[1] != 3 or
                        len(vertices) < 4 or not np.isfinite(vertices).all()):
                    raise ValueError(f'Invalid cooked vertices: {link}')
                vertices = vertices @ local[:3, :3].T + local[:3, 3]
                faces = []
                for polygon in piece['polygons']:
                    count, start = polygon['num_vertices'], polygon['index_base']
                    indices_piece = piece['indices'][start:start+count]
                    if (count < 3 or len(indices_piece) != count or
                            any(type(i) is not int or not 0 <= i < len(vertices)
                                for i in indices_piece)):
                        raise ValueError(f'Invalid cooked polygon: {link}')
                    faces.extend([count, *indices_piece])
                if not faces:
                    raise ValueError(f'Missing cooked polygons: {link}')
                convex = fcl.Convex(vertices, len(piece['polygons']), faces)
                pieces.append(fcl.CollisionObject(convex))
                geometries.append(convex)
                points.append(vertices)
            if not pieces:
                raise ValueError(f'Missing cooked pieces: {link}')
            self.objects[link] = pieces
            self.geometries[link] = geometries
            self.vertices[link] = np.concatenate(points)
        self.geometry = dict(model='PhysX_cooked_convexes_external_FCL',
            runtime_contact_truth=False, fcl_version=fcl.__version__,
            asset_manifest=manifest, cooked_sha256=manifest['sha256']['cooked'])
        self._controls()

    @staticmethod
    def _matrix(value):
        matrix = np.asarray(value, float).T
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9) or
                not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
            raise ValueError('Invalid cooked rigid transform')
        return matrix

    def _distance(self, first, second):
        fcl = self.fcl
        value = float(fcl.distance(first, second, self.distance_request,
                                   fcl.DistanceResult()))
        if not np.isfinite(value) or (value < 0 and value != -1.):
            raise ValueError(f'Invalid FCL distance: {value}')
        collision = fcl.CollisionResult()
        count = fcl.collide(first, second, self.collision_request, collision)
        if (type(collision.is_collision) not in (bool, np.bool_) or
                not isinstance(count, (int, np.integer)) or isinstance(count, bool) or
                count < 0):
            raise ValueError('Invalid FCL collision result')
        hit = bool(collision.is_collision)
        if (value < 0 and not hit) or (value > 0 and hit) or bool(count) != hit:
            raise ValueError('FCL distance/collision results are inconsistent')
        return 0. if hit else value

    def _controls(self):
        """Exercise overlap sentinel and separation on the compiled backend."""
        first = next(iter(self.objects.values()))[0]
        same = self.fcl.CollisionObject(next(iter(self.geometries.values()))[0])
        if self._distance(first, same) != 0.:
            raise ValueError('FCL identical-geometry positive control failed')
        same.setTransform(self.fcl.Transform(np.array([10., 0., 0.])))
        if self._distance(first, same) <= 9.:
            raise ValueError('FCL separated-geometry control failed')

    def query(self, transforms):
        """Return native minimum gap after updating poses from shared source FK."""
        if set(transforms) != set(self.objects):
            raise ValueError('Incomplete source FK transforms')
        for link, transform in transforms.items():
            # Validate through the same rigid-transform routine (stored USD is transposed).
            transform = self._matrix(np.asarray(transform).T)
            value = self.fcl.Transform(transform[:3, :3], transform[:3, 3])
            for obj in self.objects[link]:
                obj.setTransform(value)
        minimum = (float('inf'), None, None, None, None)
        for a, b in self.pairs:
            for ia, first in enumerate(self.objects[a]):
                for ib, second in enumerate(self.objects[b]):
                    gap = self._distance(first, second)
                    if gap < minimum[0]:
                        minimum = gap, a, b, ia, ib
        gap, a, b, ia, ib = minimum
        if not np.isfinite(gap):
            raise ValueError('No finite FCL clearance')
        return dict(gap_m=gap, link_pair=[a, b], convex_piece_indices=[ia, ib],
                    geometry=self.geometry)


if __name__ == '__main__':
    import argparse
    from .self_collision import RobotSelfCollision
    parser = argparse.ArgumentParser(description='CPU hybrid collision startup preflight')
    parser.add_argument('--asset-root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ASSET_DIR/'manifest.json').read_text())
    files = {name: args.asset_root/relative
             for name, relative in manifest['source_files'].items()}
    guard = RobotSelfCollision(files['urdf'], files['config'],
                              [f'joint{i}' for i in range(1, 7)], usd=files['usd'])
    print(json.dumps(dict(event='hybrid_collision_preflight',
                          geometry=guard.native.geometry,
                          sphere_reach_bound_m=guard.reach_bound,
                          native_reach_bound_m=guard.native.reach_bound)))
