"""Robot-only self-collision checks for commanded joint paths.

Uses unchanged configured spheres first, then pinned PhysX cooked convexes
through external CPU FCL. No scene queries, physics or actuator steps occur.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .kinematics import ArmFK


class SelfCollisionError(ValueError):
    """A commanded path cannot be certified clear of robot self-collision."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class RobotSelfCollision:
    """Checks one arm using URDF frames and its existing collision geometry."""

    def __init__(self, urdf, config, joint_names, base='base_link', *, usd=None,
                 asset_dir=None):
        try:
            self._initialize(urdf, config, joint_names, base, usd=usd,
                             asset_dir=asset_dir)
        except SelfCollisionError:
            raise
        except Exception as error:
            raise SelfCollisionError('Collision geometry initialization failed',
                dict(reason='backend_initialization', error_type=type(error).__name__,
                     error=str(error))) from error

    def _initialize(self, urdf, config, joint_names, base, *, usd, asset_dir):
        self.geometry = dict(urdf=str(Path(urdf).resolve()), config=str(Path(config).resolve()),
            urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
            units='meters', sphere_buffers='positive_only_unchanged',
            scope='configured_robot_spheres_not_PhysX_contact_truth')
        self.names = list(joint_names)
        cfg = yaml.safe_load(Path(config).read_text())['robot_cfg']['kinematics']
        if cfg['base_link'] != base:
            raise ValueError('Collision and FK base frames differ')
        edges = {j.find('child').get('link'): j
                 for j in ET.parse(urdf).getroot().findall('joint')}
        self.links = {}
        reach = []
        for link in cfg['collision_link_names']:
            spheres = cfg['collision_spheres'][link]
            if not spheres:
                raise ValueError(f'Missing collision spheres for {link}')
            centers = np.asarray([s['center'] for s in spheres], float)
            radii = np.asarray([s['radius'] for s in spheres], float)
            if (centers.shape != (len(radii), 3) or
                    not np.isfinite(centers).all() or
                    not np.isfinite(radii).all() or np.any(radii <= 0)):
                raise ValueError(f'Invalid collision spheres for {link}')
            # Preserve positive buffers; negative native buffers must not hide
            # the collision that the simulator actually enforces.
            radii += max(0., float(cfg.get('collision_sphere_buffer', 0.)))
            radii += max(0., float(cfg.get('self_collision_buffer', {}).get(link, 0.)))
            moving = []
            cursor = link
            length = float(np.max(np.linalg.norm(centers, axis=1)))
            while cursor != base:
                joint = edges[cursor]
                origin = joint.find('origin')
                if origin is not None:
                    length += np.linalg.norm(np.fromstring(origin.get('xyz', '0 0 0'), sep=' '))
                if joint.get('type') != 'fixed':
                    axis = joint.find('axis')
                    axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0, 0])
                    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-8):
                        raise ValueError('Collision displacement bound requires unit joint axes')
                    moving.append(joint.get('name'))
                cursor = joint.find('parent').get('link')
            if not set(moving) <= set(self.names):
                raise ValueError(f'Collision frame {link} requires non-arm joints')
            names = [n for n in self.names if n in moving]
            fk = ArmFK(urdf, names, base, link)
            self.links[link] = (fk, [self.names.index(n) for n in names], centers, radii)
            reach.append(length)
        ignored = cfg.get('self_collision_ignore', {})
        names = list(self.links)
        self.pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                      if b not in ignored.get(a, []) and a not in ignored.get(b, [])]
        if not self.pairs:
            raise ValueError('No self-collision pairs configured')
        # Sum of link offsets bounds each sphere's distance from any upstream
        # revolute axis, hence bounds displacement between sampled joint states.
        self.reach_bound = float(max(reach))
        if not np.isfinite(self.reach_bound) or self.reach_bound <= 0:
            raise ValueError('Invalid sphere displacement bound')

        # Both representations use this exact per-link source FK. The native
        # displacement radius is derived from cooked vertices, never spheres.
        from .cooked_collision import ASSET_DIR, CookedCollision
        try:
            self.native = CookedCollision(self.links, self.pairs, urdf, config,
                                          usd, asset_dir or ASSET_DIR)
            native_reach = []
            for link, vertices in self.native.vertices.items():
                radius = float(np.max(np.linalg.norm(vertices, axis=1)))
                cursor = link
                while cursor != base:
                    joint = edges[cursor]
                    origin = joint.find('origin')
                    if origin is not None:
                        radius += np.linalg.norm(np.fromstring(
                            origin.get('xyz', '0 0 0'), sep=' '))
                    cursor = joint.find('parent').get('link')
                native_reach.append(radius)
            self.native.reach_bound = float(max(native_reach))
            if not np.isfinite(self.native.reach_bound) or self.native.reach_bound <= 0:
                raise ValueError('Invalid cooked displacement bound')
        except Exception as error:
            raise SelfCollisionError('Cooked collision initialization failed',
                dict(reason='backend_initialization', error_type=type(error).__name__,
                     error=str(error))) from error

    def _transforms(self, joints):
        q = np.asarray(joints, float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError('Invalid self-collision joint vector')
        return {link: fk.matrix(q[indices])
                for link, (fk, indices, _, _) in self.links.items()}

    def _sphere_details(self, q, transforms):
        world = {link: (centers @ transforms[link][:3, :3].T +
                        transforms[link][:3, 3], radii)
                 for link, (_, _, centers, radii) in self.links.items()}
        distances = []
        for a, b in self.pairs:
            ca, ra = world[a]
            cb, rb = world[b]
            gaps = np.linalg.norm(ca[:, None] - cb[None, :], axis=-1)
            gaps -= ra[:, None] + rb[None, :]
            if not np.isfinite(gaps).all():
                raise SelfCollisionError('Invalid sphere distance',
                                         dict(reason='invalid_query'))
            ia, ib = np.unravel_index(np.argmin(gaps), gaps.shape)
            distances.append((float(gaps[ia, ib]), a, b, int(ia), int(ib)))
        gap, a, b, ia, ib = min(distances)
        return dict(gap_m=gap, link_pair=[a, b], sphere_indices=[ia, ib],
                    q=q.tolist(), geometry=self.geometry, backend='sphere')

    def _native_details(self, q, transforms):
        try:
            witness = self.native.query(transforms)
            if not np.isfinite(witness['gap_m']) or witness['gap_m'] < 0:
                raise ValueError('Invalid native clearance')
            return dict(witness, q=q.tolist(), backend='cooked_fcl')
        except Exception as error:
            raise SelfCollisionError('Cooked collision query failed',
                dict(reason='invalid_native_query', q=q.tolist(),
                     error_type=type(error).__name__, error=str(error))) from error

    def native_clearance_details(self, joints):
        """Return cooked geometry gap for a pose, using the source FK."""
        q = np.asarray(joints, float)
        return self._native_details(q, self._transforms(q))

    def hybrid_clearance_details(self, joints, margin_m=.001):
        """Sphere pose check followed by cooked review when needed."""
        self._margin(margin_m)
        q = np.asarray(joints, float)
        transforms = self._transforms(q)
        sphere = self._sphere_details(q, transforms)
        if sphere['gap_m'] > margin_m:
            return sphere
        return dict(self._native_details(q, transforms), sphere_gap_m=sphere['gap_m'])

    @staticmethod
    def _margin(margin_m):
        if not np.isfinite(margin_m) or margin_m < .001:
            raise ValueError('Self-collision clearance requires a margin >= 0.001 m')

    def clearance(self, joints):
        """Returns minimum sphere separation and the corresponding link pair."""
        witness = self.clearance_details(joints)
        return witness['gap_m'], *witness['link_pair']

    def clearance_details(self, joints):
        """Returns the sampled minimum and its unchanged geometry witness."""
        q = np.asarray(joints, float)
        return self._sphere_details(q, self._transforms(q))

    def validate_path(self, joints, margin_m=.001, *, max_depth=20,
                      max_checks=100000):
        """Certify every continuous linear joint chord with separate bounds.

        A sphere interval certificate uses sphere gaps on both endpoints. If
        unavailable, cooked intervals use cooked gaps on both endpoints and a
        reach bound from cooked vertices. Sampling alone never certifies an
        interval. Any unresolved interval or backend failure refuses the path.
        """
        qs = np.asarray(joints, float)
        self._margin(margin_m)
        if (qs.ndim != 2 or qs.shape[1] != len(self.names) or len(qs) == 0
                or not np.isfinite(qs).all()):
            raise ValueError('Expected a finite nonempty joint path')
        if (type(max_depth) is not int or not 0 <= max_depth <= 20 or
                type(max_checks) is not int or not 1 <= max_checks <= 100000):
            raise ValueError('Invalid collision certification resource budget')
        checks, native_checks = 0, 0
        sphere_intervals, native_intervals = 0, 0
        minimum = float('inf')

        def measure(q, location):
            nonlocal checks, minimum
            if checks >= max_checks:
                raise SelfCollisionError('Collision certification budget exhausted',
                    dict(reason='resource_budget', checked_states=checks,
                         max_checks=max_checks, location=location))
            transforms = self._transforms(q)
            witness = self._sphere_details(q, transforms)
            checks += 1
            state = dict(q=q, transforms=transforms, sphere=witness,
                         native=None, location=location)
            accepted = witness if witness['gap_m'] > margin_m else native(state)
            minimum = min(minimum, accepted['gap_m'])
            return state

        def native(state):
            nonlocal native_checks, minimum
            if state['native'] is None:
                state['native'] = self._native_details(state['q'], state['transforms'])
                native_checks += 1
            witness = state['native']
            if witness['gap_m'] <= margin_m:
                raise SelfCollisionError(
                    f"Self-collision cooked clearance {witness['gap_m']:.6f} m; "
                    f'required > {margin_m:.6f} m',
                    dict(witness, sphere_gap_m=state['sphere']['gap_m'],
                         margin_m=margin_m, location=state['location'],
                         checked_states=checks, reason='sampled_clearance'))
            minimum = min(minimum, witness['gap_m'])
            return witness

        previous = measure(qs[0], dict(kind='path_sample', sample_index=0))
        for chord, qb in enumerate(qs[1:]):
            following = measure(qb, dict(kind='path_sample', sample_index=chord+1))
            stack = [(previous, following, 0, 0., 1.)]
            while stack:
                a, b, depth, lo, hi = stack.pop()
                motion = float(np.sum(np.abs(b['q']-a['q'])))
                sphere_bound = self.reach_bound*motion
                if min(a['sphere']['gap_m'], b['sphere']['gap_m']) - sphere_bound > margin_m:
                    sphere_intervals += 1
                    continue
                da, db = native(a)['gap_m'], native(b)['gap_m']
                native_bound = self.native.reach_bound*motion
                if min(da, db) - native_bound > margin_m:
                    native_intervals += 1
                    continue
                if depth >= max_depth:
                    witness = a['native'] if da <= db else b['native']
                    raise SelfCollisionError('Cannot certify interpolated self-collision clearance',
                        dict(witness, margin_m=margin_m, checked_states=checks,
                             reason='unresolved_interval',
                             displacement_bound_m=native_bound,
                             sphere_displacement_bound_m=sphere_bound,
                             location=dict(kind='joint_chord', chord_index=chord,
                                           fraction_interval=[lo, hi], depth=depth),
                             interval_joints=[a['q'].tolist(), b['q'].tolist()]))
                fraction = (lo+hi)/2
                mid = measure((a['q']+b['q'])/2,
                    dict(kind='joint_chord', chord_index=chord,
                         fraction=fraction, depth=depth+1))
                stack.extend([(a, mid, depth+1, lo, fraction),
                              (mid, b, depth+1, fraction, hi)])
            previous = following
        return dict(minimum_sampled_clearance_m=minimum, margin_m=margin_m,
                    checked_states=checks, native_checked_states=native_checks,
                    sphere_certified_intervals=sphere_intervals,
                    native_certified_intervals=native_intervals,
                    continuous_joint_chords_certified=True,
                    geometry='configured_spheres_then_PhysX_cooked_convex_FCL',
                    runtime_contact_truth=False, native_geometry=self.native.geometry)




def check_with_diagnostics(kinematics, arm, joints, report, stage, candidate_path,
                           **position):
    """Preserve the original gate and attach its structured first-hit receipt."""
    try:
        return kinematics.check_self_collision(arm, joints)
    except SelfCollisionError as error:
        report['collision_failure'] = dict(stage=stage, arm=arm,
            candidate_plan_npz=str(candidate_path), **position,
            witness=error.details)
        raise


def scan_path(guard, joints, margin_m=.001, subdivisions=1):
    """Scan all requested samples, continuing beyond low-clearance samples.

    Linear chord subdivisions are samples only. This is not the adaptive
    certificate from validate_path and does not prove continuous clearance.
    """
    qs = np.asarray(joints, float)
    if (qs.ndim != 2 or qs.shape[1] != len(guard.names) or len(qs) == 0
            or not np.isfinite(qs).all() or type(subdivisions) is not int
            or subdivisions < 1 or not np.isfinite(margin_m) or margin_m < 0):
        raise ValueError('Expected a finite path, nonnegative margin and positive subdivisions')
    samples = [dict(sample_index=0, path_position=0., **guard.clearance_details(qs[0]))]
    for chord, (qa, qb) in enumerate(zip(qs[:-1], qs[1:])):
        for index in range(1, subdivisions + 1):
            fraction = index / subdivisions
            samples.append(dict(sample_index=len(samples), path_position=chord + fraction,
                **guard.clearance_details((1 - fraction)*qa + fraction*qb)))
    intervals = []
    for row in samples:
        if row['gap_m'] <= margin_m:
            if not intervals or intervals[-1]['last_sample'] != row['sample_index'] - 1:
                intervals.append(dict(first_sample=row['sample_index'], last_sample=row['sample_index'],
                                      path_position_interval=[row['path_position'], row['path_position']]))
            else:
                intervals[-1]['last_sample'] = row['sample_index']
                intervals[-1]['path_position_interval'][1] = row['path_position']
    return dict(minimum_sampled=min(samples, key=lambda row: row['gap_m']),
                margin_m=margin_m, threshold_sample_intervals=intervals, samples=samples,
                checked_states=len(samples), subdivisions=subdivisions,
                continuous_clearance_proven=False, scope='sampled_configured_spheres_only')
