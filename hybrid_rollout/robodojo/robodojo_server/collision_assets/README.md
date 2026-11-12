# Pinned X5 collision geometry

`cooked_shapes_cpu_v6.json` contains PhysX cooked convex pieces from `ARX.usd`,
including 32 pieces for link3. `manifest.json` pins the cooked JSON, actual source
USD, URDF, cuRobo configuration, nine links and 22 configured link pairs.

Before starting a new server, install `requirements-motion.txt` with the actual
simulator interpreter using `pip install --no-deps -r requirements-motion.txt`.
This adds `python-fcl==0.7.0.10` without upgrading existing NumPy/SciPy. Startup
runs the CPU preflight and each reset initializes guards from the articulation's
actual `cfg.spawn.usd_path`; missing dependencies, hash/frame mismatches and
invalid controls refuse startup. Source snapshots include this directory.

The USD collision geometry is self-contained: CPU USD dependency inspection
found one layer and all 11 collision mesh points/prim stacks authored in ARX.usd.
Unresolved `OmniPBR.mdl` and `gltf/pbr.mdl` are material references and do not
provide collision geometry. The manifest records this distinction.

`validate_path` requires clearance strictly greater than 1 mm (a larger margin
may be requested). It first attempts a continuous configured-sphere interval
certificate. An unresolved sphere interval is certified using cooked distances
on both endpoints and a conservative motion bound derived from cooked vertices,
or is subdivided. Depth and query budgets refuse unresolved paths. Each checked
joint state uses one source FK shared by both models. `clearance_details` and
`scan_path` remain sphere diagnostics; sampled scans are not certificates.

This is static robot geometry on the configured pair list, not live PhysX
contacts, scene-object collision, inter-arm collision or task-success evidence.
FCL distances are unsigned; its -1 overlap sentinel requires a consistent
collision query and is converted to zero gap. Any invalid or inconsistent query
refuses the path. The unchanged sphere model is not proven to fully enclose the
cooked shapes, so retaining sphere-certified intervals does not establish a
complete cooked-geometry certificate for every accepted path.
