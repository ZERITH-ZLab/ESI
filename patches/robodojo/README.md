# RoboDojo integration patches

These small patches apply to RoboDojo commit
`08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d`. They do not vendor RoboDojo.
Apply them to a **new installation**, not an active experiment's source tree.

- `asset-paths.patch`: resolve the selected `Assets` symlink and use that physical
  location when generating robot configuration paths. Required for the documented
  shared-storage installation recipe. It does not change asset geometry.
- `planner-diagnostics.patch`: attach native failure diagnostics to failed cuRobo
  planning results. Included for diagnostic parity; it does not change planning,
  collision thresholds, trajectories, task goals or scoring.

Check both patches before applying either:

```bash
# ESI_SOURCE and ROBODOJO_SOURCE are absolute paths to your installations.
test "$(git -C "$ROBODOJO_SOURCE" rev-parse HEAD)" = 08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d
git -C "$ROBODOJO_SOURCE" apply --check \
  "$ESI_SOURCE/patches/robodojo/asset-paths.patch" \
  "$ESI_SOURCE/patches/robodojo/planner-diagnostics.patch"
git -C "$ROBODOJO_SOURCE" apply \
  "$ESI_SOURCE/patches/robodojo/asset-paths.patch" \
  "$ESI_SOURCE/patches/robodojo/planner-diagnostics.patch"
```

`git apply` intentionally fails on incompatible or already patched files. To check
an existing installation without modifying it, use `git apply --reverse --check`
with the same two paths. Do not repeatedly apply or force patches. Keep the diff
as part of the environment identity; resume must use the same patched source.

The patch context comes from RoboDojo and retains its upstream license. See
the included [RoboDojo license](LICENSE); ESI's license does not relicense upstream source.
