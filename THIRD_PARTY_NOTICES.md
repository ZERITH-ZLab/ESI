# Licensing scope and third-party notices

The root MIT license applies to code and documentation owned by Yu-Mool Shu and Lipxin Zheng. It does not relicense third-party components, benchmark assets, model weights, fonts or cited material.

- **RoboDojo, XPolicyLab, OpenPI, Isaac Sim/IsaacLab and cuRobo:** obtain external projects, assets and weights under their respective licenses. Dependency locations and versions are described in [the installation guide](docs/INSTALLATION.md).
- **Derived collision geometry:** `hybrid_rollout/robodojo/robodojo_server/collision_assets/` contains geometry derived from RoboDojo robot assets. Its manifest identifies source assets; their rights are not superseded by the root code license.
- **Task reference image:** `hybrid_rollout/robodojo/task_references/store_tools_in_toolbox/completed.png` depicts benchmark assets; their original rights remain applicable.
- **Noto Sans CJK:** the bundled font and SIL Open Font License are in `hybrid_rollout/assets/fonts/`. See the adjacent README for upstream attribution.

## External integration patches

`patches/robodojo/` contains integration diffs against RoboDojo commit
`08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d`. Context and upstream portions
remain under RoboDojo's license; see the included [license](patches/robodojo/LICENSE) and [upstream LICENSE](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d/LICENSE).
These patches do not include the external project or full assets.
