# Licensing scope and third-party notices

Original ESI contributions are licensed under the [Apache License 2.0](LICENSE). This does not replace the licenses or copyright notices for inherited code, benchmark assets, model weights, fonts or cited material. In files containing both ESI contributions and inherited material, the Apache license applies to the original ESI contributions; the inherited portions retain their upstream terms. Contribution acknowledgments are in [NOTICE](NOTICE).

## GPT-as-Policy

ESI builds on [GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy). Inherited source code, documentation and other upstream material retain their original licenses, including the MIT license for GPT-as-Policy code:

Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng

The complete original copyright and MIT permission notice is preserved in [third_party/gpt-as-policy/LICENSE](third_party/gpt-as-policy/LICENSE). It applies to retained upstream portions, including those in modified files; these portions are not relicensed by ESI's root license. Components distributed by upstream under a separate license retain that separate license.

## Other components and assets

- **RoboDojo, XPolicyLab, OpenPI, Isaac Sim/IsaacLab and cuRobo:** obtain external projects, assets and weights under their respective licenses. Dependency locations and versions are described in [the installation guide](docs/INSTALLATION.md).
- **Derived collision geometry:** `hybrid_rollout/robodojo/robodojo_server/collision_assets/` contains geometry derived from RoboDojo robot assets. Its manifest identifies source assets; their rights are not superseded by the root code license.
- **Task reference image:** `hybrid_rollout/robodojo/task_references/store_tools_in_toolbox/completed.png` depicts benchmark assets; their original rights remain applicable.
- **Noto Sans CJK:** the bundled font and SIL Open Font License are in `hybrid_rollout/assets/fonts/`. See the adjacent README for upstream attribution.

## External integration patches

`patches/robodojo/` contains integration diffs against RoboDojo commit
`08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d`. Context and upstream portions
remain under RoboDojo's license; see the included [license](patches/robodojo/LICENSE) and [upstream LICENSE](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d/LICENSE).
These patches do not include the external project or full assets.
