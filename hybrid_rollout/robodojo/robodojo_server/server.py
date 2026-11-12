"""Launch RoboDojo's native evaluation environment behind the existing RPC."""
import argparse
import importlib
import json
import os
import shutil
import traceback
from pathlib import Path


class NoPolicyConnection:
    """Native reset callback only; inference is owned by the Codex tool host."""
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('Native autonomous policy loop is disabled')

    def close(self):
        pass


def main():
    import cv2  # Load before Isaac's extension dependency paths.
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ROBODOJO_GRIPPER_CONTACT_ENABLED', 'ROBODOJO_PI_FORCE_HOLD_ENABLED'):
        if os.environ.get(name, '0') not in ('0', '1'):
            p.error(f'{name} must be 0 or 1')
    p.add_argument('--task', default='build_tower')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=19113)
    p.add_argument('--eval-seed', type=int, default=0, help='Published layout group, NOT the reset layout ID')
    p.add_argument('--eval-manifest', type=Path)
    p.add_argument('--case-file', type=Path)
    p.add_argument('--collection-step-limit', type=int, default=0,
                   help='Training collection only; must match the frozen collection panel')
    p.add_argument('--enable-scene-query', action='store_true',
                   help='Opt-in privileged scene query; renders but never steps physics')
    p.add_argument('--enable-motion-tools', action='store_true',
                   help='Opt-in continuous robot-only path with jerk-bounded S-curve timing')
    p.add_argument('--gripper-contact', action=argparse.BooleanOptionalAction,
                   default=os.environ.get('ROBODOJO_GRIPPER_CONTACT_ENABLED', '1') == '1',
                   help='Enable CPU PhysX contact windows; --no-gripper-contact skips all instrumentation')
    p.add_argument('--pi-force-hold', action=argparse.BooleanOptionalAction,
                   default=os.environ.get('ROBODOJO_PI_FORCE_HOLD_ENABLED', '0') == '1',
                   help='Experimental force-assisted pi chunk pause; independent and default off')
    p.add_argument('--pi-force-hold-config',
                   default=os.environ.get('ROBODOJO_PI_FORCE_HOLD_CONFIG', '{}'),
                   help='JSON experiment settings; thresholds and optional exact native actor allowlist')
    p.add_argument('--pi05-collision-shadow', action='store_true',
                   help='Record sampled pi05 sphere clearances after ACK; never gate or change actions')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(); args.headless = True; args.enable_cameras = True
    if args.pi_force_hold and not args.gripper_contact:
        p.error('--pi-force-hold requires --gripper-contact')
    from .pi_force_hold import configured as force_configured
    force_config = force_configured(json.loads(args.pi_force_hold_config)) if args.pi_force_hold else None
    if args.enable_motion_tools:
        from . import joint_motion  # Fail early on missing planner dependencies.
    from ..evaluation import read_panel, verify_assets, case_identity
    from ..io import require
    from env.global_configs import ROOT_DIR, BENCHMARK
    registry = importlib.import_module(f'task.{BENCHMARK}.task_registry')
    evaluation_case = None
    panel = None
    if args.eval_manifest:
        panel = read_panel(args.eval_manifest, os.environ.get('ROLLOUT_EVAL_MANIFEST_SHA256'))
        from ..regression_eval import SCHEMA as EVAL_SCHEMA
        if panel['schema'] == EVAL_SCHEMA:
            require(not args.enable_scene_query and args.collection_step_limit == 0,
                    'Eval forbids scene query/identity overlays and budget overrides')
        selected = json.loads(args.case_file.read_text())
        evaluation_case = selected['case']
        require(evaluation_case in panel['cases'] and selected['identity'] == case_identity(panel, evaluation_case),
                'Selected case differs from the frozen panel')
        require((args.task, args.eval_seed) == (evaluation_case['runtime_task'], evaluation_case['eval_seed']),
                'CLI task/eval_seed differs from selected case')
        verify_assets(panel, ROOT_DIR, [evaluation_case],
                      verify_source=os.environ.get('ROLLOUT_SKIP_NATIVE_SOURCE_VERIFY') != '1')
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(ROOT_DIR)/evaluation_case['layout']['path'], args.output/'scene_layout.json')
    # Match native main.py: monitor Articulation tasks before AppLauncher.
    import yaml
    task_config_path = registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)
    with open(task_config_path) as stream:
        task_values = yaml.safe_load(stream) or {}
    enable_monitor = bool(task_values.get('Articulation'))
    if enable_monitor:
        from src.eval_client.physx_warning_monitor import get_monitor
        get_monitor().start(enabled=True)
    if args.enable_scene_query:
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    env = session = None
    try:
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR, BENCHMARK
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        from .rpc import serve
        from .session import RoboDojoSession
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root/'arx_x5.yml'))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False,
            policy_name='Pi_05', additional_info='codex_hybrid', seed=args.eval_seed,
            physx_monitor_enabled=enable_monitor)
        values = {key: load_yaml(str(root/key/(evaluation['config'][key]+'.yml')))
                  for key in ('sim', 'scene', 'camera', 'robot')}
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name='Pi_05'),
            task_env=load_yaml(registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)))
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        cfg.sim.seed = [0]  # Native startup; the controller later resets the selected layout_id.
        # EEF corrections use bounded robot-only DLS, never object-aware cuRobo.
        for robot in cfg.robot.robots:
            robot.need_planner = False
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = NoPolicyConnection
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        if evaluation_case:
            resolved = env.seed_manager.seed_info[evaluation_case['layout_id']]['scene_layout']
            require(Path(resolved).resolve() == (Path(ROOT_DIR)/evaluation_case['layout']['path']).resolve(),
                    'Native SeedManager resolved a different layout')
        from ..io import write_json
        from .step_budget import apply
        step_budget = apply(env, args.collection_step_limit, panel)
        write_json(args.output/'step_budget.json', step_budget)
        write_json(args.output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        session = RoboDojoSession(env, args.output, args.task,
            evaluation_identity=selected['identity'] if evaluation_case else None,
            scene_query_enabled=args.enable_scene_query, motion_enabled=args.enable_motion_tools,
            step_budget=step_budget, pi05_collision_shadow=args.pi05_collision_shadow,
            gripper_contact_enabled=args.gripper_contact, pi_force_hold_enabled=args.pi_force_hold,
            pi_force_hold_config=force_config)
        serve(session, args.port)
    except BaseException:
        # SimulationApp.close may exit before Python prints an uncaught error.
        # Persist the original failure before native cleanup can obscure it.
        traceback.print_exc()
        raise
    finally:
        try:
            if session:
                session._write_summary('server_close')
        finally:
            try:
                if session:
                    session.close()
            finally:
                try:
                    if env:
                        env.close()
                finally:
                    app.close()


if __name__ == '__main__':
    main()
