"""Loopback fake services verify the runtime-only snapshot can launch an episode."""
from . import test_dual_deployment as fixtures


def test_runtime_snapshot_episode_lifecycle(tmp_path):
    fixtures.test_shell_launcher_lifecycle_with_fake_services(
        tmp_path, controller_exit=0, scene_query='0', motion_tools='1', runtime_snapshot=True)
