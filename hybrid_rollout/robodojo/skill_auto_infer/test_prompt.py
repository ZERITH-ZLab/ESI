from pathlib import Path

from . import run
from .workspace import prepare_workspace
from ..skill import run as baseline


def test_full_contract_injected_once_without_exploration_requirements(tmp_path):
    prompt = run.build_prompt(run.SKILL_ROOT, tmp_path)
    gate = (run.SKILL_ROOT/'gate_prompt.md').read_text()
    robot = (run.SKILL_ROOT/'context/eef_control.md').read_text()
    assert gate == (baseline.SKILL_ROOT/'gate_prompt.md').read_text()
    assert prompt.count(gate) == 1
    assert prompt.count(robot) == 1
    for required in ('same episode and step', 'without cropping',
                     'specific detail', 'indices 0..N-1', 'norm tolerance 1e-4',
                     'No routine notes', 'task_progress', '0.05 m', '0.35 rad'):
        assert required in prompt
    for removed in ('RoboLab', 'not a JSON-only reviewer',
                    'Maintain confirmed progress and geometry estimates in `NOTES.md`'):
        assert removed not in prompt


def test_workspace_does_not_reintroduce_old_role(tmp_path):
    audit = tmp_path/'audit'; audit.mkdir()
    agent = prepare_workspace(audit, run.SKILL_ROOT)
    instructions = (agent/'AGENTS.md').read_text()
    assert 'online decision module' in instructions
    assert 'all supplied in full' in instructions
    assert 'No routine notes' in instructions
    assert 'Read context/' not in instructions
    assert 'Optional' in (agent/'NOTES.md').read_text()


def test_tools_preserve_assessment_with_mode_specific_schema():
    assert [s['name'] for s in run.tool_specs()] == ['robodojo_start', 'robodojo_execute']
    schema = run.tool_specs()[1]['inputSchema']
    assert schema['required'] == ['response']
    for variant in schema['properties']['response']['anyOf']:
        assert variant['properties']['assessment'] == baseline.response_schema()['properties']['assessment']
    config = run.agent_config(Path('/audit'), Path('/agent'))
    assert config['features.shell_tool'] is True
    assert config['features.view_image'] is True
