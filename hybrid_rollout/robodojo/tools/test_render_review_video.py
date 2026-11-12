import pytest

from .render_review_video import control_kind, validate_history


def row(decision, start, end, mode='student', **extra):
    return dict(decision=decision, start_tick=start, end_tick=end,
                executed_steps=end-start, response=dict(mode=mode, **extra))


def test_contract_not_duration():
    assert control_kind(row(0, 0, 60, 'eef')) == 'SHORT'
    assert control_kind(row(0, 0, 1, 'eef', motion={'tool': 'robodojo_move'})) == 'LONG'
    assert control_kind(row(0, 0, 1, 'edit')) == 'EDIT'


def test_zero_execution_preserves_boundary():
    validate_history([row(0, 0, 5), row(1, 5, 5, 'eef'), row(2, 5, 8)], 8)


@pytest.mark.parametrize('history,total', [([row(0, 1, 5)], 5),
    ([row(0, 0, 5)], 6), ([row(0, 0, 5), row(0, 5, 8)], 8)])
def test_reject_bad_mapping(history, total):
    with pytest.raises(ValueError):
        validate_history(history, total)


def test_partial_failure_and_color_priority():
    from .render_review_video import failure_details, frame_color, COLORS, PAUSE_COLOR, FAILURE_COLOR
    item = row(1, 5, 8, 'eef', motion={'tool': 'robodojo_move'})
    item['motion_result'] = dict(status='planning_failed', error='IK unreachable')
    assert 'planning_failed' in failure_details(item)
    assert frame_color(item) == COLORS['LONG']
    assert frame_color(item, hold=True) == PAUSE_COLOR
    assert frame_color(item, hold=True, failed=True) == FAILURE_COLOR
    item['executed_steps'] = 0
    assert failure_details(item)
    item['motion_result'] = dict(status='waypoints_reached_not_grasp_verified')
    assert not failure_details(item)


@pytest.mark.parametrize('explicit_output', [False, True])
@pytest.mark.parametrize('pause', [0, .8])
def test_encoded_holds_and_partial_failure(tmp_path, explicit_output, pause):
    import json
    from types import SimpleNamespace
    import imageio.v2 as imageio
    import numpy as np
    from .render_review_video import render, VERSION
    episode = tmp_path/'results/experiment/task/replica_0/repeat_1/attempt_0'
    (episode/'controller').mkdir(parents=True)
    (episode/'sim').mkdir()
    history = [row(0, 0, 1), row(1, 1, 1, 'eef'), row(2, 1, 3, 'eef')]
    history[1]['correction_result'] = dict(status='self_collision', error='blocked')
    history[2]['motion_result'] = dict(status='planning_failed', error='unreachable')
    (episode/'controller/history.json').write_text(json.dumps(history))
    (episode/'sim/summary.json').write_text(json.dumps(dict(step_id=3, video_frames=4,
        control_dt=.04, evaluation_case=dict(case_id='case', task='task'))))
    with imageio.get_writer(str(episode/'sim/sensors.mp4'), fps=25, macro_block_size=2) as writer:
        for k in range(4):
            writer.append_data(np.full((64, 96, 3), k*50, dtype=np.uint8))
    args = SimpleNamespace(episode=episode, output_root=tmp_path/'reviews', pause=pause,
        font=__import__('pathlib').Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), index=True)
    if explicit_output:
        args.output = tmp_path/'reviews/base/standard/layout_0/repeat_001.mp4'
    index = render(args)
    if explicit_output:
        assert index['output'] == str(args.output)
    assert index['version'] == VERSION
    if pause == 0:
        assert index['output_frames'] == index['source_frames'] == 4
        assert all(not d['holds'] for d in index['decisions'])
        assert index['decisions'][1]['failure_frame'] == 1
        with imageio.get_reader(index['output']) as reader:
            assert reader.count_frames() == 4
            for k in (1, 3):
                frame = reader.get_data(k)
                assert np.max(np.abs(frame[140, 2].astype(int)-[240,82,82])) < 25
        return
    # 4 original frames + 20 zero-failure + 20 pre-action + 20 partial-failure holds.
    assert index['output_frames'] == 64
    assert len(index['decisions']) == 3
    zero, partial = index['decisions'][1:]
    assert zero['holds'][0]['source_step'] == 1
    assert [h['type'] for h in partial['holds']] == ['review_pause', 'failure']
    assert partial['holds'][1]['source_step'] == 3
    with imageio.get_reader(index['output']) as reader:
        assert reader.count_frames() == 64
        for hold, expected in [(zero['holds'][0], [240,82,82]),
                               (partial['holds'][0], [255,255,255]),
                               (partial['holds'][1], [240,82,82])]:
            frame = reader.get_data(hold['start_frame'])
            assert np.max(np.abs(frame[140, 2].astype(int)-expected)) < 25
        # Failure feedback repeats the final state; the pre-action hold repeats step 1.
        assert abs(int(reader.get_data(partial['holds'][0]['start_frame'])[150, 50, 0])-50) < 8
        assert abs(int(reader.get_data(partial['holds'][1]['start_frame'])[150, 50, 0])-150) < 8
