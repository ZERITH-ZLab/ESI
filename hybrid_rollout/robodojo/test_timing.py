import json
import pytest
from .timing import Timing, summarize


def test_spans_survive_failure_and_summary_can_be_rebuilt(tmp_path):
    timer = Timing(tmp_path/'controller.jsonl')
    with timer.span('pi05'):
        pass
    with pytest.raises(ValueError):
        with timer.span('execute'):
            raise ValueError('original')
    result = summarize(tmp_path, lifecycle={'exit_code': 124})
    assert result['stages']['pi05']['count'] == 1
    assert result['stages']['execute']['errors'] == 1
    assert result['measurement_complete']
    assert summarize(tmp_path)['stages'] == result['stages']


def test_partial_line_and_open_span_are_not_zero_duration(tmp_path):
    timer = Timing(tmp_path/'controller.jsonl')
    timer.begin('teacher_to_execute', step_id=10)
    with timer.path.open('a') as stream:
        stream.write('{"event":')
    result = summarize(tmp_path)
    assert result['malformed_lines'] == 1
    assert not result['measurement_complete']
    assert result['open_spans'][0]['stage'] == 'teacher_to_execute'
    assert 'teacher_to_execute' not in result['stages']


def test_timing_write_failure_does_not_change_application_error(tmp_path, capsys):
    blocked = tmp_path/'file'
    blocked.write_text('not a directory')
    timer = Timing(blocked/'timing.jsonl')
    with pytest.raises(RuntimeError, match='application'):
        with timer.span('execute'):
            raise RuntimeError('application')
    assert 'Timing write failed' in capsys.readouterr().err


def test_interrupted_wait_and_modes_are_reported(tmp_path):
    timer = Timing(tmp_path/'controller.jsonl')
    token = timer.begin('teacher_to_execute')
    timer.end(token, mode='eef')
    timer.begin('teacher_to_infer')
    timer.close_open()
    result = summarize(tmp_path)
    assert result['stages']['teacher_to_execute:eef']['count'] == 1
    assert result['stages']['teacher_to_infer']['errors'] == 1
    assert not result['open_spans']


def test_missing_controller_log_is_not_complete_measurement(tmp_path):
    assert not summarize(tmp_path)['measurement_complete']
