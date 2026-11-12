"""Stream a step-aligned sensors video into a human review copy (no rollout edits).

Requires imageio, imageio-ffmpeg, numpy and Pillow. Only supports the native
recorder's initial-frame + one-frame-per-control-ACK layout; fails closed on
missing history, discontinuities or frame count mismatches. English labels avoid
requiring a CJK font; original subgoals are preserved in the optional index.
"""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import textwrap


VERSION = 'review_video.v3'
PAUSE_COLOR = '#ffffff'
FAILURE_COLOR = '#f05252'
COLORS = {'STUDENT': '#38c978', 'EDIT': '#f0ca40', 'SHORT': '#ff963c', 'LONG': '#ae80ff'}


def control_kind(row):
    response = row.get('response', {})
    mode = response.get('mode', row.get('source_detail', row.get('source')))
    if mode == 'student':
        return 'STUDENT'
    if mode == 'edit':
        return 'EDIT'
    if mode in ('eef', 'gpt_eef'):
        # Contract evidence, never a guessed distance/step-count threshold.
        return 'LONG' if response.get('motion') or row.get('motion_result') else 'SHORT'
    raise ValueError(f'Unknown action contract at decision {row.get("decision")}: {mode}')


def failure_details(row):
    """Classify execution receipts independently of how many steps were executed."""
    failures = []
    for key in ('motion_result', 'correction_result'):
        result = row.get(key) or {}
        status = result.get('status', '')
        failed = (status in {'self_collision', 'arrival_not_reached', 'via_unverified',
                  'step_budget', 'interrupted', 'terminal_or_interrupted'}
                  or status.endswith(('_failed', '_error')) or bool(result.get('error')))
        if failed:
            failures.append(str(status or 'failed') +
                            (': ' + str(result['error']) if result.get('error') else ''))
    return '; '.join(failures)


def frame_color(row, *, hold=False, failed=False):
    return FAILURE_COLOR if failed else PAUSE_COLOR if hold else COLORS[control_kind(row)]


def input_signature(episode, font):
    """Small receipt hashes plus video stat for resumable derived outputs."""
    source = episode / 'sim/sensors.mp4'
    stat = source.stat()
    return dict(video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
                **{name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
                   for name in ('controller/history.json', 'sim/summary.json')},
                font_sha256=hashlib.sha256(Path(font).read_bytes()).hexdigest())


def validate_history(history, total):
    tick, seen = 0, set()
    for row in history:
        decision = row['decision']
        start, end, count = row['start_tick'], row['end_tick'], row['executed_steps']
        if decision in seen or start != tick or end - start != count or count < 0:
            raise ValueError(f'Non-contiguous/invalid execution history at decision {decision}')
        seen.add(decision)
        control_kind(row)
        tick = end
    if tick != total:
        raise ValueError(f'History ends at {tick}, simulator ends at {total}')


def episode_path(path):
    path = path.resolve()
    if (path / 'controller/history.json').is_file():
        return path
    attempts = sorted(p for p in path.glob('attempt_*') if (p / 'controller/history.json').is_file())
    if len(attempts) != 1:
        raise ValueError('Specify an exact attempt directory when there are zero/multiple attempts')
    return attempts[0]


def destination(episode, root, case):
    parts = episode.parts
    if 'results' not in parts:
        raise ValueError('Expected archive beneath results/<experiment>; provide a normal archive path')
    experiment = parts[parts.index('results') + 1]
    identity = [p for p in parts if p.startswith(('replica_', 'repeat_', 'attempt_'))]
    name = case['case_id'] + '__' + '__'.join(identity) + '.mp4'
    return root / experiment / case['task'] / case['case_id'] / name


def render(args):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    episode = episode_path(args.episode)
    summary = json.loads((episode / 'sim/summary.json').read_text())
    history = json.loads((episode / 'controller/history.json').read_text())
    total = summary['step_id']
    validate_history(history, total)
    source = episode / 'sim/sensors.mp4'
    output = (Path(args.output) if getattr(args, 'output', None) is not None else
              destination(episode, args.output_root, summary['evaluation_case']))
    if output.resolve().is_relative_to(episode):
        raise ValueError('Review output must be outside the original episode')
    index_path = output.with_suffix('.index.json')
    if output.exists() or index_path.exists():
        raise FileExistsError(f'Refusing to replace existing review: {output}')
    with imageio.get_reader(str(source)) as probe:
        metadata = probe.get_meta_data()
        frames = probe.count_frames()
    if frames != total + 1 or summary['video_frames'] != frames:
        raise ValueError(f'Unverified frame mapping: {frames} frames, {total} steps')
    fps = metadata['fps']
    if abs(fps - 1 / summary['control_dt']) > 0.01:
        raise ValueError('Video FPS disagrees with native control period')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.mp4')
    temporary.touch(exist_ok=False)  # Exclusive reservation against concurrent renders.
    font_path = args.font
    font = ImageFont.truetype(str(font_path), 23)
    small = ImageFont.truetype(str(font_path), 20)
    w, h = metadata['size']
    border, header = 10, 116
    width, height = w + 2 * border, h + header + 2 * border
    width += width % 2
    height += height % 2
    written, entries = 0, []
    pending_frame = None
    zero_failure_ids = []

    def paint(frame, row, tick, banner='', *, hold=False, failed=False, detail=''):
        kind = control_kind(row) if row else 'STUDENT'
        color = frame_color(row, hold=hold, failed=failed) if row else PAUSE_COLOR
        canvas = Image.new('RGB', (width, height), color)
        canvas.paste(Image.fromarray(frame), (border, header + border))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((border, border, width-border-1, header-1), fill='#111824')
        label = ('PI0.5 / STUDENT' if kind == 'STUDENT' else 'GPT / ' + kind)
        title = (f'D{row["decision"]:03d} | {label} | Step {tick}/{total} | '
                 f'Executed {row["executed_steps"]} steps') if row else f'END | Step {tick}/{total}'
        headline = banner
        draw.text((24, 16), title + (' | ' + headline if headline else ''), font=font, fill=color)
        subgoal = row.get('assessment', row.get('response', {}).get('assessment', {})).get('current_subgoal', '') if row else ''
        if detail:
            subgoal = detail
        lines = textwrap.wrap(subgoal, width=max(20, (width-50)//12))[:2]
        for i, line in enumerate(lines):
            draw.text((24, 49 + i*26), line, font=small, fill='white')
        return np.asarray(canvas)

    def emit(frame, count=1):
        nonlocal written, pending_frame
        for _ in range(count):
            # Buffer one frame so a zero-step refusal can annotate its boundary
            # without adding time when pause=0.
            if pending_frame is not None:
                writer.append_data(pending_frame)
            pending_frame = frame
        written += count

    with imageio.get_reader(str(source)) as reader, imageio.get_writer(
            str(temporary), fps=fps, codec='libx264', pixelformat='yuv420p',
            macro_block_size=2, ffmpeg_log_level='error',
            output_params=['-crf', '20', '-preset', 'veryfast', '-threads', '2',
                           '-movflags', '+faststart']) as writer:
        iterator = iter(reader)
        frame = next(iterator)  # frame 0 is the observed state BEFORE the first action.
        emit(paint(frame, history[0] if history else None, 0, 'INITIAL STATE'))
        for row in history:
            failure = failure_details(row)
            entry = dict(decision=row['decision'], kind=control_kind(row),
                         start_step=row['start_tick'], end_step=row['end_tick'],
                         executed_steps=row['executed_steps'], failure=failure or None,
                         subgoal=row.get('assessment', {}).get('current_subgoal', ''),
                         output_start_frame=written, holds=[])

            def hold(banner, *, failed=False, detail=''):
                start = written
                count = round(args.pause * fps)
                emit(paint(frame, row, row['end_tick'] if failed else row['start_tick'],
                           banner, hold=True, failed=failed, detail=detail), count)
                if count:
                    entry['holds'].append(dict(start_frame=start, end_frame_exclusive=written,
                                               source_step=row['end_tick'] if failed else row['start_tick'],
                                               type='failure' if failed else 'review_pause',
                                               color=FAILURE_COLOR if failed else PAUSE_COLOR))

            if row['executed_steps'] == 0:
                # Each decision has its own feedback; never merge away decision IDs.
                if args.pause == 0 and failure:
                    zero_failure_ids.append(row['decision'])
                    pending_frame = paint(frame, row, row['end_tick'],
                        'FAILED - NO EXECUTION', failed=True,
                        detail=f'Zero-step failures {zero_failure_ids}: {failure}')
                    entry['failure_frame'] = written - 1
                else:
                    hold('FAILED - NO EXECUTION' if failure else 'NO EXECUTION',
                         failed=bool(failure), detail=failure or 'No control steps were executed.')
            else:
                zero_failure_ids.clear()
                if control_kind(row) != 'STUDENT':
                    hold('REVIEW PAUSE - BEFORE ACTION')
                entry['action_start_frame'] = written
                for tick in range(row['start_tick'] + 1, row['end_tick'] + 1):
                    frame = next(iterator)
                    last_failure = bool(failure and tick == row['end_tick'])
                    banner = ('FAILED - PARTIAL EXECUTION' if last_failure else
                              'NEW DECISION' if tick-row['start_tick'] <= max(1, round(fps*.25)) else '')
                    emit(paint(frame, row, tick, banner, failed=last_failure,
                               detail=failure if last_failure else ''))
                entry['action_end_frame_exclusive'] = written
                if failure:
                    hold('FAILED - PARTIAL EXECUTION', failed=True, detail=failure)
            entry['output_end_frame_exclusive'] = written
            entries.append(entry)
        if next(iterator, None) is not None:
            raise ValueError('Unexpected source frames remain')
        if pending_frame is not None:
            writer.append_data(pending_frame)
    temporary.rename(output)
    index = dict(version=VERSION, input_signature=input_signature(episode, args.font), source=str(source),episode=str(episode),output=str(output),fps=fps,
                 source_frames=frames,output_frames=written,control_steps=total,
                 mapping='Source frame k is state after control step k; frame 0 is reset. '
                         'Decision (start,end] owns the post-action frames. White holds repeat pre-action states; '
                         'red failure holds repeat the last executed state (or pre-action state for zero steps). '
                         'With pause=0 no frames are added; zero-step failures mark the existing boundary frame.',
                 pause_seconds=args.pause,decisions=entries,
                 decision_counts=dict(Counter(control_kind(r) for r in history)))
    if args.index:
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in index.items() if k != 'decisions'}, ensure_ascii=False, indent=2))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path, help='Exact attempt directory, or repeat containing one attempt')
    parser.add_argument('--output-root', type=Path, default=Path('/mnt/esi/review_videos'))
    parser.add_argument('--pause', type=float, default=.8, help='Extra pre-action hold per GPT decision (seconds)')
    parser.add_argument('--index', action='store_true', help='Write debug mapping JSON; off by default')
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    args = parser.parse_args()
    if not 0 <= args.pause <= 10:
        parser.error('--pause must be between 0 and 10 seconds')
    render(args)


if __name__ == '__main__':
    main()
