"""Best-effort, append-only controller timings; never model-visible content."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid


class Timing:
    def __init__(self, path):
        self.path = Path(path)
        self.writer = uuid.uuid4().hex
        self.open = {}

    def emit(self, event, **fields):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as stream:
                stream.write(json.dumps(dict(event=event, writer=self.writer,
                    mono=time.monotonic(), utc=datetime.now(timezone.utc).isoformat(),
                    **fields))+'\n')
                stream.flush()
        except Exception as error:
            print(f'Timing write failed: {error}', file=sys.stderr)

    def begin(self, stage, **fields):
        token = (uuid.uuid4().hex, stage, time.monotonic())
        self.open[token[0]] = token
        self.emit('start', span_id=token[0], stage=stage, **fields)
        return token

    def end(self, token, status='ok', **fields):
        self.open.pop(token[0], None)
        self.emit('end', span_id=token[0], stage=token[1],
                  duration_s=time.monotonic()-token[2], status=status, **fields)

    def close_open(self, status='interrupted'):
        for token in list(self.open.values()):
            self.end(token, status=status)

    @contextmanager
    def span(self, stage, **fields):
        token = self.begin(stage, **fields)
        try:
            yield
        except BaseException as error:
            self.end(token, 'error', error_type=type(error).__name__)
            raise
        else:
            self.end(token)


def summarize(directory, **lifecycle):
    """Rebuild from completed spans; unfinished spans are unknown, never zero.

    Native activities and RPC spans nest within teacher/service intervals.
    Their totals must not be added together. No private text is copied.
    """
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        pending, stages = {}, {}
        malformed = 0
        records = 0
        for path in sorted(directory.glob('*.jsonl')):
            with path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        malformed += 1
                        continue
                    records += 1
                    key = (event.get('writer'), event.get('span_id'))
                    if event.get('event') == 'start':
                        pending[key] = event
                    elif event.get('event') == 'end':
                        start = pending.pop(key, None)
                        stage = event['stage']
                        mode = event.get('mode')
                        if mode:
                            stage += ':'+mode
                        stats = stages.setdefault(stage, dict(count=0, total_s=0., max_s=0., errors=0))
                        duration = event['duration_s']
                        stats['count'] += 1
                        stats['total_s'] += duration
                        stats['max_s'] = max(stats['max_s'], duration)
                        stats['errors'] += event.get('status') != 'ok'
                        if start is None:
                            malformed += 1
        result = dict(schema='robodojo.timing.v1', stages=stages,
            open_spans=list(pending.values()), malformed_lines=malformed,
            measurement_complete=bool(records) and not pending and not malformed,
            note='Controller-observed wall times; nested stages overlap. Open spans have unknown duration.',
            **lifecycle)
        # Only one summarizer should run after the controller is stopped.
        from .io import write_json
        write_json(directory/'summary.json', result)
        return result
    except Exception as error:
        print(f'Timing summary failed: {error}', file=sys.stderr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    existing = args.directory/'summary.json'
    metadata = {}
    if existing.exists():
        metadata = json.loads(existing.read_text()).get('lifecycle', {})
    result = summarize(args.directory, lifecycle=metadata)
    if result is not None:
        print(json.dumps(result, indent=2))
