"""Read-only RGB + instruction viewer. No MCAP, FK or archived-code execution."""
import argparse
from collections import Counter, OrderedDict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from hybrid_rollout.robodojo.tools.render_review_video import control_kind, failure_details, validate_history

HERE = Path(__file__).parent
ROOT = Path('/mnt/esi')
DEFAULTS = [ROOT/'results']


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def probe_video(path):
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=nb_frames,avg_frame_rate,width,height', '-of', 'json', str(path)]
    streams = json.loads(subprocess.check_output(args, text=True, timeout=30))['streams']
    if len(streams) != 1:
        raise ValueError('视频流缺失')
    stream = streams[0]
    # Native recorder uses MP4 with frame count metadata; do not guess from duration.
    if not str(stream.get('nb_frames', '')).isdigit():
        raise ValueError('视频缺少帧数，无法核验逐步映射')
    return dict(frames=int(stream['nb_frames']), fps=float(Fraction(stream['avg_frame_rate'])),
                width=stream['width'], height=stream['height'])


def byte_range(value, size):
    if size <= 0:
        raise ValueError('Empty video')
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
    if not match or not any(match.groups()):
        raise ValueError('Invalid range')
    left, right = match.groups()
    if not left:
        count = int(right)
        if count <= 0:
            raise ValueError('Invalid suffix')
        start, end = max(0, size - count), size - 1
    else:
        start, end = int(left), min(int(right), size - 1) if right else size - 1
    if start >= size or end < start:
        raise ValueError('Unsatisfiable range')
    return start, end, True


class Store:
    def __init__(self, roots, defaults):
        self.roots = [Path(r).resolve() for r in roots]
        self.defaults = [str(self.permitted(p)) for p in defaults]
        self.entries, self.cache = {}, OrderedDict()
        self.lock = threading.RLock()

    def permitted(self, path):
        path = Path(path).expanduser().resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError('路径超出允许的数据根目录')
        return path

    def file(self, archive, relative):
        return self.permitted(Path(archive)/relative)

    def scan(self, paths):
        found, issues = {}, []
        prune = {'source_snapshot','observations','proposals','codex_workspace','debug_video',
                 'timing','logs','controller','sim','node_modules','codex_runtime','prepared','attempts'}
        for raw in paths:
            root = self.permitted(raw)
            if not root.is_dir():
                issues.append(f'目录不存在：{root}')
                continue
            for directory, dirs, files in os.walk(root, followlinks=False):
                archive = Path(directory)
                if 'annotation.json' in files or ('status.json' in files and archive.parent.name == 'episodes'):
                    dirs[:] = []
                    try:
                        from .annotations import entry_from_annotation
                        entry = entry_from_annotation(self, archive)
                        found[entry['id']] = entry
                    except (ValueError, OSError, KeyError) as error:
                        issues.append(f'{archive}: {error}')
                    continue
                if (archive/'sim/summary.json').is_file():
                    dirs[:] = []
                    try:
                        archive = self.permitted(archive)
                        summary = read(self.file(archive, 'sim/summary.json'))
                        if not self.file(archive, 'controller/history.json').is_file():
                            continue
                        case = summary.get('evaluation_case') or {}
                        parts = archive.parts
                        experiment = parts[parts.index('results')+1] if 'results' in parts else root.name
                        key = hashlib.sha256(str(archive).encode()).hexdigest()[:24]
                        repeat = case.get('policy_rng_seed')
                        found[key] = dict(id=key, experiment=experiment, archive=str(archive),
                            variant=case.get('variant','unknown'), layout=case.get('layout_id'),
                            repeat=repeat+1 if isinstance(repeat,int) else None,
                            attempt=archive.name, success=summary.get('success'),
                            complete=bool(summary.get('terminated') or summary.get('truncated')),
                            steps=summary.get('step_id'))
                    except (ValueError, OSError, KeyError) as error:
                        issues.append(f'{archive}: {error}')
                    continue
                dirs[:] = sorted(d for d in dirs if d not in prune and not d.startswith(('.', '_')))
        entries = sorted(found.values(), key=lambda e:(e['experiment'],e['variant'],
                         e['layout'] if isinstance(e['layout'],int) else -1,e['repeat'] or 0,e['attempt']))
        with self.lock:
            self.entries.update(found)
        return dict(episodes=entries, issues=issues)

    def entry(self, key):
        with self.lock:
            if key not in self.entries:
                raise KeyError('请先扫描该 episode')
            return self.entries[key]

    def episode(self, key):
        entry = self.entry(key)
        archive = entry['archive']
        names = ['controller/history.json','controller/controller_protocol.json', 'controller/result.json',
                 'controller/codex_workspace/worker.json','controller/request_000.json',
                 'sim/summary.json','sim/sensors.mp4']
        paths = [self.file(archive, name) for name in names]
        if entry.get('annotation'):
            paths.append(self.permitted(entry['annotation']))
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else (str(p),None) for p in paths)
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached[0] == signature:
                self.cache.move_to_end(key)
                return cached[1]
        data = self.build(entry)
        if entry.get('annotation'):
            from .annotations import attach_annotation
            data = attach_annotation(self, data)
        with self.lock:
            self.cache[key] = signature, data
            self.cache.move_to_end(key)
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        return data

    def build(self, entry):
        archive = entry['archive']
        summary = read(self.file(archive,'sim/summary.json'))
        history = read(self.file(archive,'controller/history.json'))
        protocol = read(self.file(archive,'controller/controller_protocol.json'), {})
        worker = read(self.file(archive,'controller/codex_workspace/worker.json'), {})
        result = read(self.file(archive,'controller/result.json'), {})
        total, dt = summary['step_id'], float(summary['control_dt'])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('无效的 control_dt')
        validate_history(history, total)
        video = probe_video(self.file(archive,'sim/sensors.mp4'))
        if video['frames'] != total+1 or summary['video_frames'] != total+1 or abs(video['fps']-1/dt) > .01:
            raise ValueError('视频帧数/FPS与控制步不一致，拒绝显示错位标签')
        guided = protocol.get('controller_route') == 'subtask_feedforward_v1' or any('conditioning' in h for h in history)
        decisions, issues = [], []
        for h in history:
            c = h.get('conditioning')
            state = 'legacy' if not guided else 'missing'
            if isinstance(c,dict) and 'subtask' in c and 'effective_prompt' in c:
                state = 'null' if c['subtask'] is None else 'subtask'
                expected = c.get('episode_instruction','')
                if state == 'subtask':
                    expected += '\nCurrent subtask: ' + str(c.get('subtask_text','')).strip()
                if expected != c.get('effective_prompt'):
                    issues.append(f'D{h["decision"]}: 保存的完整 prompt 与 subtask 文本不一致')
                if c.get('prediction_id') != h.get('prediction_id') or c.get('step_id') != h['start_tick'] or c.get('episode_id') != summary['episode_id']:
                    issues.append(f'D{h["decision"]}: conditioning 身份/步数不一致')
            if state == 'missing':
                issues.append(f'D{h["decision"]}: 前馈路线缺少完整 conditioning，不能视为 null')
            subtask = c.get('subtask') if isinstance(c,dict) else None
            free_text = (isinstance(c, dict) and c.get('subtask_format') == 'free_text_v1'
                         and isinstance(subtask, str) and subtask.strip() == c.get('subtask_text'))
            if state == 'subtask' and ((not isinstance(subtask,dict) and not free_text) or not c.get('subtask_text')):
                raise ValueError(f'D{h["decision"]}: subtask 记录格式损坏')
            response = h.get('response',{})
            decisions.append(dict(id=h['decision'], start=h['start_tick'], end=h['end_tick'],
                executed_steps=h['executed_steps'], state=state,
                text=c.get('subtask_text') if isinstance(c,dict) else None,
                instruction_format=('free_text' if free_text else 'structured') if state=='subtask' else None,
                skill=subtask.get('skill_type') if isinstance(subtask,dict) else None,
                arm=subtask.get('arm') if isinstance(subtask,dict) else None,
                conditioning=c, control=control_kind(h), failure=failure_details(h),
                assessment=h.get('assessment',response.get('assessment',{})),
                reason=response.get('reason',''), response=response))
        counts = Counter(d['state'] for d in decisions)
        instruction = next((d['conditioning'].get('episode_instruction') for d in decisions if d['conditioning']), None)
        if not instruction:
            instruction = read(self.file(archive,'controller/request_000.json'), {}).get('instruction')
        return dict(entry=entry, dt=dt, total_steps=total, video=video, duration=(total+1)*dt,
            video_url=f'/api/video/{entry["id"]}', decisions=decisions, issues=issues,
            model=worker.get('model'), effort=worker.get('reasoning_effort'),
            wall_seconds=result.get('wall_seconds'), route=protocol.get('controller_route','auto/baseline'),
            counts=dict(counts), zero_steps=sum(d['executed_steps']==0 for d in decisions),
            instruction_formats=dict(Counter(d['instruction_format'] for d in decisions if d['state']=='subtask')),
            instruction=instruction)


def create_app(roots=None, defaults=None):
    store = Store(roots or [ROOT], defaults if defaults is not None else DEFAULTS)
    app = FastAPI(title='Policy Review', docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.mount('/static', StaticFiles(directory=HERE/'static'), name='static')

    @app.get('/')
    def index():
        return FileResponse(HERE/'static/index.html')

    @app.get('/api/config')
    def config():
        return dict(default_paths=store.defaults)

    @app.get('/api/scan')
    def scan(path: str | None = None):
        try:
            return store.scan([path] if path else store.defaults)
        except (ValueError,OSError) as error:
            raise HTTPException(400,str(error)) from error

    @app.get('/api/episode/{key}')
    def episode(key: str):
        try:
            return store.episode(key)
        except KeyError as error:
            raise HTTPException(404,str(error)) from error
        except (ValueError,OSError,subprocess.SubprocessError) as error:
            raise HTTPException(422,str(error)) from error

    @app.api_route('/api/video/{key}', methods=['GET','HEAD'])
    def video(key: str, request: Request):
        try:
            path = store.file(store.entry(key)['archive'],'sim/sensors.mp4')
            size = path.stat().st_size
        except (KeyError,ValueError,OSError) as error:
            raise HTTPException(404,'视频不可用') from error
        try:
            start,end,partial = byte_range(request.headers.get('range'),size)
        except ValueError:
            return JSONResponse({'detail':'Range not satisfiable'}, status_code=416,
                                headers={'Content-Range':f'bytes */{size}'})
        headers = {'Accept-Ranges':'bytes','Content-Length':str(end-start+1),'Cache-Control':'no-cache'}
        if partial:
            headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        def stream():
            with path.open('rb') as f:
                f.seek(start)
                left = end-start+1
                while left:
                    block = f.read(min(left,1024*1024))
                    if not block: break
                    left -= len(block)
                    yield block
        return StreamingResponse(iter(()) if request.method=='HEAD' else stream(),
            status_code=206 if partial else 200,media_type='video/mp4',headers=headers)
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=18791)
    parser.add_argument('--allowed-root',action='append')
    parser.add_argument('--path',action='append',help='Default experiment/episode directory; repeatable')
    args = parser.parse_args()
    uvicorn.run(create_app(args.allowed_root,args.path),host=args.host,port=args.port)
