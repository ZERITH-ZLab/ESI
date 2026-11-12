"""Read-only rollout progress notifications. No simulator/model calls."""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

CONFIG = Path.home() / '.config/gpt-as-policy/ntfy.json'
ROOT = '/mnt/esi/results'
LABELS = {'student': 'π0.5', 'edit': 'GPT修改', 'eef': 'GPT EEF'}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def init_config(path):
    if path.exists():
        raise SystemExit(f'配置已存在，请直接编辑：{path}')
    url = input('ntfy完整订阅地址（https://服务器/主题）：').strip()
    if urlsplit(url).scheme != 'https' or not urlsplit(url).netloc or not urlsplit(url).path.strip('/'):
        raise SystemExit('需要完整的HTTPS主题地址')
    token = getpass.getpass('访问令牌（公共主题直接回车）：').strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(url=url, token=token, results_root=ROOT, interval_seconds=300), f, indent=2)
    print(f'已保存配置：{path}（权限600；不要提交到Git）')


class Progress:
    """Incrementally read complete JSON lines; retain incomplete final line."""
    def __init__(self):
        self.offset = 0
        self.inode = None
        self.step = None
        self.mode = None
        self.goal = ''
        self.counters = {}

    def update(self, path):
        try:
            stat = path.stat()
            if self.inode != stat.st_ino or stat.st_size < self.offset:
                self.__init__()
                self.inode = stat.st_ino
            with path.open('rb') as f:
                f.seek(self.offset)
                while True:
                    line = f.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    self.offset = f.tell()
                    try:
                        e = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(e, dict):
                        continue
                    if e.get('event') == 'codex_decision':
                        r = e.get('response', {})
                        self.mode = r.get('mode', e.get('mode'))
                        self.goal = r.get('assessment', {}).get('current_subgoal', '')
                    if e.get('event') == 'tool_done':
                        self.step = e.get('step_id', self.step)
                        self.counters = e.get('counters', self.counters)
        except OSError:
            pass


def latest_batch(experiment, replica=None):
    paths = list(experiment.glob(f'_replicas/replica_{replica if replica is not None else "*"}/attempt_*/batch.json'))
    if not paths:
        return None, {}
    paths.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in paths:
        data = read_json(path)
        if data:
            return path, data
    return None, {}


def render(name, batch, row, progress):
    result = row.get('result') or {}
    archive = Path(row['archive']) if row.get('archive') else None
    run = read_json(archive/'controller/run.json') if archive else {}
    step = result.get('step_id', progress.step)
    counters = progress.counters
    student = result.get('student_steps', counters.get('student_steps', '?'))
    edited = result.get('edited_steps', counters.get('edited_steps', 0))
    recovery = result.get('recovery_steps', counters.get('recovery_steps', 0))
    corrected = edited + recovery
    elapsed = '?'
    try:
        end = row.get('finished_utc') or row.get('reset', {}).get('finished_utc') or batch.get('finished_utc')
        end = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
        elapsed = f"{max(0,(end-datetime.fromisoformat(row['started_utc'])).total_seconds())/60:.0f}"
    except (ValueError, KeyError, TypeError):
        pass
    episodes = batch.get('episodes', [])
    done = sum(e.get('state') in ('completed', 'skipped_completed') for e in episodes)
    failed = sum(e.get('state') == 'failed' for e in episodes)
    lines = [f"{name}｜case {row.get('index', -1)+1}/{batch.get('planned_rollouts', '?')}",
        f"{row.get('variant', '?')} / layout {row.get('layout_id', '?')}",
        f"批次：{batch.get('state')}；本case：{row.get('state', '等待启动')}",
        f"当前 step：{step if step is not None else '?'}/{run.get('max_episode_steps', '?')}",
        f"最近决策：{LABELS.get(progress.mode, progress.mode or '等待决策')}（提交不等于执行成功）",
        f"当前子目标：{str(progress.goal or '等待反馈')[:300]}",
        f"累计：student {student}步，GPT修正 {corrected}步（edit {edited} / eef {recovery}）",
        f"本case耗时：{elapsed}分钟",
        f"批次已完成/跳过：{done}；运行失败：{failed}（completed不代表任务成功）"]
    if 'success' in result:
        lines.append(f"本case任务成功：{result['success']}")
    if row.get('failure_kind'):
        lines.append(f"失败类别：{row['failure_kind']}")
    return '\n'.join(lines)


def publish(config, message):
    headers = {'Content-Type': 'text/plain; charset=utf-8', 'Title': 'RoboDojo progress'}
    if config.get('token'):
        headers['Authorization'] = 'Bearer ' + config['token']
    body = message.encode('utf-8')[:3900].decode('utf-8', errors='ignore').encode('utf-8')
    request = Request(config['url'], data=body, headers=headers, method='POST')
    with urlopen(request, timeout=20) as response:
        response.read(4096)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', help='results下的实验文件夹名称')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--once', action='store_true', help='只打印当前摘要，不发通知')
    parser.add_argument('--replica', type=int, help='多worker时指定replica；默认最新批次')
    args = parser.parse_args()
    if args.init:
        return init_config(args.config)
    if not args.experiment or Path(args.experiment).name != args.experiment or args.experiment in ('.', '..'):
        parser.error('请指定单个实验文件夹名称')
    config = read_json(args.config)
    if not args.once and not config.get('url'):
        parser.error('先使用 --init 保存ntfy配置')
    interval = max(30, int(config.get('interval_seconds', 300)))
    experiment = Path(config.get('results_root', ROOT))/args.experiment
    progress = Progress()
    current = None
    sent = None
    next_send = 0
    print(f'监控：{experiment}；间隔{interval}秒。Ctrl+C仅停止通知。', flush=True)
    while True:
        path, batch = latest_batch(experiment, args.replica)
        row = (batch.get('episodes') or [{}])[-1]
        key = (str(path), row.get('archive'))
        if key != current:
            progress = Progress()
            current = key
        if row.get('archive'):
            progress.update(Path(row['archive'])/'logs/controller.log')
        terminal = bool(batch) and batch.get('state') not in ('running', 'starting')
        signature = (key, progress.offset, json.dumps(batch, sort_keys=True))
        message = render(args.experiment, batch, row, progress) if batch else f'{args.experiment}：等待batch.json生成'
        if args.once:
            print(message)
            return
        now = time.monotonic()
        if now >= next_send and signature != sent:
            try:
                publish(config, message)
                sent = signature
                print(message, flush=True)
                if terminal:
                    print('批次已结束，最终通知已发送，监控退出。', flush=True)
                    return
            except Exception as error:
                # Do not print exception URL or credentials. Retry next interval.
                print(f'通知失败：{type(error).__name__}；下个周期重试。', flush=True)
            next_send = now + interval
        time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n监控停止；实验不受影响。')
