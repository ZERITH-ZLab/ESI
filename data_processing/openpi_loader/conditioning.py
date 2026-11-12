"""Versioned text conditioning; never changes windows, actions or loss masks."""
from dataclasses import dataclass

import numpy as np

from data_processing.subtasks.io import digest, read, require
from data_processing.training_index.build import checksum

VERSION = 'current_subtask_half_v1'


def online_instruction_cutoffs(segments):
    """Same online text stays one interval; trim only its last decision's tail.

    Bounds are half-open executed-step ranges. Zero-action decisions do not
    create training starts. No semantic rewriting or merging across null/gaps.
    """
    runs = []
    previous_end = 0
    for s in segments:
        start, end = s['start_step'], s['end_step']
        require(start >= previous_end and end >= start, 'Unordered/overlapping online decisions')
        previous_end = end
        if start == end:
            continue
        text = s['online_conditioning']['subtask_text']
        if runs and runs[-1]['end'] == start and runs[-1]['text'] == text:
            run = runs[-1]
            run.update(end=end, last_start=start)
        else:
            run = dict(end=end, last_start=start, text=text, ids=[])
            runs.append(run)
        run['ids'].append(s['segment_id'])
    return {sid: run['last_start'] + (run['end']-run['last_start']+1)//2
            for run in runs for sid in run['ids']}


def use_subtask(start, end, step, horizon=50):
    require(horizon == 50, 'Subtask conditioning v1 requires horizon=50')
    require(0 <= start <= step < end, 'Window start is outside its subtask')
    length = end-start
    return end-step >= 25 if length >= 50 else 2*(step-start) < length


def format_prompt(task, subtask=None):
    """Shared offline/online renderer. No reasoning, outcomes or evidence."""
    require(isinstance(task, str) and task.strip(), 'Task text is required')
    if subtask is None:
        return task  # Preserve baseline text byte-for-byte.
    require(isinstance(subtask, str) and subtask.strip(), 'Subtask text is required')
    return f'{task}\nCurrent subtask: {subtask.strip()}'


class Conditioning:
    def __init__(self, dataset, mode):
        require(mode in ('task', 'subtask'), 'conditioning.mode must be task or subtask')
        self.mode = mode
        self.enabled = np.zeros(len(dataset.rows), dtype=bool)
        self.texts = {}
        hashes = {}
        online_keys = set()
        online_cutoffs = {}
        if mode == 'subtask':
            for key, ep in dataset.episodes.items():
                path = ep['annotation_path']
                require(path in dataset.hashes and checksum(path) == dataset.hashes[path],
                        f'Annotation changed or absent from index sources: {path}')
                annotation = read(path)
                require(annotation['input_hash'] == ep['input_hash'] and
                        annotation['version'] == ep['annotation_version'] and
                        annotation['total_steps'] == ep['total_steps'], f'{key}: annotation/index mismatch')
                hashes[path] = dataset.hashes[path]
                if annotation.get('route') == 'online_subtask_outcome':
                    from data_processing.online_subtask_outcomes import VERSION as ONLINE_VERSION
                    require(annotation['version'] == ONLINE_VERSION, 'Unsupported online annotation version')
                    online_keys.add(key)
                    online_cutoffs[key] = online_instruction_cutoffs(annotation['segments'])
                segments = {s['segment_id']: s for s in annotation['segments']}
                require(len(segments) == len(annotation['segments']), f'{key}: duplicate segment IDs')
                for sid, segment in segments.items():
                    self.texts[(key, sid)] = segment
            used = {}
            for n, row in enumerate(dataset.rows):
                key = (row['episode_key'], row['segment_id'])
                require(key in self.texts, f'{key}: subtask missing from annotation')
                s = self.texts[key]
                require(s['outcome'] == 'success' and not s['needs_review'] and
                        s['training_eligibility']['low_level'], f'{key}: indexed subtask is no longer eligible')
                if row['episode_key'] in online_keys:
                    c = s['online_conditioning']
                    task = dataset.episodes[row['episode_key']]['task_instruction']
                    text = c['subtask_text']
                    require(c['episode_instruction'] == task and
                            (c['subtask'] is None) == (text is None) and
                            c['effective_prompt'] == format_prompt(task, text), f'{key}: online prompt mismatch')
                    require(s['start_step'] <= row['start_step'] < s['end_step'], f'{key}: window outside decision')
                    # Preserve text/null; remove conditioning only in the latter
                    # half of the final decision of a continuous same-text run.
                    self.enabled[n] = text is not None and row['start_step'] < online_cutoffs[row['episode_key']][row['segment_id']]
                else:
                    text = s['instruction'].get('text')
                    require(isinstance(text, str) and text.strip(), f'{key}: missing canonical instruction')
                    self.enabled[n] = use_subtask(s['start_step'], s['end_step'], row['start_step'], dataset.horizon)
                used[key] = text
            self.texts = used  # Do not retain reasoning/evidence in worker datasets.
        self.contract = dict(mode=mode, rule_version='route_aware_online_last_decision_half_v2' if online_keys else VERSION,
                             annotation_sources_sha256=digest(hashes))

    def prompt(self, dataset, index):
        row = dataset.rows[index]
        task = dataset.episodes[row['episode_key']]['task_instruction']
        text = self.texts[(row['episode_key'], row['segment_id'])] if self.enabled[index] else None
        return format_prompt(task, text)

    def report(self, dataset, sampler):
        def group(mask):
            enabled = mask & self.enabled
            mass = float(sampler.probabilities[mask].sum())
            imass = float(sampler.probabilities[enabled].sum())
            return dict(windows=int(mask.sum()), with_subtask=int(enabled.sum()),
                        task_only=int((mask & ~self.enabled).sum()),
                        sampling_mass=mass, subtask_sampling_mass=imass,
                        subtask_fraction=imass/mass if mass else 0.)
        controls = np.array([r['start_control'] for r in dataset.rows])
        rounds = np.array([dataset.episodes[r['episode_key']]['collection_round'] for r in dataset.rows])
        return dict(**self.contract, overall=group(np.ones(len(controls), dtype=bool)),
                    controls={c: group(controls == c) for c in ('gpt', 'pi05')},
                    rounds={r: group(rounds == r) for r in sorted(set(rounds))})


@dataclass(frozen=True)
class StrictPaligemmaTokenizer:
    """Check the installed OpenPI Pi05 serialization before its truncating tokenizer."""
    inner: object

    def length(self, prompt, state=None):
        cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
        encoder = self.inner._tokenizer
        if state is not None:
            bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1])-1
            state_str = ' '.join(map(str, bins))
            count = len(encoder.encode(f'Task: {cleaned}, State: {state_str};\nAction: ', add_bos=True))
        else:
            count = len(encoder.encode(cleaned, add_bos=True))+len(encoder.encode('\n'))
        require(count <= self.inner._max_len,
                f'Prompt requires {count} tokens, limit is {self.inner._max_len}; '
                f'refusing silent truncation: {prompt!r}')
        return count

    def tokenize(self, prompt, state=None):
        self.length(prompt, state)
        return self.inner.tokenize(prompt, state)


def prepare_text_transforms(dataset, data_config):
    """Preflight ALL indexed prompts with real normalized states, without reading RGB."""
    import dataclasses
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer

    inputs = list(data_config.model_transforms.inputs)
    tokenizers = [i for i, t in enumerate(inputs) if isinstance(t, transforms.TokenizePrompt)]
    require(len(tokenizers) == 1, 'Expected exactly one OpenPI prompt tokenizer')
    i = tokenizers[0]
    original = inputs[i]
    require(isinstance(original.tokenizer, PaligemmaTokenizer), 'Expected PaligemmaTokenizer')
    strict = StrictPaligemmaTokenizer(original.tokenizer)
    inputs[i] = dataclasses.replace(original, tokenizer=strict)
    dc = dataclasses.replace(data_config, model_transforms=dataclasses.replace(
        data_config.model_transforms, inputs=inputs))
    pre = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
                              transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)])
    maximum = 0
    # Small dummy images only satisfy AlohaInputs' schema. Text/state transformations
    # are identical to training; model image resize is intentionally not invoked.
    images = {f'observation.images.{c}': np.zeros((3, 1, 1), np.uint8)
              for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
    for n, row in enumerate(dataset.rows):
        states, commands = dataset._arrays(row['episode_key'])
        t = row['start_step']
        raw = dict(images, **{'observation.state': states[t].astype(np.float32, copy=True),
                             'action': commands[t:t+1].astype(np.float32, copy=True),
                             'prompt': dataset.conditioning.prompt(dataset, n)})
        value = pre(raw)
        length = strict.length(value['prompt'], value['state'] if original.discrete_state_input else None)
        maximum = max(maximum, length)
    return dc, dict(windows_checked=len(dataset), max_tokens=maximum,
                    token_limit=original.tokenizer._max_len, silent_truncation=False)
