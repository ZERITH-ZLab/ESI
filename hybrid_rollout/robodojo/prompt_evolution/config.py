"""Independent offline model/budget settings. Online rollout settings are untouched."""
DEFAULTS = dict(
    enabled=False, model='gpt-6-astra', reasoning_effort='medium',
    workflow='conversation_v2', conversation_max_turns=1000,
    conversation_max_reads_per_turn=4, conversation_max_read_decisions=40,
    conversation_max_updates_per_turn=8, conversation_memory_chars=6000,
    annotation_config='data_processing/subtasks/config.local.json',
    auto_compact_token_limit=480000, model_context_window=544000, max_input_chars=180000,
    timeout_seconds=900, max_attempts=3, max_images_per_episode=24,
    max_images_per_call=8, image_max_edge=640, max_image_rounds=3,
    sparse_decision_stride=8, dense_decision_radius=2,
    max_issues_per_chunk=8, max_patches_per_scope=3, max_rule_words=180,
    max_growth_words=120, max_shared_words=2200, max_task_words=900,
    min_task_cases=2, min_global_cases=3,
    min_global_tasks=2, account_wait_seconds=600, publish='auto_verified',
)


def settings(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown prompt_evolution configuration keys')
    cfg = dict(DEFAULTS, **value)
    if cfg['workflow'] != 'conversation_v2':
        raise ValueError('Only conversation_v2 is supported; segmented_v1 was removed')
    if type(cfg['enabled']) is not bool:
        raise ValueError('prompt_evolution.enabled must be boolean')
    for key, default in DEFAULTS.items():
        if type(default) is int and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError(f'prompt_evolution.{key} must be a positive integer')
    if cfg['publish'] not in ('auto_verified', 'candidate_only'):
        raise ValueError('publish must be auto_verified or candidate_only')
    if cfg['reasoning_effort'] not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('Invalid offline reasoning effort')
    if not isinstance(cfg['model'], str) or not cfg['model']:
        raise ValueError('An explicit offline model is required')
    if not isinstance(cfg['annotation_config'], str) or not cfg['annotation_config'].strip():
        raise ValueError('annotation_config must point to the annotation account configuration')
    if cfg['max_images_per_call'] > cfg['max_images_per_episode']:
        raise ValueError('Per-call image budget exceeds per-episode budget')
    if cfg['auto_compact_token_limit'] > cfg['model_context_window'] * 9 // 10:
        raise ValueError('Compaction threshold must leave at least 10% context headroom')
    return cfg
