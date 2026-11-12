"""Evidence and publication invariants shared by persistent analysis and review."""

def validate_report(report, records, cfg, image_ids, *, scan=False):
    ids={r['decision'] for r in records};seen=set()
    if len(report['issues'])>cfg['max_issues_per_chunk']:raise ValueError('Too many issues; merge duplicate recovery attempts')
    for issue in report['issues']:
        if issue['id'] in seen:raise ValueError('Duplicate issue ID')
        seen.add(issue['id'])
        refs=issue['evidence_decisions']+issue['counterexample_decisions']
        if (not issue['evidence_decisions'] or not set(refs)<=ids
                or issue['start_decision'] not in ids or issue['end_decision'] not in ids
                or issue['start_decision']>issue['end_decision']):
            raise ValueError('Issue must cite real decision IDs in this evidence window')
        if not set(issue['image_ids'])<=image_ids:raise ValueError('Cited image was not supplied')
        if scan and issue['verified']:raise ValueError('Initial scan is not verified evidence')
        if issue['category']=='successful_recovery' and issue['recovery']!='recovered':
            raise ValueError('Successful recovery needs a recovered outcome')
        if issue['verified'] and not issue['observed_facts'].strip():raise ValueError('Verified issue needs explicit facts')


def support_check(patch, dossiers, cfg):
    """Frequency is a publication floor AFTER difficulty and strategy relevance checks."""
    groups={}
    for field in ('evidence_refs','success_refs','counterexample_refs'):
        refs=patch[field]
        if len(set(refs))!=len(refs) or not set(refs)<=set(dossiers):raise ValueError('Unknown or duplicate dossier reference')
        groups[field]=[dossiers[r] for r in refs]
    failures=groups['evidence_refs'];successes=groups['success_refs']
    if not failures or any(not x['verified'] or x['category']!='strategy_failure'
                           or not x['why_strategy_failed'].strip() for x in failures):
        raise ValueError('Patch must target verified strategy failures, not easy execution errors')
    if any(not x['verified'] for group in groups.values() for x in group):
        raise ValueError('All cited dossiers must be verified')
    if not successes or any(x['category']!='successful_recovery' or x['recovery']!='recovered' for x in successes):
        raise ValueError('No verified successful alternative: preserve as candidate for closed-loop testing')
    scope=patch['scope'];cases={(x['task'],x['case_id']) for x in failures}
    if scope=='shared':
        if len(cases)<cfg['min_global_cases'] or len({x['task'] for x in failures})<cfg['min_global_tasks']:
            raise ValueError('Global rule lacks independent cross-task support')
    elif len(cases)<cfg['min_task_cases'] or any(x['task']!=scope for g in groups.values() for x in g):
        raise ValueError('Task rule lacks independent in-task support or cites other tasks')
