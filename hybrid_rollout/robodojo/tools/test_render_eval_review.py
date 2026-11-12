from . import render_eval_review as review
from ..regression_eval import specs


def test_combines_seed_campaigns_without_collisions(tmp_path, monkeypatch):
    def report(shared, experiment, *, unified):
        rows = [dict(c, archive=f'/original/{experiment}/seed{seed}/{c["case_id"]}',
                     native_success=False, native_score=.25, steps=1050, gpt_steps=50)
                for seed in range(3) for c in specs(seed)]
        return dict(completed=15, planned=15, episodes=rows)
    monkeypatch.setattr(review, 'report', report)
    jobs = review.plan(tmp_path, tmp_path/'reviews', [('base','old',False),('plain','new',True)])
    assert len(jobs) == len({j['output'] for j in jobs}) == 30
    for label in ('base','plain'):
        subset = [j for j in jobs if j['model'] == label]
        assert len(subset) == 15
        assert {j['policy_rng_seed'] for j in subset} == {0,1,2}
        assert all(f'repeat_{j["policy_rng_seed"]+1:03d}.mp4' in j['output'] for j in subset)
        assert not any('_pseed' in j['output'] for j in subset)


def test_rejects_incomplete_eval(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(review, 'report', lambda *a, **kw: dict(completed=14,planned=15))
    with pytest.raises(ValueError, match='finish eval first'):
        review.plan(tmp_path, tmp_path/'reviews', [('base','old',False)])
