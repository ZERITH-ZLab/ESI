"""Strict schemas accepted by Codex structured output."""
S={'type':'string'};I={'type':'integer'};B={'type':'boolean'}
def enum(*items):return dict(type='string',enum=list(items))
def obj(**fields):return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
def arr(item):return dict(type='array',items=item)
ISSUE=obj(id=S,start_decision=I,end_decision=I,
    category=enum('easy_execution_error','strategy_failure','execution_limitation','infrastructure','uncertain','successful_recovery'),
    recovery=enum('recovered','unresolved','unknown'),mechanism=S,observed_facts=S,
    existing_strategy=S,why_strategy_failed=S,alternative=S,
    alternative_basis=enum('observed_success','hypothesis','none'),
    evidence_decisions=arr(I),counterexample_decisions=arr(I),
    online_observable_trigger=S,image_ids=arr(S),verified=B)
IMAGE=obj(start_decision=I,end_decision=I,camera=enum('cam_high','cam_left_wrist','cam_right_wrist'),
          sampling=enum('sparse','dense'),question=S)
EPISODE=obj(overview=S,issues=arr(ISSUE),image_requests=arr(IMAGE),limitations=arr(S))
PATCH=obj(id=dict(S,description='Unique proposal ID across the final patch set; distinct from the target rule_id.'),
          scope=S,operation=enum('replace','add','delete'),
          rule_id=dict(S,description='1-64 ASCII letters/digits/_/-. For add: unused learned_ ID within the scope. For replace/delete: exact existing editable rule ID.'),title=S,text=S,
          mechanism=S,evidence_refs=arr(S),success_refs=arr(S),counterexample_refs=arr(S),
          rationale=S,regression_risks=S,why_existing_rule_insufficient=S)
PROPOSAL=obj(summary=S,patches=arr(PATCH),deferred_issues=arr(S))
AUDIT=obj(verdict=enum('approve','reject','needs_closed_loop'),
          evidence_supports_change=B,same_mechanism=B,online_observable=B,
          preserves_goals_and_protocol=B,does_not_target_easy_errors=B,
          counterexamples_reviewed=B,concise_nonredundant=B,reason=S)
