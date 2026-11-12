from .schemas import S,I,B,obj,arr,enum,ISSUE,IMAGE,PATCH,AUDIT
READ=obj(case_key=S,start_decision=I,end_decision=I,question=S)
PICTURE=obj(case_key=S,**IMAGE['properties'])
UPDATE=obj(case_key=S,group_id=S,**ISSUE['properties'])
QUERY=obj(kind=enum('cases','issues','groups','patches','prior_patches','cited_issues'),offset=I,limit=I,question=S)
ANALYST=obj(action=enum('continue','finish'),reason=S,memory=S,
            queries=arr(QUERY),reads=arr(READ),images=arr(PICTURE),updates=arr(UPDATE),
            patches=arr(PATCH),deferred_issues=arr(S))
REVIEW=obj(action=enum('continue','finish'),reason=S,memory=S,
           queries=arr(QUERY),reads=arr(READ),images=arr(PICTURE),verdicts=arr(obj(patch_id=S,**AUDIT['properties'])))
