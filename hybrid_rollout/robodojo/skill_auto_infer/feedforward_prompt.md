## Host-managed feedforward loop

GPT selects the active goal and needed corrections; pi05 supplies detailed motion.
1. Call robodojo_start once for current RGB/state/task and prior execution feedback.
   It does NOT infer pi05.
2. Assess the previous execution and identify the active goal, immediate operation
   and unresolved error from observations and same-episode history. Reuse an
   existing goal when still valid. Call pi05_infer once with latest observation_id,
   current_intent and subtask (a structured skill or null).
3. The host returns H50 and the same conditioning.current_intent. Reuse this
   pre-inference reference to review the executable prefix under the goal-directed
   gate. Call robodojo_execute or an available bounded correction tool with the
   latest decision_id and usual assessment/reason.
4. Execution returns fresh observations without inference. Repeat from step 2,
   including after corrections and zero-step failures. Stop at rollout_finished.

current_intent is one concise English line (at most 600 characters): the grounded
active goal and immediate operation, including a critical execution requirement or
unresolved error when relevant. It is a public decision summary, not a reasoning
transcript or another full task plan. Supply it even when subtask is null. If the
target is unresolved, state that limitation and the immediate informative action;
do not invent an object/arm. The host archives and echoes it ONLY for GPT review;
pi05 receives only the original episode instruction plus optional canonical subtask.
Before inference, do not fabricate predicted_next_intent or intent_evidence.

Exactly one inference per observation round. No prompt revisions and repeat
predictions, candidate sampling or proposal selection. Invalid inputs that ran no
inference may be corrected; never retry uncertain inference/execution transport
failures. Use the latest next_call IDs. Inference/FK do not advance physics or
predict future object images. Only execution advances steps. Discard unused
proposal suffixes; never reuse old proposals.

## Choosing a subtask or null

Provide subtask when the current skill is clear and should continue. Use null
when the skill is nearing completion, the episode is transitioning between skills,
or the object/acting arm cannot be determined from available observations.
Null means episode-task-only input, NEVER an empty episode instruction. It does
not clear current_intent, unresolved work or failure history. No fixed conditioning
quota or guessed future 25-step boundary is required. Do not give a future skill
before its prerequisites hold. If no predefined skill fits, use a concise custom
only for a clearly grounded operation; otherwise use null.

Keep the goal and entity names stable while the same operation continues. Change
it after observed completion, contradicted prerequisites, infeasibility, or an
explicit task-level replan. Explain meaningful changes briefly in reason; retain
unfinished work in task_progress. Do not switch merely to follow pi05's proposal.
Only permitted observations, task text and same-episode evidence ground goals.
After compaction, use the recorded pre-inference intent and execution history;
previous GPT judgments are evidence to reassess, not physical truth.

## Recovery and completion

Maintain the unresolved recovery purpose across skill changes and control handoffs.
Normal transport/support preferences are subordinate to a grounded task-specific
recovery goal; do not replace that goal with ordinary placement without a reason.
Follow the task's actual recovery limits. Do not infer completion from a release,
changed skill label, corrected approach, or return home alone. If apparent completion
is not confirmed by native termination, reassess visible remaining conditions and
use purposeful actions; do not repeatedly consume steps waiting without evidence.
The original native budget, termination, no-reset and observation restrictions apply.

## Training-compatible skill fields

Identify the current need BEFORE selecting a skill name. Prefer a predefined
skill that accurately describes that operation, at meaningful skill granularity:
approach (move near an object), align (align with an object), grasp (acquire it),
transport (carry it to a destination), place (set it at/on/in a destination),
push (move it along a support surface through non-grasp contact), release (intentionally end a grasp as an independent goal), rotate (change its orientation), handover
(transfer between grippers), retreat (withdraw from an object), return_home.
Use approach for an independent move into the interaction region and align for
an independent contact-readiness adjustment. Do not label every approach as grasp
just because grasping is the later goal. Small acquisition adjustments, closure
and retention-test lifts can remain grasp; do not force a fixed skill sequence or
split every movement. Transport retains an object toward a destination; place
establishes support and release. Moving a held object is transport, not push.
Push includes sliding/advancing through contact and small feedback-driven skew
corrections; do not select it merely because the reasoning says 'reposition'.
Moving an already-held object, including a held drag, is transport. Select release
only when ending the grasp is itself the intended operation, without a supported
placement goal. A release within ordinary place or handover stays in that skill;
ending a pushing contact is not release. Opening alone does not prove release,
and release alone does not prove placement, flipping or parent-task success.
Release uses object_ref, destination_ref=null and target_relation=none.
Rotate describes an object-orientation goal, distinguishing in-plane alignment
from a face change where the observation supports that distinction.
Use custom only when the meaningful action is not adequately represented by one
predefined skill or a natural sequence of predefined skills. Do not fragment
continuous coupled movement merely to avoid custom. Describe only its core goal
in a short, direct, unambiguous single-line phrase; no reasoning or outcome claims.

For a non-null subtask, fill every declared field; use null for unused text slots:
- skill_type: a predefined skill above or custom.
- arm: left/right/both; identify the arm performing the key skill. If unknown use
  subtask=null, not a guessed arm.
- object_ref/destination_ref: stable grounded snake_case names, rendered as words.
  Object is required for predefined skills except return_home; custom must identify
  its object in the concise description and object_ref when applicable. Transport/place/push require
  destination_ref and target_relation to/on/in/near. Otherwise relation is none.
- target_orientation_ref: null unless the action explicitly requires an OBJECT
  orientation as a core goal. Never insert a customary orientation by assumption,
  and never use this slot for wrist pose. Rotate requires a grounded orientation.
- custom_description: null except custom, which requires the concise action text.
- giver_arm/receiver_arm: unknown except handover; handover uses arm=both and two
  distinct known giver/receiver arms.

The host writes canonical English (for example `Left arm: Grasp digit 5.`).
Do not submit free-form combined episode prompts or alter the episode instruction.
Subtask is conditioning, not proof of success or a substitute for correction.
Review against current_intent using the goal-directed gate below. Skill names
never override the actual goal, its prerequisites, or task-specific recovery.

