# Shared manipulation guidance

Use normal RGB observations, robot proprioception and executed-action feedback.
Object poses, hidden functional points, category IDs and evaluator measurements
are not required inputs. Do not claim exact object coordinates or contact forces
from images. Head-camera cam_high is the fixed visual reference for task left/right;
wrist views help inspect contact and do not redefine task directions. Robot action
coordinates retain the interface's explicit convention. A desired state or issued
command is not evidence that it occurred.

These are conditional strategies, not a mandatory action sequence or reporting
checklist. Use existing assessment fields and normal returned observations. Apply
them to student proposals as well as GPT corrections under the route's decision
gate. Do not interrupt an effective action merely for a stylistic preference.

## Observe and align before uncertain contact
For easily tipped objects, tight gripper clearance, uncertain contact or a failed
grasp, approach a nearby non-contact viewing pose and inspect fresh feedback before
closing. Check the grasp region, opening, approach direction and clearance. If not
aligned, adjust and inspect again. Do not combine uncertain approach, closure and
transport into one long action. Choose an accessible, graspable region with useful
opposing contact; neither the geometric middle nor a narrow tip is always suitable.
Task-specific shape guidance takes precedence over a generic middle grasp.

## Use an idle wrist camera when the object is occluded
During upright recovery or awkward wrist orientations, the holding arm may block
the head view while the object blocks that arm's wrist camera. If the other arm
is free and not supporting anything, move it to a nearby, collision-free viewing
pose and use its wrist image to inspect the object, base and supporting surface.
Keep the holding arm and gripper target steady; do not release or twist further
just to obtain a view. Approach from a clear side without contacting the object,
the holding arm or completed placements. If neither wrist can view safely, lower
onto support and reposition or regrasp. Use actual images to verify the state;
an occluded view is not evidence of success. Preserve this option across compaction.

## Verify retention before transport
A close command or stopped fingers does not prove possession. Where retention is
uncertain, use a short test lift and check that the object follows without slipping
or uncontrolled turning before carrying it away. If the object sags, slips or turns
unexpectedly relative to the gripper during lifting or carrying, stop transport
and lower it onto a supporting surface before adjusting the grasp. Do not keep
carrying an object that is progressively slipping or tipping in the grasp.
Distinguish this loss of retention from an intentional, controlled change of object
orientation; a non-horizontal object is not by itself evidence of an unstable grasp.
After a miss or unstable hold, change contact position, orientation or hand using
the observed error; do not repeat the same close-and-lift attempt without a reason
to expect a different result.

## Support long or unstable objects with two hands when useful
Judge size and length relative to the gripper opening, reachable contact regions
and unsupported overhang, not relative to objects in other tasks. A hammer or shoe
may turn around a single contact because its heavier portion is poorly supported;
use the visible shape and observed motion, not the object name alone, to decide.
If single-hand holding cannot prevent slipping or uncontrolled turning, or the
visible shape makes stable single-hand support unlikely, prefer separated,
non-conflicting two-hand contacts when both hands can reach and withdraw.
For an already unstable hold, first set the object down and reposition it as needed;
establish two-hand support before lifting again rather than chasing a falling object.
Keep the existing support while the other hand acquires or supports the object;
verify the support before moving. Coordinate both hands with the object instead of
pulling or twisting against each other. Establish support before sequential release
and withdrawal. If there is insufficient room for two hands, set down and regrasp
instead of forcing the second hand in. Two-hand carrying is distinct from handover.
Use ordinary grasp and transport skills with both arms for shared support; handover
means transferring support from one hand to the other, not carrying with both.
Long motion tools move one arm while holding the other; use available coordinated
short dual-arm targets and feedback for simultaneous carrying.

## Push on a support surface, or recover an inaccessible object
Pushing can replace grasp-and-carry when the object has continuous support, the
path and destination are clear, and the goal mainly changes position without
precise orientation control. Choose contact unlikely to tip it, push in short
segments, and reassess displacement and rotation. Do not assume a pure translation
or predictable rotation. Prefer grasping for deliberate large orientation changes,
precise orientation retention, lifting, stacking or crossing obstacles; small
observed push corrections remain appropriate when they improve alignment.
Avoid surface edges, trapping gaps and completed objects. Stop and reassess if the
object tips, turns unexpectedly or approaches an unsafe boundary. When a wall or
obstacle prevents grasping, a small push or an accessible edge grasp and drag may
first move it into open space; observe again before regrasping. Do not repeatedly
attempt the same blocked approach.

## Recover uprightness or the required face
Distinguish a planar turn, a toppled object and a true face reversal before acting.
For a toppled object with a heavier base and a narrower accessible upper end,
consider grasping that end to restore uprightness. Examples include the head of
a nesting doll or a tabletop figurine. "Head" means a securely graspable region
away from the heavier base, not necessarily a literal head or the smallest tip.
When the grasp permits controlled settling, gravity tends to bring the heavier
base below the support point; this can reduce the rotation needed from the wrist.
Treat this as a strategy to verify visually, not a known center-of-mass measurement
or a guarantee that the object can pivot in a rigid grasp.

Align near the chosen contact, verify retention, and lift slowly with only enough
clearance for controlled settling. Observe whether the base hangs upright or
slightly inclined, then correct remaining tilt with small arm movements. Lower
until the base rests on the tabletop; make any final supported adjustments while
holding. Release only at that supported pose, withdraw without contact, and verify
independent stability. Do not release from height and expect the impact to right
the object: it can topple again or knock over completed placements. Recover in a
clear area when needed. If the upper end is inaccessible, fragile, slippery or
unable to retain the body, use supported repositioning, regrasping or a feasible
handover rather than forcing this method or twisting from an unsuitable belly grasp.

Make the main orientation adjustment where there is clearance. If the current
hold or wrist posture cannot reach the supported destination, use a supported
regrasp or a feasible handover. Verify the resulting orientation from feedback;
a wrist turn alone does not prove the object was corrected. Low-drop face-flipping
is only a task-specific option where explicitly described, not a general strategy.

## Transfer between hands
Keep the giver supporting the object while the receiver approaches and acquires
support. Confirm the receiver can retain it before releasing the giver; a closing
command alone is insufficient. Withdraw the giver without hitting the object or
other hand. If contact or clearance is unclear, prefer a supported regrasp.

## Place with support, then verify after withdrawal
For tipping-prone objects, stacking, slots, tight spaces or orientation-sensitive
placement, inspect alignment near the destination before descending in short
segments. Keep holding until support is established, then explicitly open at the
supported pose. Open commands during a moving approach can release too early.
Withdraw along a clear path, then check that the object remains at the intended
location and orientation and that nearby completed objects were not disturbed.
Being held upright is not evidence of independent stability. Recheck any disturbed
placement; a previously completed label is not a permanent guarantee.

## Preserve the goal through feedback
Observe before uncertain contact and verify after consequential changes, without
adding a stop to every ordinary motion. A failed plan may have executed a prefix:
use its actual receipt and new observations. Failure of one motion does not itself
complete or cancel the recovery goal. Continue with a different feasible action,
or explicitly defer it based on evidence and the remaining budget. Hand control
back when the student can usefully continue under the route's gate; a new proposal
alone is not evidence that the unresolved error disappeared. Unknown evidence
must remain unknown rather than becoming a claim of contact, retention or stability.

## Preserve guidance across context compaction
These shared strategies and the current task's placement requirements remain in
force after compaction. Preserve the relevant strategy, unresolved recovery goal,
object and supporting arm, observed attempt outcomes, and next unfinished step in
the compacted context; keep plans distinct from verified results. A compacted
context or new student proposal does not reset the recovery goal or prove success.
If relevant details were lost, consult ../manipulation_guidance.md and
../task_success_conditions.md from the agent workspace, and the host-maintained
episode history for what actually happened. Do not reread everything each round
or create extra summaries solely to restate these rules. Recheck physical state
from current observations before continuing.
