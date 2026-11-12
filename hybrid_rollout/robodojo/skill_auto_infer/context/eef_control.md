# RoboDojo / dual ARX X5 embodiment contract

This contract is injected in full. It defines data and action semantics;
SKILL.md defines the workflow and the gate defines the decision criteria.

## Robot and action semantics

- `robot_profile=robodojo`, simulator RoboDojo, robot dual ARX X5.
- Policy is the official RoboDojo-finetuned OpenPI/JAX checkpoint, configuration
  `pi05_base_aloha_full_sim_arx-x5_seed_0`, normalizer `arx_x5_sim`.
- Three image attachments: `cam_high`, `cam_left_wrist`, `cam_right_wrist`.
  These are the default visual evidence; supplemental viewing follows SKILL.md.
- Proprio/action order is `[left joints 1..6, left opening, right joints 1..6,
  right opening]`. Native opening is continuous **0=closed, 1=open**. The
  student proposal retains continuous opening (only clipped to native [0,1]).
  Native gripper proprio is the last controller opening command, not a
  measured finger gap or proof of grasp. Arm joints and EEF poses are measured.
- Every inference returns all **50** target steps and both-arm FK poses.
  `student` still executes only 1–15 steps before observing/inferencing again.
  `edit`/`eef` retain the 1–5-step limit. Remaining proposal suffix is discarded.
- `current_eef` has `left` and `right`; each trajectory row has both arms.
  Each pose refers to that arm's **link6 in environment-origin coordinates**,
  meters and quaternion **wxyz**. Both arms share these axes, not two rotated
  root frames. The robot-only FK is checked against actual measured link poses.
- `target` for `eef` has `left` and `right`, each containing `position`,
  `quaternion_wxyz`, `gripper_closed` (true means closed, converted to opening=0).
  Both must be explicit. To leave an arm stationary, give its current measured
  pose and intended gripper command. Per arm, target distance <=0.05 m and
  rotation <=0.35 rad from current measured pose.
- `edit` has `left` and `right`, each the existing `delta_position`,
  `delta_rotation_vector`, `gripper=keep/open/closed`. Offsets are in the common
  environment frame. A zero/keep edit leaves that arm's student target unchanged;
  it does NOT freeze that arm. `keep` preserves the continuous proposal opening.
- Ordinary `eef`/`edit` corrections use local damped least-squares IK, recomputed after every
  real control ACK. Task increments <=2 cm/0.1 rad and each joint increment
  <=0.05 rad. No cuRobo planning or teleportation. A requested target is not
  evidence of arrival/contact; inspect after-execution RGB and measured pose.
- Do not reuse DROID's +0.1311 m pad offset. X5's native config reports
  `gripper_bias=0.145` but that scalar alone does not establish an offset axis;
  derive grasp geometry from robot geometry and observations if needed.
- Native control observations are at 25 Hz (0.04 s); physics steps internally
  use the upstream interpolation/controller. Codex observes at chunk boundaries,
  not each physics substep. All executed control observations are recorded.

## Rigid-object pick-and-place preferences

These are preferences, not mandatory grasp templates or extra reporting tasks.
Consider whether the task requires a particular final orientation, alignment or
insertion pose; prefer a grasp compatible with that placement and stable transport.
Prefer accessible opposing contacts that fit the gripper and support the intended
operation. Do not impose a middle-body grasp on a wide, tipping-prone or otherwise
unsuitable shape; use the shared guidance and task-specific grasp region. The geometric
centre may be empty and is not necessarily the centre of mass. Object origins are
not grasp points; the link6 target is not the finger contact centre.
If stability is uncertain, use a short test lift and the returned normal RGB/state
to check following, slip or unintended rotation before committing to transport.
Do not regrasp an already stable object merely to satisfy this preference.
Account for the whole carried object's extent and orientation when choosing
clearance. Prefer establishing stable support before releasing, rather than
dropping an object in an easily tipping pose; verify using returned normal views.
Task requirements and current evidence override these heuristics. Keep aligned
student actions; an aesthetically imperfect grasp alone does not justify takeover.
Reuse existing assessment fields; do not add a spoken checklist, extra scene
queries or repeated image viewing solely to demonstrate compliance.

## Compact input representation (not a new action representation)

student_eef_trajectory contains columns defined once and 50 rows of
[step_index, left_values, right_values]. Each arm has eight columns:
[x, y, z, qw, qx, qy, qz, opening]. Both arms in one row share the control time.
Indices 0..49 are proposal-local, not episode step_id. student steps=N executes
indices 0..N-1 in order, never selected or skipped rows.

Only displayed proposal values are rounded: positions to 4 decimal places
(0.1 mm), quaternions to 5 and continuous opening to 3. Display quaternions
are approximate; output still requires a unit quaternion (norm tolerance 1e-4).
Current measured EEF stays unrounded with both quaternion and RPY, the latter
using scipy Rotation.as_euler('xyz', degrees=True). Opposite quaternion signs
represent the same orientation; angle wrapping is not a physical jump.
Full-precision policy actions, FK and execution validators are unchanged.
edit uses rotation vectors, NOT Euler-angle increments; eef uses absolute wxyz.
Only mode-relevant action fields are submitted: student/stop omit edit and
target, edit supplies edit only, eef supplies target only. The host fills
inactive legacy fields for archival compatibility, never as additional actions.

previous_result.assessment contains the previous assessment once; its reason
and action parameters remain in previous_result.response. Execution counts,
gripper commands and source are in previous_result, not duplicated in
previous_observation. Prior image/state references remain. Audit receipts are
omitted from the teacher view, not from recorded evidence. A recorded decision
receipt alone is not execution evidence. Judge actual execution feedback.
