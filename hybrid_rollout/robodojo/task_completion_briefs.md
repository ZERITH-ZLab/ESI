# Task completion briefs

Maintainer reference: task_success_conditions.md retains the full source audit. Only the selected task section below is injected. Names and geometry refer to asset definitions, not live state. Keep random variants explicit when their criteria differ.

## align_blocks

Push the three blocks into a straight line: center-to-line XY error <=0.008m and one local axis of each block aligned with the line (either direction) within 8°. Lifting any block more than 0.007m above its initial height causes failure. Return the arms.

## arrange_largest_number

Read the digits and arrange them on the pads from left to right in the fixed head-camera view to form the largest number. Center each digit on its pad, make it upright and normally readable, and check for a mirrored face rather than only an in-plane turn. Return the arms.

Operation preference, not a success requirement: For a digit and destination clearly on opposite sides, prefer placing it stably in a clear central area reachable by both arms, releasing and withdrawing the source-side arm, then using the destination-side arm to finish. Avoid disturbing placed digits or occupying their pads. Preserve a suitable student staging/handoff plan: an intermediate placement can be correct progress, not a wrong destination. Do not add a handoff when direct placement is comfortably reachable.

Flip recovery: First use the permitted images to distinguish a true mirrored face reversal from a digit that is merely rotated in the tabletop plane. Do not mistake ordinary rotation or an ambiguous view for a mirror flip and attempt difficult flipping maneuvers; for rotation alone, adjust the in-plane orientation instead.

Use a securely graspable region for ordinary transport; do not force a middle grasp when the shape or opening makes it unsuitable. Only for a confirmed face reversal, grasping near the head, tail or one side to create a temporarily unstable grasp is one possible method, not a guarantee: bring the digit toward upright close to the tabletop, tilt it slightly toward the intended falling side, then release from very low height so table contact can turn it onto the other face. Use small adjustments and fresh observations to verify each attempt; never assume release corrected the face. After correction, reassess orientation and regrasp stably for transport. If the current wrist posture prevents a supported maneuver and there is clearance, a verified handover is an alternative; retain support until the receiver holds the digit.

For flip recovery, do not use extreme wrist/arm postures, sustained wrist twisting, flinging, or release from high above the tabletop. If a low, controlled maneuver is not feasible, stop that attempt rather than lifting higher or twisting farther; avoid bouncing the digit out of reach.

Limit the same unresolved flip problem to 2-3 attempts in total, not a fresh budget on each decision. If there is no clear improvement, stop repeating the maneuver and work on other achievable targets. When only difficult unresolved corrections remain, end the episode only if the existing early-abandon tool is available and its gate permits it.

## build_tower

Build the eight-piece tower. Keep block0..7 upright within 5° (block4 uses local +y, the others +z). Block0 rests above one of {block1, block5} and one of {block2, block6}, with root-height gaps >0.025m. Place a member of each pair above block0; block7 goes above one member of each pair by >0.025m, within block0's 0.043m support circle. Block3 goes above block7 by >0.012m and block4 above block3 by >0.015m, each within the lower piece's 0.023m support circle. Align the local x axes of block3/block7 and block4/block3 in either direction within 20°. Return the arms.

## classify_objects

Sort the objects by category into the three baskets, keeping each category together in its own basket. Leave the objects inside rather than on the rims. Return the arms.

## classify_objects_by_language

Follow this episode's language instruction to put each category into its designated basket. Leave the objects inside the baskets. Return the arms.

## cover_blocks

Complete these stages in order: cover only the initially leftmost block; cover leftmost and middle; cover all three; expose red while keeping green/blue covered; expose red/green while keeping blue covered; expose all and return the arms. Keep each cup inverted (local -z upward within 10°). A cup must fully cover the block in XY, with block top below cup top and block bottom above cup bottom -0.03m. Moving any block more than 0.05m from its initial position causes failure. The final arrangement alone does not replace the intermediate stages.

## deposit_coin

Deposit coin0 completely inside piggy_bank. The entire bounding-box projection of coin0 must be strictly inside the bank's XY projection; coin bottom must be at or above both the bank bottom functional point and bounding-box bottom, and coin top at or below both the bank center functional point and bounding-box top. Return the arms.

## fasten_screws

Seat all three nuts on their matching bolts: nut local +z upward within 10°, root XY error <0.001m, and bolt top above nut bottom by >0.009m. Open all target grippers to at least 0.9 and return the arms.

## fill_egg_holder

Put all four eggs in the holder: roots inside its XY bounds and strictly 0.01–0.04m above its bottom functional point. The holder's GetParentLink normalized joint position must exceed 0.9; interpret this using the actual hinge definition. Return the arms.

## fill_pen_holder

Hold the holder with one hand and insert all four items with the other. Items must stay inside, not across the rim. Finally set the holder down upright, confirm stable support, release, then return the arms.

Item orientation: Keep the cap end upward and insert the opposite end. Identify the cap visually; if unclear, improve the view. Check orientation before insertion and after release. If the grasp prevents cap-up insertion, set the item on clear support and regrasp; do not insert cap-first for convenience.

Holder viewing pose: Optionally tilt the opening slightly toward head camera cam_high for visibility. Keep a stable grasp without spilling contents. Tilt only enough to see; no fixed angle is required. Align insertion with the opening and holder axis. Observe and realign after changing its pose.

Insertion guidance: Inspect alignment near the opening and use short motions near the rim. Release when observations indicate the item has entered, is not caught on the rim, and will remain inside; withdraw and verify settling. Decide release from observations, not a fixed depth. Stop advancing if the item catches the rim or causes unintended holder movement; observe and realign with the actual opening, not its remembered position. Regrasp if the holding angle is unsuitable.

Holder recovery: Recover from unintended tilt, slip or instability, not a stable viewing adjustment. Secure any held item on support if needed. Lower the holder upright onto clear support; while still holding it, observe that its base rests steadily before releasing. Regrasp and return to the previously observed undisturbed insertion pose, allowing a small viewing tilt. If the reference is unclear, choose a stable reachable pose. Observe and realign. Preserve cap-up orientation, current holder pose and unresolved alignment across compaction.

## fold_clothes

Fold both sleeves inward toward the opposite chest, then fold the lower hem toward the shoulders. Required offsets in metres (strictly inside ranges): left sleeve relative to right chest x/y (-0.125,0.14); right sleeve relative to left chest x (-0.14,0.125), y (-0.125,0.14); each hem relative to its same-side shoulder x (-0.12,0.12), y (-0.095,0.17). Hem and shoulder lines must differ by <45°. Finish by opening all target grippers to >=0.7 and returning the arms; entering this state triggers the fold check. Warning: returning either arm after the hem is folded but a sleeve remains outside its allowed region causes failure. Complete the sleeves before this return, rather than repeatedly returning with an incomplete fold.

## general_pickup

Lift the target so its root is more than 0.1m above its reference height. No placement or arm return is required.

## hang_mugs

Hang all three mugs: lift each root >0.045m above its reference, place its handle point above 0.9m and within 0.045m of the holder support point. Align handle local z with support local z within 60°, and handle local x with either support +x or -x within 60°. Return the arms.

## imitate_sorting_sequence

Observe and remember the demonstrated object order. Wait for the demonstration arm to finish and withdraw before acting. Place your corresponding objects into the receiving basket in that order; leave later objects outside until their turn and do not disturb the demonstration objects. After a failed grasp or recovery, continue the unfinished item without skipping ahead or repeating completed items. Return the arms after the sequence.

## insert_key

Insert the key with root XY error <0.007m and slot top above key bottom by >0.025m. Keep key local +z upward within 7° and local +x within 30° of the slot's local [0.5,-sqrt(3)/2,0] direction. No arm return is required.

## insert_tubes

Insert all three tubes into the slot region: roots inside its XY bounds and above its bottom, slot top above each tube bottom by >0.045m, tube local +y upward within 30°. Return the arms.

## make_kong

Observe which mahjong category the support arm pushes. Keep the three matching tiles local +z upward within 30° and the other nine local +y upward within 7°. First orient mahjong9_0 within 7° of quaternion wxyz [0,0.707,0.707,0]; then orient its local +y upward within 7°, place its XY within 0.015m of [0.319,-0.15]m and open target grippers to >=0.6. Wait before acting: target arms must remain within 0.3m per axis and 30° of origin until the support arm is within 0.3m per axis and 40°. No final arm return is required.

## make_toast

Place bread in both toaster slots and exactly two pieces in the shelf. Each slot needs a bread root inside its functional XY boundary and strictly 0.03–0.065m above the slot point. Each of all four pieces must have one local +/-x or +/-y axis upward within 30°. Shelf pieces need their roots inside the shelf XY bounds and above its bottom. Set the toast_botton normalized joint position >0.85. Return the arms.

## match_and_pick_from_conveyor

Identify and lift the matched target1 so its root rises >0.1m above its reference. No placement or arm return is required.

## organize_table

Place the mouse on its pad, align the keyboard with the marked frame, put the figurine on its stand, and place the alarm clock on top of the drawer. Let the placements settle and return the arms.

Keep the figurine and alarm clock upright and supported. With tight grasp clearance, observe and align before contact; if an object topples, use the shared upright-recovery or handover strategy.

In this task version, the generated instruction also mentions opening the drawer and storing miscellaneous items, but native completion covers the four named placements and arm return; drawer storage is not an additional completion condition.

## pack_objects_into_box

Place the car, electric toothbrush, hammer and shoe inside the box with their front sides facing left as instructed. The visible fronts are the car nose, toothbrush brush head, hammer head (not the handle end), and shoe toe: point these toward the left side of the head-camera image. Obtain a clearer view if a part is occluded. Keep the box upright and in its initial tabletop orientation. Return the arms.

Before packing, inspect the box surroundings and approach paths. Move obstructing objects to clear staging positions; if necessary, move the still-empty box to open space reachable by both arms, preserving its orientation. Plan positions with room for bulky objects, later entries and hand withdrawal, preferably on the bottom. Stack only when necessary on stable support.

Orient each object in clear space outside the box and verify retention before entry. If it sags or turns uncontrollably in the grasp, set it down, reposition and regrasp; prefer feasible two-hand support when one hand cannot stabilize it. Ensure the hold permits descent, release and withdrawal; regrasp or hand over if needed. Lower close to the support in short, slow segments, establish support, then open in place. After withdrawal, check the new and earlier placements for displacement or rotation. Do not squeeze completed objects aside. Recheck final orientations before returning home.

When all items are inside and most face correctly, gentle box shaking may help minor misalignment settle through contact. Secure the box; use small, slow back-and-forth motions near the table, keeping it upright without spilling. Set it down and inspect every item. If alignment worsens or does not improve, stop and correct the item directly. Shaking does not guarantee alignment or replace orienting reversed items.

## pick_from_conveyor_by_image

Follow the episode instruction and identify the target object on the conveyor by matching it to the image on the board. Obtain a clearer view if the match is uncertain; do not choose solely because an object is closest to the gripper. Pick up that object and place it fully inside the basket, supported near the bottom rather than perched on the rim. Follow the instruction for positioning the basket. No arm return is required.

## play_Xylophone

Strike notes 0 through 7 in order. At each note the mallet beat point must be inside that note's functional XY region and strictly 0.01–0.036m above its hit point. After each of the first seven hits, lift the mallet >0.025m above its just-hit reference height before the next note. No final lift or arm return is required.

## play_stacking_toy

Place block0..3 on pole0, block4..6 on pole1, block7..8 on pole2 and block9 on pole3. Within each group, adjacent root XY errors and the first root's XY error to the pole support must be <0.001m. Base top must be >0.005m above every block bottom. Return the arms.

## play_tic_tac_toe

Place the nine pieces on board cells: each root XY within 0.015m of a cell center and height 0–0.02m above the board root (inclusive). Wait for the support arm before moving: target arms remain within 0.3m per axis and 30° of origin until the support arm is within 0.3m per axis and 40°. No additional final arm return is required.

## plug_in_charger

Insert the charger into the socket region: root inside socket XY bounds and above its bottom, socket top >0.015m above charger bottom, and charger local +y upward within 10°. Return the arms.

## pour_balls_into_vase

Pour all seven balls into the vase: roots inside vase XY bounds and above its bottom. Finish with the cup local +z upward within 15° and return the arms.

## pour_by_language

Follow the current instruction's bottle0→bowl0, bottle1→bowl1, bottle2→bowl2 order. Each pour is checked when the bottle becomes upright again (local +z within 30°). At least 95% of effective poured liquid must be in its bowl; bottle residue <=10% of effective particles, permitted scattered particles <=20% of all particles. Effective poured liquid excludes residue and permitted scatter; aim to pour cleanly and nearly empty each bottle. Returning either arm while a later bottle is complete but an earlier one is incomplete causes failure. No separate final two-arm return is required.

## pour_liquid_into_cup

Pour into the cup, then bring the bottle upright (local +z within 30°); entering upright triggers evaluation. At least 97% of effective poured liquid must be in the cup; bottle residue <=15% of effective particles and permitted scatter <=20% of all particles. Effective poured liquid excludes residue and permitted scatter; aim to pour cleanly and nearly empty the bottle. No arm return is required.

## press_by_number

Read num0 and num1. Press/release button0 exactly num0 times, then button2 once; press/release button1 exactly num1 times, then button2 once and return the arms. Each press must reach joint ratio <0.5 and each release >0.9 before the next stage. Extra counted presses fail, including a third button2 press; avoid bouncing or unnecessary re-presses (counted down transitions start above 0.95).

## push_T

Push t onto target_t: root XY error <0.007m and orientation error <7°. Lifting t >0.01m above its reference causes failure; do not carry it. Return the arms.

## put_bottles_into_dustbin

Put all four bottles fully inside the bin's XY projection, with their tops below bin top and their bottoms 0–0.4m above bin bottom (inclusive). Return the arms.

## solve_equation

Read the current equation and fill its missing mat with a valid digit/operator. Place it within 0.018m in XY, oriented with local +x/+y within 45° of world +x/+y. Subtraction results must be nonnegative; division must have a nonzero divisor and no remainder. The displayed result can contain one or two digits. Return the arms.

## sort_nesting_dolls_by_size

Arrange the five nesting dolls from smallest to largest, left to right in the fixed head-camera view. Align their bases front to back in a straight row, leave gaps, and keep every doll upright. Their faces need not point the same way. Return the arms.

First compare all five sizes and plan the whole row in a clear strip. If necessary, temporarily move obstructing dolls or objects aside. Allow for each base's size, especially the largest, and for gripper access and withdrawal. Keep carrying paths clear of other dolls and recheck any disturbed placement.

For an upright doll, approach downward to grasp a narrow, securely graspable head region. Do not force the gripper around a belly wider than its opening or merely pinch a slippery tip. Observe alignment near contact, close and verify retention with a short test lift before transport.

For a fallen doll, first find an accessible grasp on its head; do not default to gripping its broad middle and twisting it upright in midair. Use a low, controlled lift with only enough clearance for the body to hang below the head. Gravity may help the body hang upright or slightly inclined; verify this visually. Correct the remaining tilt with small arm adjustments. If the head is inaccessible or the body cannot settle under control, use a supported reposition or regrasp instead of forcing a wrist twist.

Lower until the doll's base rests on the tabletop, then make small supported alignment adjustments. Keep holding until it can stand on its base; open at that supported pose, withdraw without contact, and verify it remains upright and aligned. Do not release a suspended doll to land upright: the impact may topple it again or knock over already placed dolls. Recover away from the completed row if clearance is insufficient.

## stack_blocks

Stack the three blocks with adjacent root-height gaps >=0.005m and small XY offsets. XY tolerance depends on block_0 category: 5/8/11/14/17 <=0.015m; 6/9/12/15/18 <=0.0175m; 7/10/13/16/19 <=0.02m; otherwise <=0.015m. No fixed label order is required. Return the arms.

## stack_blocks_by_language

Stack the blocks bottom to top in the order specified by this episode's instruction. Center each upper block over its supporting block. Inspect alignment before lowering, establish support before release, and check for shifting or toppling after withdrawing. Return the arms.

## stack_blocks_random

Stack the three blocks with adjacent root XY error fixed <=0.015m and height gaps >=0.005m. No fixed label order is required. Return the arms.

## stack_bowls

Stack all three bowls with adjacent root XY error <=0.04m and height gaps >=0.005m. Keep local +z upward within 45° for all bowls and within 7° for the bottom bowl. No fixed label order is required. Return the arms.

## store_laptop_and_headphones

Hang the headset with headband_bridge within 0.05m and 30° of support_cradle. Store the laptop with hinge normalized position >0.85, local +y upward within 10°, and its bottom functional point inside the rack's XY bounds and above its bottom. Interpret hinge direction using the actual asset definition. Return the arms.

## store_tools_in_toolbox

Seat the hammer, tape measure, wrench and pliers fully in their matching toolbox slots, lying flat with heads and handles aligned to the slot shapes. Being inside the toolbox is not enough: tools must not perch on a rim, bridge a slot or rest across another tool. Return the arms.

If a fixed completed-state reference is attached, use it to identify slot assignment and target orientation, then locate those features in the CURRENT image. Do not copy its absolute image coordinates or assume the live tools are placed. Without a reference, use visible slot shapes and current observations; do not invent occluded assignments.

Maintain a controlled hold during transport; the tool need not remain horizontal throughout, but must not slip or turn uncontrollably. Prefer feasible two-hand support for tools that one hand cannot stabilize. Adjust major orientation errors outside the slot, inspect alignment nearby, and lower in short segments. Confirm support before opening and withdrawing.

For a small residual offset with the tool supported near its slot, use a closed gripper to push in short segments toward the seated position, observing the result after each segment. Use visible support and clearance; do not force a blocked tool downward. If the position or orientation differs substantially from the slot or reference, or pushing is blocked or does not improve seating, regrasp and realign instead. Confirm the current tool is seated before starting the next one. Check that later actions have not disturbed earlier tools.

## swap_T

Exchange the two T objects' initial poses: root XY error <0.02m and orientation error <3° relative to the other object's initial pose. Return the arms.

## swap_blocks

Use the empty mat to swap the two blocks, confirming each transfer with button0. First lift one block >0.03m while the other stays at its own mat, place it at the empty mat, then press/release. Next lift the other block >0.03m from its updated reference while the first stays at the empty mat, place it at the first block's original mat, then press/release. Finally lift the staged block >0.03m while the other stays at its exchanged mat, place it at the remaining original mat, then press/release and return the arms. Mat XY error must be <0.03m. Press ratio <0.5, release >0.9; more than three counted presses fails. Do not skip intermediate placements or confirmations.

## sweep_blocks

Sweep all the target blocks onto the dustpan. Follow the instruction for acquiring and handing over the broom; use the right hand for the broom and the other hand to position the dustpan. Keep the dustpan to the left of the broom in the fixed head-camera view, with its mouth facing the incoming blocks.

Prefer grasping the rear part of the handle, away from the pan body, leaving room to adjust the pan's pitch. If a grasp near the body prevents keeping the entry lip against the table, set the pan down and regrasp. Slightly raise the handle's rear end while keeping the entry lip in contact with the tabletop, making a shallow entry ramp. Do not lift the whole pan or tilt it steeply. Observe that there is no gap under the entry lip before sweeping in short, controlled passes.

If an object goes underneath the pan, stop moving the broom immediately. Move the pan back a short distance away from the incoming objects to clear the trapped object, then restore entry-lip contact with the table. Confirm the object is in front of the mouth and can enter before resuming. Do not repeatedly sweep while the lip is raised or an object remains underneath. Avoid spilling already collected objects during adjustment.

Check for blocks left outside or pushed away and collect them. At completion, set the loaded pan flat and supported on the tabletop, verify the contents remain inside, and return the arms without disturbing them.

## sweep_blocks_random

Sweep all the target objects onto the dustpan. Follow the instruction for acquiring and handing over the broom; use the right hand for the broom and the other hand to position the dustpan. Keep the dustpan to the left of the broom in the fixed head-camera view, with its mouth facing the incoming objects.

Prefer grasping the rear part of the handle, away from the pan body, leaving room to adjust the pan's pitch. If a grasp near the body prevents keeping the entry lip against the table, set the pan down and regrasp. Slightly raise the handle's rear end while keeping the entry lip in contact with the tabletop, making a shallow entry ramp. Do not lift the whole pan or tilt it steeply. Observe that there is no gap under the entry lip before sweeping in short, controlled passes.

If an object goes underneath the pan, stop moving the broom immediately. Move the pan back a short distance away from the incoming objects to clear the trapped object, then restore entry-lip contact with the table. Confirm the object is in front of the mouth and can enter before resuming. Do not repeatedly sweep while the lip is raised or an object remains underneath. Avoid spilling already collected objects during adjustment.

Check for objects left outside or pushed away and collect them. At completion, set the loaded pan flat and supported on the tabletop, verify the contents remain inside, and return the arms without disturbing them.

## arrange_largest_number_random

Read the digits and arrange them on the pads from left to right in the fixed head-camera view to form the largest number. Center each digit on its pad, make it upright and normally readable, and check for a mirrored face rather than only an in-plane turn. Return the arms.

Operation preference, not a success requirement: For a digit and destination clearly on opposite sides, prefer placing it stably in a clear central area reachable by both arms, releasing and withdrawing the source-side arm, then using the destination-side arm to finish. Avoid disturbing placed digits or occupying their pads. Preserve a suitable student staging/handoff plan: an intermediate placement can be correct progress, not a wrong destination. Do not add a handoff when direct placement is comfortably reachable.

Flip recovery: First use the permitted images to distinguish a true mirrored face reversal from a digit that is merely rotated in the tabletop plane. Do not mistake ordinary rotation or an ambiguous view for a mirror flip and attempt difficult flipping maneuvers; for rotation alone, adjust the in-plane orientation instead.

Use a securely graspable region for ordinary transport; do not force a middle grasp when the shape or opening makes it unsuitable. Only for a confirmed face reversal, grasping near the head, tail or one side to create a temporarily unstable grasp is one possible method, not a guarantee: bring the digit toward upright close to the tabletop, tilt it slightly toward the intended falling side, then release from very low height so table contact can turn it onto the other face. Use small adjustments and fresh observations to verify each attempt; never assume release corrected the face. After correction, reassess orientation and regrasp stably for transport. If the current wrist posture prevents a supported maneuver and there is clearance, a verified handover is an alternative; retain support until the receiver holds the digit.

For flip recovery, do not use extreme wrist/arm postures, sustained wrist twisting, flinging, or release from high above the tabletop. If a low, controlled maneuver is not feasible, stop that attempt rather than lifting higher or twisting farther; avoid bouncing the digit out of reach.

Limit the same unresolved flip problem to 2-3 attempts in total, not a fresh budget on each decision. If there is no clear improvement, stop repeating the maneuver and work on other achievable targets. When only difficult unresolved corrections remain, end the episode only if the existing early-abandon tool is available and its gate permits it.

## fold_clothes_random

Fold both sleeves inward toward the opposite chest, then fold the lower hem toward the shoulders. Required offsets in metres (strictly inside ranges): left sleeve relative to right chest x/y (-0.125,0.14); right sleeve relative to left chest x (-0.14,0.125), y (-0.125,0.14); each hem relative to its same-side shoulder x (-0.12,0.12), y (-0.095,0.17). Hem and shoulder lines must differ by <45°. Finish by opening all target grippers to >=0.7 and returning the arms; entering this state triggers the fold check. Warning: returning either arm after the hem is folded but a sleeve remains outside its allowed region causes failure. Complete the sleeves before this return, rather than repeatedly returning with an incomplete fold.

## hang_mugs_random

Hang all three mugs: lift each root >0.045m above its reference, place its handle point above 0.9m and within 0.045m of the holder support point. Align handle local z with support local z within 60°, and handle local x with either support +x or -x within 60°. Return the arms.

## make_toast_random

Place bread in both toaster slots and exactly two pieces in the shelf. Each slot needs a bread root inside its functional XY boundary and strictly 0.03–0.065m above the slot point. Each of all four pieces must have one local +/-x or +/-y axis upward within 30°. Shelf pieces need their roots inside the shelf XY bounds and above its bottom. Set the toast_botton normalized joint position >0.85. Return the arms.

## pack_objects_into_box_random

Place the car, electric toothbrush, hammer and shoe inside the box with their front sides facing left as instructed. The visible fronts are the car nose, toothbrush brush head, hammer head (not the handle end), and shoe toe: point these toward the left side of the head-camera image. Obtain a clearer view if a part is occluded. Keep the box upright and in its initial tabletop orientation. Return the arms.

Before packing, inspect the box surroundings and approach paths. Move obstructing objects to clear staging positions; if necessary, move the still-empty box to open space reachable by both arms, preserving its orientation. Plan positions with room for bulky objects, later entries and hand withdrawal, preferably on the bottom. Stack only when necessary on stable support.

Orient each object in clear space outside the box and verify retention before entry. If it sags or turns uncontrollably in the grasp, set it down, reposition and regrasp; prefer feasible two-hand support when one hand cannot stabilize it. Ensure the hold permits descent, release and withdrawal; regrasp or hand over if needed. Lower close to the support in short, slow segments, establish support, then open in place. After withdrawal, check the new and earlier placements for displacement or rotation. Do not squeeze completed objects aside. Recheck final orientations before returning home.

When all items are inside and most face correctly, gentle box shaking may help minor misalignment settle through contact. Secure the box; use small, slow back-and-forth motions near the table, keeping it upright without spilling. Set it down and inspect every item. If alignment worsens or does not improve, stop and correct the item directly. Shaking does not guarantee alignment or replace orienting reversed items.

## pour_liquid_into_cup_random

Pour into the cup, then bring the bottle upright (local +z within 30°); entering upright triggers evaluation. At least 97% of effective poured liquid must be in the cup; bottle residue <=15% of effective particles and permitted scatter <=20% of all particles. Effective poured liquid excludes residue and permitted scatter; aim to pour cleanly and nearly empty the bottle. No arm return is required.

## push_T_random

Push t onto target_t: root XY error <0.007m and orientation error <7°. Lifting t >0.01m above its reference causes failure; do not carry it. Return the arms.

## sort_nesting_dolls_by_size_random

Arrange the five nesting dolls from smallest to largest, left to right in the fixed head-camera view. Align their bases front to back in a straight row, leave gaps, and keep every doll upright. Their faces need not point the same way. Return the arms.

First compare all five sizes and plan the whole row in a clear strip. If necessary, temporarily move obstructing dolls or objects aside. Allow for each base's size, especially the largest, and for gripper access and withdrawal. Keep carrying paths clear of other dolls and recheck any disturbed placement.

For an upright doll, approach downward to grasp a narrow, securely graspable head region. Do not force the gripper around a belly wider than its opening or merely pinch a slippery tip. Observe alignment near contact, close and verify retention with a short test lift before transport.

For a fallen doll, first find an accessible grasp on its head; do not default to gripping its broad middle and twisting it upright in midair. Use a low, controlled lift with only enough clearance for the body to hang below the head. Gravity may help the body hang upright or slightly inclined; verify this visually. Correct the remaining tilt with small arm adjustments. If the head is inaccessible or the body cannot settle under control, use a supported reposition or regrasp instead of forcing a wrist twist.

Lower until the doll's base rests on the tabletop, then make small supported alignment adjustments. Keep holding until it can stand on its base; open at that supported pose, withdraw without contact, and verify it remains upright and aligned. Do not release a suspended doll to land upright: the impact may topple it again or knock over already placed dolls. Recover away from the completed row if clearance is insufficient.

## stack_bowls_random

Stack all three bowls with adjacent root XY error <=0.04m and height gaps >=0.005m. Keep local +z upward within 45° for all bowls and within 7° for the bottom bowl. No fixed label order is required. Return the arms.

## store_laptop_and_headphones_random

Hang the headset with headband_bridge within 0.05m and 30° of support_cradle. Store the laptop with hinge normalized position >0.85, local +y upward within 10°, and its bottom functional point inside the rack's XY bounds and above its bottom. Interpret hinge direction using the actual asset definition. Return the arms.
