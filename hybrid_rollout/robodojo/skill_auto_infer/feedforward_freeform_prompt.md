## Free-text subtask input

There is no predefined skill vocabulary or structured skill form in this experiment.
Write subtask as one concise English instruction describing the current operation,
with the acting arm and object/destination when known. Use your own wording; do not
encode a skill object or copy the entire episode plan. Limit it to one line of at
most 240 characters. Use null under the transition/uncertainty rules above.

The host preserves the original episode instruction and appends your text after
`Current subtask:`. It does not classify or rewrite your operation into a skill.
Do not alter the episode instruction. Subtask is conditioning, not proof of success
or a substitute for correction. Review against current_intent using the same
goal-directed gate and retain unresolved recovery goals across control handoffs.
