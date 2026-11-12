# Subtask feedforward

In the feedforward route, GPT receives observations and supplies a structured subtask or null before π05 inference. The renderer appends a non-null instruction to the episode instruction; null keeps the episode instruction alone.

After inference, GPT reviews the proposed prefix and chooses student execution or a correction. Execution produces the next observation. Inference requests are bound to observation IDs to reject stale or duplicate requests.

The contract is defined by `feedforward_prompt.md`, `feedforward_gate_prompt.md` and `subtask_instruction.py`. Select the route through the collection or evaluation configuration.

Per-episode records include:

- `observations/NNN/conditioning_selection.json`: the validated input before inference.
- `proposals/NNN/conditioning.json`: the input associated with the prediction.
- `history.json`: conditioning and control decisions for executed actions.

Use [policy visualization](../../../data_processing/policy_review/README.md) to inspect the actual online instructions.
