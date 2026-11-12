# Policy visualization

View online subtask instructions, control sources, decisions and rollout videos. Validated online outcome annotations can be overlaid; drafts are not treated as results.

From the repository root, using the control-tools Python environment:

```bash
python -m data_processing.policy_review \
  --host 127.0.0.1 --port 18791 \
  --path /mnt/esi/results --allowed-root /mnt/esi
```

Install `requirements.txt` from this directory if needed; video inspection also requires `ffprobe`.

On a remote deployment, forward the port:

```bash
ssh -N -L 18791:127.0.0.1:18791 your_user@your_host
```

Open `http://127.0.0.1:18791`. Use Scan to discover new experiments. The viewer supports playback speed, frame stepping, decision navigation and filtering by control source or instruction type. Both structured and free-text instructions are displayed as recorded; null denotes episode-only conditioning.

Video frames show simulator progress, not model waiting time. Selecting a decision shows its pre-execution observation. Zero-step failures may share an observation time and remain separate events.

The viewer reads source data without rewriting it. To inspect another storage root, pass `--allowed-root` and `--path` explicitly. Keep the service bound to localhost when using SSH forwarding.
