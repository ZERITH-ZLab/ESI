# Collection review videos

Run from the ESI repository with the RoboDojo Python environment.
Review copies are written separately from archived sensors.mp4 and receipts.

- Green: student; yellow: edit; orange: short GPT EEF; purple: long GPT motion.
- White: added pre-action review pause (default 0.8 seconds). Actual stationary
  source frames retain the corresponding control color.
- Red: failed control feedback. Zero-step failures show the current state;
  partial execution retains all action frames, marks the final frame red, then
  holds that final state in red. Failure status/error comes from execution receipts.
- Each zero-step decision has its own feedback. --pause 0 disables ordinary
  holds; failures still receive at least one feedback frame.

Single episode:

```bash
python -m hybrid_rollout.robodojo.tools.render_review_video /path/to/attempt_0 \
  --output-root /mnt/esi/review_videos \
  --pause 0.8 --index
```

Batch collection (preview first, then remove --dry-run):

```bash
python -m hybrid_rollout.robodojo.tools.render_review_batch \
  /mnt/esi/results/collection_run \
  --output-root /mnt/esi/review_videos \
  --workers 2 --pause 0.8 --dry-run
```

Batch scans all archived attempts at the collection layout depth, including
unsuccessful episodes with complete history/summary/video. Source snapshots are
excluded. It validates history continuity and each video's T+1 frame mapping.
It always writes per-video .index.json with action/hold ranges and provenance,
and a batch report updated after each result. One failed encode does not prevent
other episodes from being processed. Any failures/excluded episodes give exit 1.

Repeating the command skips compatible v2 videos with matching inputs and pause
settings. Old versions, missing indexes, changed inputs/settings, and existing
.partial.mp4 files are reported rather than overwritten. Choose a fresh output
root for a different version/settings. Inspect partial output before moving it
out of the way to retry a failed encode. Each encoder uses two FFmpeg threads.
