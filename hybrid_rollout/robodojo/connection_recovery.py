"""Process-shared circuit breaker for classified transport/compaction failures.

Counts failed episode attempts, not native task failures or quota/auth errors.
State lasts for one scheduler invocation; receipts preserve deferred cases on disk.
"""
import time

EPISODE_RETRIES = 1
RETRYABLE_EPISODE_FAILURES = frozenset({
    'simulator_connection_failed', 'codex_transport_failed', 'codex_compaction_failed',
})
RETRY_DELAY_SECONDS = 30
WINDOW_SECONDS = 15 * 60
WORKER_THRESHOLD = 3
CONSECUTIVE_THRESHOLD = 3


class ConnectionRecovery:
    def __init__(self, ctx, stop, jobs, workers=8):
        self.stop = stop
        self.lock = ctx.RLock()
        self.last_failure = ctx.Array('d', [-1.] * workers, lock=False)
        self.consecutive = ctx.Array('i', workers, lock=False)
        self.deferred = ctx.Array('i', jobs, lock=False)
        self.trigger = ctx.Value('i', 0, lock=False)

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            recent = [f'worker_{i}' for i, t in enumerate(self.last_failure)
                      if t >= 0 and 0 <= now - t <= WINDOW_SECONDS]
            return dict(window_seconds=WINDOW_SECONDS, recent_failed_workers=recent,
                        consecutive_failures=list(self.consecutive),
                        pending_recollection_job_indices=[i for i, v in enumerate(self.deferred) if v],
                        pause_reason={0: None, 1: 'worker_consecutive_connection_failures',
                                      2: 'multiple_workers_connection_failures'}[self.trigger.value])

    def failed(self, worker, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self.last_failure[worker] = now
            self.consecutive[worker] += 1
            status = self.snapshot(now)
            if not self.trigger.value:
                if self.consecutive[worker] >= CONSECUTIVE_THRESHOLD:
                    self.trigger.value = 1
                elif len(status['recent_failed_workers']) >= WORKER_THRESHOLD:
                    self.trigger.value = 2
            if self.trigger.value:
                self.stop.set()  # Drain running episodes; do not terminate them.
            return self.snapshot(now)

    def completed(self, worker):
        with self.lock:
            self.consecutive[worker] = 0

    def defer(self, job_index):
        with self.lock:
            self.deferred[job_index] = 1


def wait_before_retry(cancelled):
    """Wait without holding an account lock, with prompt drain/retire checks."""
    deadline = time.monotonic() + RETRY_DELAY_SECONDS
    while not cancelled():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1., remaining))
