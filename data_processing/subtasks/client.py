"""Small REST adapter; no Codex auth files and no credentials in saved artifacts."""
import json
import os
import time
import signal
import subprocess
import tempfile
import re
from pathlib import Path

import httpx

from .repair import response_schema
from .io import require
from .io import write


def codex_failure_detail(workspace):
    """Prefer fatal JSON events over stderr (often empty with codex exec --json)."""
    fatal, error = None, None
    events, stderr = workspace / "events.jsonl", workspace / "stderr.log"
    if events.is_file():
        with events.open(errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "turn.failed":
                    failure = event.get("error")
                    fatal = failure.get("message") if isinstance(failure, dict) else failure
                elif event.get("type") == "error":
                    error = event.get("message")
    detail = fatal or error
    # Some Codex versions embed the upstream JSON error as a message string.
    if isinstance(detail, str):
        try:
            upstream = json.loads(detail)
        except ValueError:
            upstream = None
        if isinstance(upstream, dict) and isinstance(upstream.get("error"), dict):
            info = upstream["error"]
            detail = ": ".join(str(v) for v in (
                upstream.get("status"), info.get("code") or info.get("type"), info.get("message")) if v)
    if not isinstance(detail, str) or not detail.strip():
        detail = ""
        if stderr.is_file():
            with stderr.open("rb") as stream:
                stream.seek(max(0, stderr.stat().st_size - 4096))
                detail = stream.read().decode(errors="replace").strip()
        if not detail:
            detail = "No fatal event or stderr message; the process failed or did not write its final output"
    # Error summaries may enter status.json/stdout; do not forward common secret forms.
    detail = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", detail)
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", detail)
    detail = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[REDACTED]", detail)
    detail = re.sub(r"(?i)([?&](?:token|access_token|refresh_token|api_key|key)=)[^&\s]+", r"\1[REDACTED]", detail)
    return f"{' '.join(detail.split())[:1200]}; inspect {events} and {stderr}"


class CodexClient:
    """One leased, independently logged-in profile; no credential copying.

    The caller owns account_lock for the lifetime of this client. The child inherits
    the lock FD so killing the batch parent cannot release a still-running account.
    """
    def __init__(self, config, profile, lease_fd, workspace, schema_factory=response_schema):
        self.config, self.profile, self.lease_fd = config, profile, lease_fd
        self.workspace = Path(workspace)
        self.schema_factory = schema_factory

    def complete(self, prompt, payload, correction=None, *, images=()):
        from hybrid_rollout.robodojo.codex_backend.profiles import account_home, clean_login_env
        cfg = self.config
        self.workspace.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="call_", dir=self.workspace))
        schema_path = workspace / "schema.json"
        final_path = workspace / "last_message.json"
        write(schema_path, self.schema_factory(correction, payload))
        final_path.unlink(missing_ok=True)
        home = account_home(self.profile, cfg["shared_root"])
        env = clean_login_env(home, cfg["proxy_file"])
        command = [cfg["codex_bin"], "exec", "--ignore-user-config", "--ignore-rules",
                   "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                   "--cd", str(workspace), "--model", cfg["model"],
                   "--output-schema", str(schema_path), "--output-last-message", str(final_path),
                   "--color", "never", "--json"]
        overrides = {"model_provider": "openai", "forced_login_method": "chatgpt",
                     "cli_auth_credentials_store": "file", "approval_policy": "never",
                     "model_reasoning_effort": cfg["reasoning_effort"],
                     "web_search": "disabled", "features.shell_tool": False,
                     "features.unified_exec": False, "features.apply_patch_freeform": False,
                     "features.view_image": False, "features.multi_agent": False,
                     "features.apps": False, "features.skill_search": False,
                     "features.skip_host_skill_discovery": True,
                     "features.hooks": False, "features.image_generation": False,
                     "sqlite_home": str(workspace / "runtime_db"),
                     "log_dir": str(workspace / "runtime_logs")}
        if "auto_compact_token_limit" in cfg:
            overrides["model_auto_compact_token_limit"] = cfg["auto_compact_token_limit"]
        if "model_context_window" in cfg:
            overrides["model_context_window"] = cfg["model_context_window"]
        for key, value in overrides.items():
            command += ["-c", f"{key}={json.dumps(value)}"]
        for path in images:
            command += ["--image", str(Path(path).resolve())]
        command.append("-")
        request = prompt + "\n\nEPISODE LEDGER:\n" + json.dumps(payload, ensure_ascii=False)
        if correction is not None:
            request += "\nRepair request (follow mode and supplied schema):\n" + json.dumps(correction)
        # No shell interpolation; only the final structured message enters the pipeline.
        with (workspace / "events.jsonl").open("w") as stdout, (workspace / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                       env=env, text=True, start_new_session=True,
                                       pass_fds=(self.lease_fd,) if isinstance(self.lease_fd, int)
                                       else tuple(self.lease_fd))
            try:
                process.communicate(request, timeout=cfg["timeout_seconds"])
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if process.returncode != 0 or not final_path.is_file():
            raise ValueError(f"Codex exited {process.returncode}; {codex_failure_detail(workspace)}")
        if cfg.get("reject_compaction"):
            # This offline workflow needs complete evidence in each bounded call.
            # A compressed call is not silently accepted as a reviewed judgment.
            for line in (workspace / "events.jsonl").read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = str(event.get("type", "")) + " " + str((event.get("item") or {}).get("type", ""))
                if "compact" in kind.lower():
                    raise ValueError(f"Context compaction detected; split evidence input; inspect {workspace}")
        return dict(backend="codex", profile=self.profile, model=cfg["model"],
                    runtime=str(workspace), text=final_path.read_text())


class APIClient:
    def __init__(self, config, transport=None, schema_factory=response_schema):
        self.config, self.transport = config, transport
        self.schema_factory = schema_factory

    def complete(self, prompt, payload, correction=None):
        cfg = self.config
        key = os.environ.get(cfg["api_key_env"])
        require(key, f"Set {cfg['api_key_env']} before annotation")
        user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if correction is not None:
            user += "\nRepair request (follow mode and supplied schema):\n"
            user += json.dumps(correction, ensure_ascii=False)
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        fmt = dict(type="json_schema", name="subtask_annotation", schema=self.schema_factory(correction, payload), strict=True)
        if cfg["api_mode"] == "responses":
            endpoint = "/responses"
            body = dict(model=cfg["model"], input=messages, text={"format": fmt},
                        max_output_tokens=cfg["max_output_tokens"], store=False)
        else:
            endpoint = "/chat/completions"
            body = dict(model=cfg["model"], messages=messages,
                        response_format={"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}},
                        max_completion_tokens=cfg["max_output_tokens"])
        with httpx.Client(timeout=cfg["timeout_seconds"], transport=self.transport,
                          follow_redirects=False) as client:
            for attempt in range(cfg["http_retries"] + 1):
                try:
                    result = client.post(cfg["base_url"].rstrip("/") + endpoint,
                                         headers={"Authorization": f"Bearer {key}"}, json=body)
                except httpx.TransportError:
                    if attempt == cfg["http_retries"]:
                        raise RuntimeError("API network/timeout error; inspect connectivity or increase timeout") from None
                    time.sleep(min(2 ** attempt, 20))
                    continue
                if result.status_code in (408, 429, 500, 502, 503, 504) and attempt < cfg["http_retries"]:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                # Do not persist response bodies on auth/HTTP errors: gateways may echo secrets.
                require(result.is_success, f"API HTTP {result.status_code}; check endpoint/model/credentials/quota")
                return result.json()
        raise RuntimeError("API retry budget exhausted")


def extract_text(response, mode):
    if mode == "codex":
        return json.loads(response["text"])
    if mode == "responses":
        require(response.get("status") == "completed", "API response incomplete/refused; increase output budget or inspect raw response")
        chunks = [part for item in response.get("output", []) if item.get("type") == "message"
                  for part in item.get("content", [])]
        require(not any(p.get("type") == "refusal" for p in chunks), "Model refused annotation")
        text = "".join(p.get("text", "") for p in chunks if p.get("type") == "output_text")
    else:
        choices = response.get("choices", [])
        require(len(choices) == 1 and choices[0].get("finish_reason") == "stop", "Incomplete chat completion")
        message = choices[0]["message"]
        require(not message.get("refusal"), "Model refused annotation")
        text = message.get("content")
    require(isinstance(text, str) and text.strip(), "API returned no annotation text")
    return json.loads(text)
