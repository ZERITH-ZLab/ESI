"""Pinned rollout-only gateway; historical source snapshots retain their identity."""

import os
from .codex_backend.profiles import profile
from .model_config import model_selection

AUTH_PROFILE = profile()['name']
AUTH_MODE = profile()['auth_mode']
PROVIDER = profile()['provider']
MODEL, EFFORT = model_selection(os.environ)
BASE_URL = profile()['base_url']
WIRE_API = 'responses'
DEFAULT_CODEX_IMAGE_MAX_EDGE = 480  # teacher attachments only; 0 keeps originals
