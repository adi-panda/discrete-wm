"""Lightweight helpers for pushing checkpoints to the HuggingFace Hub.

Checkpoints live under `<repo>/<exp_name>/<filename>`. Uploads happen in a
background thread so training is not blocked by network I/O; failures are
logged and swallowed so a flaky connection never kills a run.
"""

import os
import threading
import traceback

from huggingface_hub import HfApi

DEFAULT_REPO = "adipanda/discrete-wm"
DEFAULT_EXP = "breakout-v1"

_api = HfApi()
_lock = threading.Lock()


def _upload(local_path, path_in_repo, repo_id, repo_type):
    try:
        with _lock:
            _api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
        print(f"[hf] uploaded {local_path} -> {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"[hf] upload FAILED for {local_path}: {e}")
        traceback.print_exc()


def push_checkpoint(local_path, exp_name=DEFAULT_EXP, repo_id=DEFAULT_REPO,
                    remote_name=None, blocking=False):
    """Push a local checkpoint to `<repo>/<exp_name>/<remote_name>`.

    Runs in a background thread by default. Set blocking=True to wait.
    """
    if not os.path.exists(local_path):
        print(f"[hf] skip push: {local_path} does not exist")
        return None
    if remote_name is None:
        remote_name = os.path.basename(local_path)
    path_in_repo = f"{exp_name}/{remote_name}"

    if blocking:
        _upload(local_path, path_in_repo, repo_id, "model")
        return None
    t = threading.Thread(
        target=_upload,
        args=(local_path, path_in_repo, repo_id, "model"),
        daemon=True,
    )
    t.start()
    return t


def ensure_repo(repo_id=DEFAULT_REPO, private=True):
    """Create the repo if it doesn't exist. Safe to call repeatedly."""
    try:
        _api.create_repo(repo_id=repo_id, repo_type="model",
                         private=private, exist_ok=True)
    except Exception as e:
        print(f"[hf] ensure_repo({repo_id}) warning: {e}")
