"""Shared in-container environment probe.

Run inside the agent container at agent-setup time. Reports Python /
torch / CUDA / cuDNN state, a curated package list, and ``/data`` +
``/workspace`` directory listings as a JSON blob. Both
``ReActFinAgent`` and ``SingleShotLLMAgent`` prepend the rendered
output to their first user message so the model knows what's actually
installed (especially critical for single-shot, which has no smoke
loop to discover availability via traceback).

The package allowlist is kept in sync with ``docker/Dockerfile.base``
— prefixes that aren't installed are dropped from the output by the
``pip list`` filter, but listing them here adds noise so we don't.
"""
from __future__ import annotations

import tempfile
import textwrap

from harbor.environments.base import BaseEnvironment


# Curated allowlist filtering the in-container ``pip list``. Must stay
# aligned with ``docker/Dockerfile.base`` — any prefix listed here that
# is not actually installed silently drops from the output. Order is
# alphabetical for readability; the filter is case-insensitive prefix
# match against ``pip list --format=freeze`` lines.
_PACKAGE_ALLOWLIST: tuple[str, ...] = (
    "arch",
    "catboost",
    "h5py",
    "keras",
    "lightgbm",
    "matplotlib",
    "ml_collections",
    "numpy",
    "optuna",
    "pandas",
    "pyyaml",
    "requests",
    "scikit-learn",
    "scipy",
    "sklearn",
    "statsmodels",
    "tensorflow",
    "torch",
    "tqdm",
    "transformers",
    "xgboost",
)


def _probe_script() -> str:
    """Return the in-container probe script as a string.

    Built as one large dedented heredoc so the upload+exec round-trip
    is a single file. The cuDNN-functional check runs a real LSTM
    forward pass with cuDNN enabled and a second one with it disabled
    — that distinguishes "cuDNN headers present but broken at
    runtime" from "cuDNN truly OK", which the cudnn_version metadata
    alone cannot.
    """
    allowlist_repr = ", ".join(repr(p) for p in _PACKAGE_ALLOWLIST)
    return textwrap.dedent(
        f"""\
        import json, os, subprocess, sys
        info = {{"python_version": sys.version.split()[0]}}
        cuda_usable = False
        cudnn_ok = False
        cudnn_disabled_rnn_ok = False
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["torch_cuda_build_version"] = torch.version.cuda
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
                info["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
                try:
                    import torch.nn as nn
                    torch.cuda.init()
                    torch.zeros(1).cuda()
                    cuda_usable = True
                except Exception as exc:
                    info["cuda_init_error"] = str(exc)[:200]
                if cuda_usable:
                    try:
                        torch.backends.cudnn.enabled = False
                        nn.LSTM(4, 4, batch_first=True).cuda()(
                            torch.zeros(2, 3, 4, device="cuda")
                        )
                        cudnn_disabled_rnn_ok = True
                    except Exception as exc:
                        info["cudnn_disabled_rnn_error"] = str(exc)[:200]
                    try:
                        torch.backends.cudnn.enabled = True
                        nn.LSTM(4, 4, batch_first=True).cuda()(
                            torch.zeros(2, 3, 4, device="cuda")
                        )
                        cudnn_ok = True
                    except Exception as exc:
                        info["cudnn_enabled_rnn_error"] = str(exc)[:200]
        except Exception as exc:
            info["torch_error"] = str(exc)[:200]
        info["cuda_usable"] = cuda_usable
        info["cudnn_functional_ok"] = cudnn_ok
        info["cudnn_disabled_rnn_ok"] = cudnn_disabled_rnn_ok
        if cudnn_ok:
            info["cuda_advisory"] = "cuDNN functional. Use GPU freely."
        elif cudnn_disabled_rnn_ok:
            info["cuda_advisory"] = (
                "cuDNN broken at runtime; set torch.backends.cudnn.enabled=False "
                "before constructing RNN/LSTM layers."
            )
        elif cuda_usable:
            info["cuda_advisory"] = (
                "CUDA usable but cuDNN broken; prefer CPU for RNN/Conv layers."
            )
        else:
            info["cuda_advisory"] = "CUDA unavailable; use CPU."
        try:
            pip_out = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True, text=True, timeout=20,
            ).stdout
            allow = ({allowlist_repr},)
            interesting = sorted({{
                line for line in pip_out.splitlines()
                if any(line.lower().startswith(p) for p in allow)
            }})
            info["packages"] = interesting
        except Exception as exc:
            info["packages_error"] = str(exc)[:200]
        try:
            info["data_listing"] = sorted(os.listdir("/data"))[:40]
        except Exception as exc:
            info["data_listing_error"] = str(exc)[:200]
        try:
            info["workspace_listing"] = sorted(os.listdir("/workspace"))[:40]
        except Exception as exc:
            info["workspace_listing_error"] = str(exc)[:200]
        print(json.dumps(info, indent=2))
        """
    )


async def collect_env_probe(environment: BaseEnvironment) -> str:
    """Run the probe in the agent container; return the rendered JSON blob.

    Uploads a small probe script to ``/workspace/.openfinai/_env_probe.py``,
    execs it, and returns stdout (or a one-line failure marker). The 120s
    timeout accommodates cold-start torch CUDA init + the cuDNN-functional
    LSTM forward pass on slow disks.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(_probe_script())
        tmp.flush()
        await environment.upload_file(
            tmp.name, "/workspace/.openfinai/_env_probe.py"
        )
    result = await environment.exec(
        "python /workspace/.openfinai/_env_probe.py",
        timeout_sec=120,
    )
    if result.return_code != 0:
        return (
            f"(env probe failed: rc={result.return_code} "
            f"stderr={(result.stderr or '')[:300]})"
        )
    return (result.stdout or "(empty env probe)").strip()
