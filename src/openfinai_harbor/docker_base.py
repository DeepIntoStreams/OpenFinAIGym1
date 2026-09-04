"""Base-image provisioning helper for curated OpenFinGym task bundles.

Every task's ``environment/Dockerfile`` starts with ``FROM`` the shared
base image (``BASE_IMAGE_TAG``), a published registry image.
``ensure_base_image`` makes that image available before a trial, using
the same pull-first policy as the Podman provider
(``environments/podman.py``): if the image is not already cached
locally, it ``docker pull``s it from the registry.  A local build from
the given ``dockerfile`` is the escape hatch only — taken when
``force_rebuild`` is set (a developer who edited it) or when the pull
fails (offline / registry unreachable).  Rebuilding the full
scientific/ML stack locally is slow, so it is never the default path.

Used by ``run_trial.py``.  The function is idempotent and cheap when
the image is cached (a single ``docker image inspect`` call).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


__all__ = ["BASE_IMAGE_TAG", "ensure_base_image"]


# Registry-qualified tag.  Per-task Dockerfiles `FROM` this exact ref;
# Harbor's Singularity provider pulls it via `docker://`; Docker pulls
# it when the task's `docker_image` field is set (default flow).
BASE_IMAGE_TAG = "nihao0630/openfinai-base:v1"


def ensure_base_image(
    *,
    dockerfile: Path,
    build_context: Path,
    force_rebuild: bool = False,
    log_prefix: str = "[openfinai-base]",
) -> None:
    """Cache, pull, or build the shared base image.

    A forced rebuild skips cache and pull checks. Otherwise a local build is
    the fallback when pulling fails, and requires a valid Dockerfile.
    """
    if not force_rebuild:
        check = subprocess.run(
            ["docker", "image", "inspect", BASE_IMAGE_TAG],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            print(f"{log_prefix} {BASE_IMAGE_TAG} is cached")
            return

        # Default flow: pull the published image rather than rebuilding the
        # full ML stack locally. Inherit stdio so pull progress streams.
        print(
            f"{log_prefix} {BASE_IMAGE_TAG} not found locally; "
            "pulling from the registry ...",
            flush=True,
        )
        pull = subprocess.run(["docker", "pull", BASE_IMAGE_TAG])
        if pull.returncode == 0:
            print(f"{log_prefix} {BASE_IMAGE_TAG} pulled", flush=True)
            return
        print(
            f"{log_prefix} pull failed (exit {pull.returncode}; offline or "
            "registry unreachable); falling back to local build ...",
            flush=True,
        )

    # Local build: force_rebuild, or the pull above failed
    if not dockerfile.exists():
        raise RuntimeError(
            f"openfinai-base Dockerfile not found at {dockerfile}; "
            "cannot build locally (and the registry pull was unavailable)."
        )

    # Best-effort relative path; falls back to absolute when ``dockerfile``
    # is not under ``build_context``.
    try:
        df_display: Path | str = dockerfile.relative_to(build_context)
    except ValueError:
        df_display = dockerfile
    print(
        f"{log_prefix} {BASE_IMAGE_TAG} "
        f"{'force-rebuild requested' if force_rebuild else 'pull unavailable'}; "
        f"building from {df_display} "
        "(building the full ML stack; this can take several minutes) ...",
        flush=True,
    )
    # TTYs can inherit BuildKit output. Pipes need line-forwarded plain output
    # to avoid buffering the Docker-to-Python process chain.
    if sys.stdout.isatty():
        cmd = [
            "docker", "build",
            "-t", BASE_IMAGE_TAG,
            "-f", str(dockerfile),
            str(build_context),
        ]
        result = subprocess.run(cmd, cwd=str(build_context))
        returncode = result.returncode
    else:
        cmd = [
            "docker", "build",
            "--progress=plain",
            "-t", BASE_IMAGE_TAG,
            "-f", str(dockerfile),
            str(build_context),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(build_context),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        try:
            assert proc.stdout is not None  # for mypy; PIPE is set
            for line in proc.stdout:
                print(line, end="", flush=True)
        finally:
            returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"docker build of {BASE_IMAGE_TAG} failed "
            f"(exit {returncode}); see build output above."
        )
    print(f"{log_prefix} {BASE_IMAGE_TAG} built", flush=True)
