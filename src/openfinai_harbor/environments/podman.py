"""Run Harbor trial containers with rootless Podman in Kubernetes.

This provider avoids ``CAP_SYS_ADMIN`` by using user namespaces and shares the
pod network so trial containers can reach the host verifier.

Pattern:

Modeled on Harbor's Docker provider (long-running container), NOT on
Singularity (one-shot exec + in-container FastAPI server).  Each trial
gets its own container via::

    podman run -d --name <session> --network=host -v <data>:/data \\
        <image> sleep infinity

Subsequent operations use ``podman exec`` and ``podman cp``; cleanup stops and
removes the container. See ``deploy/k8s/README.md`` for deployment details.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


__all__ = ["PodmanEnvironment"]


_VALID_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def _sanitize_container_name(name: str) -> str:
    """Podman container names: [a-zA-Z0-9][a-zA-Z0-9_.-]*. Replace bad chars with '-'."""
    sanitized = _VALID_NAME_RE.sub("-", name)
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "h" + sanitized
    return sanitized[:128]


class PodmanEnvironment(BaseEnvironment):
    """Per-trial sandbox using rootless Podman.

    Container lifecycle:
    - ``start`` pulls the image (or builds locally if ``force_build``),
      sweeps stale containers from previous Pod incarnations (by label),
      then ``podman run -d`` a long-running ``sleep infinity`` container
      with the trial's mounts bound in. Each container is labelled with
      ``harbor-session=<sanitized_session_id>``.
    - All subsequent ops (`exec`, `upload_*`, `download_*`) target the
      named container.
    - ``stop`` does ``podman stop`` (+ optionally ``podman rm``).
    """

    _LABEL_KEY = "harbor-session"
    _SLEEP_CMD: tuple[str, ...] = ("sleep", "infinity")

    def __init__(
        self,
        *args: Any,
        podman_storage_root: str | None = None,
        podman_run_user: str | int | None = None,
        podman_force_pull: bool = False,
        **kwargs: Any,
    ) -> None:
        # Strip our kwargs before forwarding the rest to BaseEnvironment —
        # Harbor's factory passes through everything in `environment.kwargs`
        # plus its own constructor kwargs, so we must whitelist ours.
        self._podman_storage_root: str | None = podman_storage_root
        self._podman_run_user: str | int | None = podman_run_user
        self._podman_force_pull: bool = podman_force_pull
        super().__init__(*args, **kwargs)
        self._container_name = _sanitize_container_name(f"hb-{self.session_id}")
        self._started: bool = False

    # Required class-level metadata

    @staticmethod
    def type() -> str:
        # Third-party providers may return arbitrary identifiers; harbor's
        # EnvironmentType enum is for built-ins (base.py:282-292).
        return "podman"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Trials need additive /data mounts and networking to reach the host
        # verifier. Sandbox GPU passthrough is not supported here.
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        """Either a registry image or a local Dockerfile must be present.

        Mirrors ``DockerEnvironment._validate_definition`` semantics
        (docker.py:298-306): registry image preferred (every curated and
        generated task.toml in this repo now sets ``docker_image``), local
        Dockerfile is the force_build escape hatch.
        """
        if self.task_env_config.docker_image:
            return
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(
                "PodmanEnvironment requires either `[environment].docker_image` "
                f"in task.toml or a local `{dockerfile}`. Neither was found."
            )

    # Preflight (host-side capability checks)

    @classmethod
    def preflight(cls) -> None:
        """Run once at trainer-Pod startup. Surface fragility loudly.

        We *don't* raise on every issue — some signals (vfs fallback,
        missing /dev/fuse) are warnings, not blockers — but we print so
        the trainer log clearly shows what mode podman ended up in. Running
        this preflight in a throwaway Pod reproduces these checks cheaply
        before a real RL Job burns hours.
        """
        if not shutil.which("podman"):
            raise SystemExit(
                "podman binary not found on PATH. The K8s trainer Pod's "
                "init block should `apt-get install -y podman uidmap "
                "fuse-overlayfs slirp4netns` (see "
                "deploy/k8s/openfinai_rl_job.tpl.yml)."
            )

        # podman info — checks daemon-less binary actually runs and
        # exposes its storage driver. graphDriver=overlay (good) vs vfs
        # (catastrophic for image pull throughput) is the key signal.
        try:
            info = subprocess.run(
                ["podman", "info", "--format", "{{.Store.GraphDriverName}}"],
                capture_output=True, text=True, timeout=15,
            )
            driver = info.stdout.strip() if info.returncode == 0 else "?"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise SystemExit(f"`podman info` failed: {e}") from e
        if driver == "vfs":
            print(
                "[podman.preflight] WARNING: GraphDriver=vfs. Image pulls "
                "will be very slow and PVC usage will balloon. Investigate "
                "/dev/fuse availability + fuse-overlayfs install.",
                flush=True,
            )
        else:
            print(f"[podman.preflight] GraphDriver={driver}", flush=True)

        # /dev/fuse — fuse-overlayfs needs the device node.
        if not Path("/dev/fuse").exists():
            print(
                "[podman.preflight] WARNING: /dev/fuse not present. fuse-overlayfs "
                "will not work; podman will fall back to vfs.",
                flush=True,
            )

        # /proc/sys/user/max_user_namespaces > 0 — Pod can't change this.
        try:
            max_userns = int(
                Path("/proc/sys/user/max_user_namespaces").read_text().strip()
            )
            if max_userns <= 0:
                raise SystemExit(
                    "kernel.user.max_user_namespaces=0. Rootless Podman "
                    "cannot run on this node. Escalate to cluster admin."
                )
        except (FileNotFoundError, ValueError):
            print(
                "[podman.preflight] WARNING: could not read "
                "/proc/sys/user/max_user_namespaces; assuming OK.",
                flush=True,
            )

        # /etc/subuid + /etc/subgid — must have an entry for the current user.
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "root"
        for fname in ("/etc/subuid", "/etc/subgid"):
            try:
                text = Path(fname).read_text()
            except FileNotFoundError:
                print(
                    f"[podman.preflight] WARNING: {fname} not present; "
                    f"rootless mappings may fail.",
                    flush=True,
                )
                continue
            if not any(line.startswith(f"{user}:") for line in text.splitlines()):
                print(
                    f"[podman.preflight] WARNING: {fname} has no entry for "
                    f"{user!r}. Pod-init script should append "
                    f"'{user}:100000:65536' before the first trial.",
                    flush=True,
                )

    # Internal helpers

    def _podman_base(self) -> list[str]:
        """Argv prefix for every podman call (sets storage root if configured)."""
        cmd = ["podman"]
        if self._podman_storage_root:
            cmd.extend(["--root", self._podman_storage_root])
        return cmd

    async def _run_podman(
        self,
        argv: list[str],
        *,
        timeout_sec: int | None = None,
        check: bool = True,
        capture: bool = True,
    ) -> ExecResult:
        """Spawn a podman subprocess, await it, return ExecResult."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture else None,
            stderr=asyncio.subprocess.PIPE if capture else None,
        )
        try:
            if timeout_sec:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec
                )
            else:
                stdout_b, stderr_b = await proc.communicate()
        except asyncio.TimeoutError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.communicate(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
            raise RuntimeError(
                f"podman command timed out after {timeout_sec}s: "
                f"{' '.join(argv)}"
            )
        stdout = stdout_b.decode(errors="replace") if stdout_b else None
        stderr = stderr_b.decode(errors="replace") if stderr_b else None
        result = ExecResult(
            stdout=stdout, stderr=stderr, return_code=proc.returncode or 0
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"podman command failed (rc={result.return_code}): "
                f"{' '.join(argv)}\nstdout: {stdout}\nstderr: {stderr}"
            )
        return result

    # Lifecycle

    async def start(self, force_build: bool) -> None:
        # 1. Sweep stale containers from prior crashed Pods (same label).
        try:
            await self._run_podman(
                self._podman_base()
                + [
                    "rm",
                    "-f",
                    "--filter",
                    f"label={self._LABEL_KEY}={self._container_name}",
                ],
                check=False,
                timeout_sec=30,
            )
        except Exception as e:
            self.logger.debug(f"Orphan sweep failed (non-fatal): {e}")

        image = self.task_env_config.docker_image
        use_prebuilt = bool(image) and not force_build

        if use_prebuilt:
            # 2a. Pull from registry (idempotent — fast if already cached).
            pull_cmd = self._podman_base() + ["pull", image]
            if self._podman_force_pull:
                pull_cmd.insert(-1, "--policy=always")
            await self._run_podman(pull_cmd, timeout_sec=600)
        else:
            # 2b. Build the local Dockerfile.
            dockerfile = self.environment_dir / "Dockerfile"
            if not dockerfile.exists():
                raise FileNotFoundError(
                    f"force_build=True but {dockerfile} not found"
                )
            image = _sanitize_container_name(f"hb-img-{self.session_id}").lower()
            build_cmd = self._podman_base() + [
                "build",
                "-t",
                image,
                "-f",
                str(dockerfile),
                str(self.environment_dir),
            ]
            await self._run_podman(build_cmd, timeout_sec=1800)

        # 3. Compose `podman run -d` argv.
        run_argv: list[str] = self._podman_base() + [
            "run",
            "--detach",
            "--name",
            self._container_name,
            "--network=host",
            "--label",
            f"{self._LABEL_KEY}={self._container_name}",
            "--memory",
            f"{self.task_env_config.memory_mb}m",
            "--cpus",
            str(self.task_env_config.cpus),
        ]
        # Mounts: harbor's trial layer passes ServiceVolumeConfig dicts
        # (type/source/target/read_only). We only support `bind` for now —
        # `volume` and `image` would need named-volume creation upfront.
        for mount in self._mounts:
            if mount.get("type") != "bind":
                continue
            src = mount["source"]
            tgt = mount["target"]
            spec = f"{src}:{tgt}"
            if mount.get("read_only"):
                spec += ":ro"
            run_argv.extend(["-v", spec])
        # Optional user override (default is to trust image USER directive).
        if self._podman_run_user is not None:
            run_argv.extend(["--user", str(self._podman_run_user)])
        # cwd at sandbox start: prefer task's declared workdir, fallback /workspace.
        workdir = self.task_env_config.workdir or "/workspace"
        run_argv.extend(["-w", workdir])
        run_argv.append(image)
        run_argv.extend(self._SLEEP_CMD)

        await self._run_podman(run_argv, timeout_sec=120)
        self._started = True

        # 4. Pre-create writable mount targets inside the container with
        # permissive perms so the in-container agent UID can write to them
        # (logs/verifier, logs/agent, artifacts). Mirrors DockerEnvironment.
        writable_targets = [
            mount["target"]
            for mount in self._mounts
            if mount.get("type") == "bind" and not mount.get("read_only")
        ]
        if writable_targets:
            mk_cmd = "mkdir -p " + " ".join(
                shlex.quote(t) for t in writable_targets
            )
            chmod_cmd = "chmod 0777 " + " ".join(
                shlex.quote(t) for t in writable_targets
            )
            await self._run_podman(
                self._podman_base()
                + [
                    "exec",
                    "--user",
                    "0",
                    self._container_name,
                    "bash",
                    "-c",
                    f"{mk_cmd} && {chmod_cmd}",
                ],
                timeout_sec=30,
                check=False,
            )

    async def stop(self, delete: bool) -> None:
        if not self._started:
            return
        try:
            await self._run_podman(
                self._podman_base() + ["stop", "--time", "10", self._container_name],
                timeout_sec=30,
                check=False,
            )
        except Exception as e:
            self.logger.warning(f"podman stop failed (continuing): {e}")
        if delete:
            try:
                await self._run_podman(
                    self._podman_base()
                    + ["rm", "--force", self._container_name],
                    timeout_sec=30,
                    check=False,
                )
            except Exception as e:
                self.logger.warning(f"podman rm failed: {e}")
        self._started = False

    # Per-command exec + file transfer

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        merged_env = self._merge_env(env)

        argv: list[str] = self._podman_base() + ["exec"]
        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            argv.extend(["-w", effective_cwd])
        if merged_env:
            for k, v in merged_env.items():
                argv.extend(["-e", f"{k}={v}"])
        if user is not None:
            argv.extend(["--user", str(user)])
        argv.append(self._container_name)
        # Single shell so the command can use pipes/redirects.
        argv.extend(["bash", "-c", command])
        return await self._run_podman(
            argv, timeout_sec=timeout_sec, check=False
        )

    async def upload_file(
        self, source_path: Path | str, target_path: str
    ) -> None:
        await self._run_podman(
            self._podman_base()
            + [
                "cp",
                str(source_path),
                f"{self._container_name}:{target_path}",
            ],
            timeout_sec=300,
        )

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str
    ) -> None:
        # `podman cp` on a directory source copies recursively; harbor
        # expects directory-to-directory semantics so ensure target exists.
        await self.exec(
            f"mkdir -p {shlex.quote(target_dir)}",
            user="root" if self._podman_run_user is None else None,
            timeout_sec=30,
        )
        await self._run_podman(
            self._podman_base()
            + [
                "cp",
                str(source_dir),
                f"{self._container_name}:{target_dir}",
            ],
            timeout_sec=600,
        )

    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        await self._run_podman(
            self._podman_base()
            + [
                "cp",
                f"{self._container_name}:{source_path}",
                str(target_path),
            ],
            timeout_sec=300,
        )

    async def download_dir(
        self, source_dir: str, target_dir: Path | str
    ) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self._run_podman(
            self._podman_base()
            + [
                "cp",
                f"{self._container_name}:{source_dir}",
                str(target),
            ],
            timeout_sec=600,
        )
