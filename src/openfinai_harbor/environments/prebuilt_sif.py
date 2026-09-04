"""Use a pre-built native SIF with Harbor's Singularity provider.

This supports incompatible architectures or hosts that cannot pull OCI images.
Build the SIF from
``deploy/singularity/openfinai_sandbox.def``::

    apptainer build /path/to/sif_cache/openfinai-sandbox.sif \
        deploy/singularity/openfinai_sandbox.def

and select this provider from a Harbor trial config (see
``config/harbor_trial_rl_singularity.yaml``)::

    environment:
      import_path: openfinai_harbor.environments.prebuilt_sif:PrebuiltSifSingularityEnvironment
      kwargs:
        sif_path: /path/to/sif_cache/openfinai-sandbox.sif

If ``sif_path`` is omitted, ``OPENFINAI_SANDBOX_SIF`` is used. The host ML
environment may be mounted at ``/opt/conda``; Harbor retains lifecycle and
mount handling.
"""

from __future__ import annotations

import os
from pathlib import Path

from harbor.environments.singularity.singularity import SingularityEnvironment

__all__ = ["PrebuiltSifSingularityEnvironment", "SIF_PATH_ENV"]

#: Environment variable consulted when the trial config does not pass
#: ``sif_path`` explicitly.
SIF_PATH_ENV = "OPENFINAI_SANDBOX_SIF"


class PrebuiltSifSingularityEnvironment(SingularityEnvironment):
    """Singularity provider that runs a pre-built ``.sif`` instead of pulling.

    Args:
        sif_path: Absolute path to the pre-built SIF. Falls back to
            ``$OPENFINAI_SANDBOX_SIF`` when not given.
        **kwargs: Forwarded to :class:`SingularityEnvironment`.
    """

    def __init__(self, *args, sif_path: str | os.PathLike | None = None, **kwargs):
        resolved = sif_path or os.environ.get(SIF_PATH_ENV, "").strip()
        if not resolved:
            raise ValueError(
                "PrebuiltSifSingularityEnvironment needs the path to a pre-built "
                "SIF: pass `sif_path` under environment.kwargs in the trial "
                f"config, or set ${SIF_PATH_ENV}. Build one with "
                "`apptainer build <out>.sif deploy/singularity/openfinai_sandbox.def`."
            )
        path = Path(resolved).expanduser()
        if path.suffix != ".sif":
            raise ValueError(f"expected a .sif image path, got {path!s}")
        if not path.is_file():
            raise FileNotFoundError(f"pre-built sandbox SIF not found: {path!s}")
        self._prebuilt_sif = str(path)
        super().__init__(*args, **kwargs)

    @property
    def _docker_image(self) -> str:
        return self._prebuilt_sif
