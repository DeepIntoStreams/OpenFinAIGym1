"""SFT training subpackage.

Module map:
* :mod:`.driver` — unified ``train(config, mode=...)`` entry point.
* :mod:`.config` — SkyRL train-config builder + per-run ``SFTRuntimeConfig``.
* :mod:`.dataset` — chat-history → ``SFTExample`` rendering + collation.
* :mod:`.loop` — training loop, policy-dispatch factory, checkpoint I/O.
* :mod:`.manifest` — git SHA, config hash, system-prompt snapshot.
* :mod:`.ray_setup` — Ray init + SkyRL/Ray compatibility patch.
* :mod:`.policy` — single-process (no Ray/FSDP) policy dispatch.
* :mod:`.tokenization` — chat-template helpers and ``parse_cli``.

Callers import from the submodules directly; this package has no
implicit re-export wall.
"""
