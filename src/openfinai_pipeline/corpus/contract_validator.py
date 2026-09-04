"""Validate generated reward modules against formal rules R1–R6.

- R1: parameters whose name is a canonical ctx key MUST NOT have a default.
- R2: parameters whose name is NOT a canonical ctx key (and not the ``name``
  Loss-base identifier kwarg) MUST have a literal default — they are
  hyperparameters.
- R3: ``forward()`` MUST NOT take a parameter whose name is in
  ``INIT_ONLY_CANONICAL_NAMES`` (gt-like data).
- R4: ``__init__`` MUST NOT take a parameter whose name is in
  ``FORWARD_ONLY_CANONICAL_NAMES`` (pred-like data).
- R5: exactly one Loss subclass per file.
- R6: torch only; no separate ``compute()`` method.
"""

from __future__ import annotations

import ast
from typing import Any

from openfinai_pipeline.prompts.canonical_ctx import (
    ALL_CANONICAL_NAMES,
    FORWARD_ONLY_CANONICAL_NAMES,
    INIT_ONLY_CANONICAL_NAMES,
    LOSS_IDENTIFIER_KWARG,
)


def validate_reward_contract(code: str) -> list[str]:
    """Return a list of contract violations; empty list means the code conforms.

    Args:
        code: The generated Python source for the reward module.

    Returns:
        Human-readable violation messages, one per failed check. Each message
        references its rule label (R1-R6) so the reviewer in round N+1 can act
        on the failure.
    """
    violations: list[str] = []

    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [
            f"R5: source could not be parsed (SyntaxError: {exc.msg} "
            f"at line {exc.lineno}). Fix the syntax before any other check."
        ]

    loss_classes = _find_loss_subclasses(tree)

    if not loss_classes:
        violations.append(
            "R5: no Loss subclass found. Define exactly one class that "
            "inherits from `Loss` (imported from `reward_bank`)."
        )
        return violations

    if len(loss_classes) > 1:
        names = ", ".join(c.name for c in loss_classes)
        violations.append(
            f"R5: found {len(loss_classes)} Loss subclasses ({names}); "
            "exactly one is allowed per file."
        )

    cls = loss_classes[0]
    init_node = _find_method(cls, "__init__")
    forward_node = _find_method(cls, "forward")

    init_params = _signature_params(init_node) if init_node else {}
    forward_params = _signature_params(forward_node) if forward_node else {}

    if forward_node is None:
        violations.append(
            f"R5: class `{cls.name}` does not define `forward()`. Override "
            "forward(self, ...) returning a scalar torch.Tensor."
        )

    if _find_method(cls, "compute") is not None:
        violations.append(
            f"R6: class `{cls.name}` defines a separate `compute()` method. "
            "The Loss base has no forward->compute wrapper; do all reduction "
            "inside `forward()` and return the scalar tensor directly."
        )

    for param, info in init_params.items():
        if param == LOSS_IDENTIFIER_KWARG:
            continue
        is_canonical = param in ALL_CANONICAL_NAMES
        has_default = info["has_default"]

        if param in FORWARD_ONLY_CANONICAL_NAMES:
            violations.append(
                f"R4: `__init__` declares parameter `{param}`, which is a "
                "forward-only canonical name (carries agent output). The "
                "evaluator caches the metric instance after the first "
                "score() call, so a pred bound at init would freeze on the "
                f"first call's data. Move `{param}` to `forward()` instead."
            )
            continue

        if is_canonical and has_default:
            violations.append(
                f"R1: `__init__` parameter `{param}` is a canonical ctx-key "
                "name and MUST NOT have a default value. Canonical names "
                "carry essential ctx tensors and are always required. "
                f"Remove the default from `{param}`."
            )
        elif not is_canonical and not has_default:
            hint = (
                "Either rename it to a canonical name "
                f"({sorted(INIT_ONLY_CANONICAL_NAMES)}) so it binds from ctx, "
                "or give it a literal default value to make it a "
                "hyperparameter."
            )
            violations.append(
                f"R2: `__init__` parameter `{param}` is non-canonical and "
                f"has no default. Non-canonical params must be tunable "
                f"hyperparameters with literal defaults. {hint}"
            )

    for param, info in forward_params.items():
        is_canonical = param in ALL_CANONICAL_NAMES
        has_default = info["has_default"]

        if param in INIT_ONLY_CANONICAL_NAMES:
            violations.append(
                f"R3: `forward()` declares parameter `{param}`, which is an "
                "init-only canonical name (carries reference / ground-truth "
                "data). The agent must not see ground truth at scoring "
                f"time. Capture `{param}` in `__init__` instead."
            )
            continue

        if is_canonical and has_default:
            violations.append(
                f"R1: `forward()` parameter `{param}` is a canonical ctx-key "
                "name and MUST NOT have a default value. "
                f"Remove the default from `{param}`."
            )
        elif not is_canonical and not has_default:
            violations.append(
                f"R2: `forward()` parameter `{param}` is non-canonical and "
                f"has no default. Forward params should be canonical agent-"
                f"output names ({sorted(FORWARD_ONLY_CANONICAL_NAMES)}). If "
                "you genuinely need an extra knob at forward time, give it a "
                "literal default."
            )

    return violations


def _find_loss_subclasses(tree: ast.AST) -> list[ast.ClassDef]:
    """Find every ClassDef whose base list mentions a class named ``Loss``.

    Catches both ``class X(Loss)`` and ``class X(reward_bank.Loss)``.
    """
    found: list[ast.ClassDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "Loss":
                found.append(node)
                break
    return found


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node  # type: ignore[return-value]
    return None


def _signature_params(fn: ast.FunctionDef) -> dict[str, dict[str, Any]]:
    """Return ``{param_name: {"has_default": bool}}`` for a FunctionDef.

    Skips ``self`` and varargs/kwargs. Combines positional-only,
    positional-or-keyword, and keyword-only parameters.
    """
    out: dict[str, dict[str, Any]] = {}
    args = fn.args

    pos_args = list(args.posonlyargs) + list(args.args)
    n_defaults = len(args.defaults)
    n_pos = len(pos_args)
    for idx, arg in enumerate(pos_args):
        if arg.arg == "self":
            continue
        has_default = idx >= (n_pos - n_defaults)
        out[arg.arg] = {"has_default": has_default}

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out[arg.arg] = {"has_default": default is not None}

    return out
