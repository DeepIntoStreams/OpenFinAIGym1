"""Bundle each evaluator's reward classes and dependencies into one module."""

from __future__ import annotations

import ast
import builtins
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "BundleResult",
    "RewardBundlerError",
    "bundle_used_rewards",
]


class RewardBundlerError(RuntimeError):
    """Raised when a self-contained reward module cannot be produced."""


@dataclass
class BundleResult:
    """Bundle path, included names, and their source files."""

    out_path: Path
    classes_bundled: list[str] = field(default_factory=list)
    helpers_bundled: list[str] = field(default_factory=list)
    sources_used: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _SourceModule:
    """Parsed view of a single source file we may extract from."""

    path: Path
    tree: ast.Module
    # Top-level name -> defining node.
    top_level_defs: dict[str, ast.stmt] = field(default_factory=dict)
    # Import alias -> full import statement; rendering deduplicates statements.
    imports: dict[str, ast.stmt] = field(default_factory=dict)


def bundle_used_rewards(
    *,
    curated_imports: list[dict[str, Any]],
    generated_imports: list[dict[str, Any]],
    curated_root: Path,
    generated_root: Path | None,
    out_path: Path,
) -> BundleResult:
    """Write requested reward classes and transitive dependencies to one file.

    Raises ``RewardBundlerError`` for missing sources, unresolved names, or
    conflicting definitions.
    """
    if not curated_imports and not generated_imports:
        raise RewardBundlerError(
            "no rewards to bundle — neither curated nor generated imports were "
            "provided. assemble_evaluator should reject this earlier."
        )

    # Parse requested sources.
    sources: dict[Path, _SourceModule] = {}
    for entry in curated_imports:
        src_path = (curated_root / f"{entry['module_name']}.py").resolve()
        if src_path not in sources:
            sources[src_path] = _parse_source(src_path)
    for entry in generated_imports:
        if generated_root is None:
            raise RewardBundlerError(
                "generated_imports were provided but generated_root is None"
            )
        src_path = (generated_root / f"{entry['module_name']}.py").resolve()
        if src_path not in sources:
            sources[src_path] = _parse_source(src_path)

    # Generated-only bundles still import ``Loss`` from ``reward_bank``.
    # Parse the bank unconditionally so dependency collection can inline that
    # base; name-driven traversal still omits unrelated reward classes.
    curated_bank_path = (curated_root / "reward_bank.py").resolve()
    if curated_bank_path.exists() and curated_bank_path not in sources:
        sources[curated_bank_path] = _parse_source(curated_bank_path)

    # Build the global name registry.
    name_to_source: dict[str, _SourceModule] = {}
    for src in sources.values():
        for name in src.top_level_defs:
            if name in name_to_source and name_to_source[name].path != src.path:
                # Identical duplicate definitions are safe; conflicts are not.
                existing_src = name_to_source[name]
                a = ast.unparse(existing_src.top_level_defs[name])
                b = ast.unparse(src.top_level_defs[name])
                if a != b:
                    raise RewardBundlerError(
                        f"class/helper name collision: {name!r} is defined in "
                        f"both {existing_src.path} and {src.path} with "
                        "different bodies. Rename one to disambiguate."
                    )
                continue
            name_to_source[name] = src

    # Seed the dependency walk and verify requested classes exist.
    requested: list[tuple[str, _SourceModule]] = []
    for entry in curated_imports + generated_imports:
        cls_name = entry["class_name"]
        if cls_name not in name_to_source:
            raise RewardBundlerError(
                f"requested class {cls_name!r} not found in any provided "
                "source file."
            )
        requested.append((cls_name, name_to_source[cls_name]))

    # Fixed-point dependency walk.
    kept_defs: dict[str, tuple[_SourceModule, ast.stmt]] = {}
    needed_imports: dict[str, ast.stmt] = {}  # alias -> stmt
    worklist: list[tuple[str, _SourceModule]] = list(requested)

    while worklist:
        name, src = worklist.pop()
        if name in kept_defs:
            continue
        if name not in src.top_level_defs:
            owning = name_to_source.get(name)
            if owning is None:
                # Parameters, locals, and builtins need no module-level resolution.
                continue
            src = owning
        node = src.top_level_defs[name]
        kept_defs[name] = (src, node)

        for ref in _collect_unresolved_names(node):
            if ref in kept_defs:
                continue
            if ref in name_to_source:
                worklist.append((ref, name_to_source[ref]))
            elif ref in src.imports:
                needed_imports[ref] = src.imports[ref]
            # Other names are presumed parameters, locals, or builtins.

    # Render output.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render_module(
        kept_defs=kept_defs,
        needed_imports=needed_imports,
        sources=sources,
    )
    out_path.write_text(rendered, encoding="utf-8")

    # Build telemetry.
    classes = sorted(
        n for n, (_src, node) in kept_defs.items() if isinstance(node, ast.ClassDef)
    )
    helpers = sorted(
        n for n, (_src, node) in kept_defs.items() if not isinstance(node, ast.ClassDef)
    )
    sources_used: dict[str, list[str]] = {}
    for name, (src, _node) in kept_defs.items():
        sources_used.setdefault(str(src.path), []).append(name)
    for src_path in sources_used:
        sources_used[src_path].sort()

    return BundleResult(
        out_path=out_path,
        classes_bundled=classes,
        helpers_bundled=helpers,
        sources_used=sources_used,
    )


def _parse_source(path: Path) -> _SourceModule:
    """Parse a Python source file and index its top-level definitions."""
    if not path.exists():
        raise RewardBundlerError(f"reward source not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise RewardBundlerError(f"failed to parse {path}: {exc}") from exc

    src = _SourceModule(path=path, tree=tree)
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            src.top_level_defs[node.name] = node
        elif isinstance(node, ast.Assign):
            # Register each simple name target.
            for target in node.targets:
                if isinstance(target, ast.Name):
                    src.top_level_defs[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            src.top_level_defs[node.target.id] = node
        elif isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.asname or alias.name.split(".")[0]
                src.imports[key] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                key = alias.asname or alias.name
                src.imports[key] = node
        elif isinstance(node, ast.Try):
            # Retain the whole guarded import block.
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    for alias in inner.names:
                        key = alias.asname or alias.name.split(".")[0]
                        src.imports[key] = node  # whole try-block, not the inner Import
                elif isinstance(inner, ast.ImportFrom):
                    for alias in inner.names:
                        key = alias.asname or alias.name
                        src.imports[key] = node

    return src


# Ignore builtins and names guaranteed by the reward runtime.
_SKIP_NAMES: frozenset[str] = frozenset(dir(builtins)) | frozenset(
    {"self", "cls", "kwargs", "args", "_unused_legacy"}
)


def _collect_unresolved_names(node: ast.AST) -> set[str]:
    """Collect names not bound in a local scope."""
    collector = _FreeNameCollector()
    collector.visit(node)
    return {n for n in collector.free_names if n not in _SKIP_NAMES}


class _FreeNameCollector(ast.NodeVisitor):
    """Collect free names using the limited scopes needed by reward sources."""

    def __init__(self) -> None:
        self.free_names: set[str] = set()
        self._bound_stack: list[set[str]] = [set()]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_function_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        bound = {arg.arg for arg in self._all_args(node.args)}
        self._bound_stack.append(self._bound_stack[-1] | bound)
        try:
            self.generic_visit(node)
        finally:
            self._bound_stack.pop()

    def _enter_function_scope(self, node: Any) -> None:
        bound: set[str] = {arg.arg for arg in self._all_args(node.args)}
        # Assignment targets inside the function are locally bound.
        bound |= self._collect_locally_assigned(node)
        self._bound_stack.append(self._bound_stack[-1] | bound)
        try:
            self.generic_visit(node)
        finally:
            self._bound_stack.pop()

    @staticmethod
    def _all_args(args: ast.arguments) -> list[ast.arg]:
        out: list[ast.arg] = []
        out.extend(args.posonlyargs or [])
        out.extend(args.args or [])
        out.extend(args.kwonlyargs or [])
        if args.vararg is not None:
            out.append(args.vararg)
        if args.kwarg is not None:
            out.append(args.kwarg)
        return out

    @staticmethod
    def _collect_locally_assigned(func_node: Any) -> set[str]:
        names: set[str] = set()
        for inner in ast.walk(func_node):
            if isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets: Iterable[ast.AST]
                if isinstance(inner, ast.Assign):
                    targets = inner.targets
                else:
                    targets = [inner.target]
                for tgt in targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
            elif isinstance(inner, ast.For) and isinstance(inner.target, ast.Name):
                names.add(inner.target.id)
            elif isinstance(inner, ast.comprehension):
                for sub in ast.walk(inner.target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        return names

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class assignments must not hide names referenced by methods.
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self._bound_stack[-1]:
                self.free_names.add(node.id)


_HEADER = '''\
"""Auto-generated by openfinai_pipeline.benchmark.install.reward_bundler.

Self-contained subset of the curated reward bank plus any auto-generated
rewards used by the assembled evaluator for THIS task. Do not edit by
hand — re-bundling will overwrite changes.

The bundled set is intentionally minimal: only the classes the
evaluator imports plus their transitively-referenced module-level
helpers. The ``Loss`` base from the curated bank is included when any
subclass is. ``TradingReward`` and friends are NOT included — the
auto-pipeline never assembles trading rewards into a generated task.
"""
'''


def _render_module(
    *,
    kept_defs: dict[str, tuple[_SourceModule, ast.stmt]],
    needed_imports: dict[str, ast.stmt],
    sources: dict[Path, _SourceModule],
) -> str:
    """Render imports and definitions in deterministic dependency order."""
    seen_import_src: set[str] = set()
    rendered_imports: list[str] = []
    for stmt in needed_imports.values():
        src = ast.unparse(stmt)
        if src in seen_import_src:
            continue
        seen_import_src.add(src)
        rendered_imports.append(src)
    rendered_imports.sort()

    # Preserve definition order within each source.
    by_source: dict[Path, list[ast.stmt]] = {}
    for _name, (src, node) in kept_defs.items():
        by_source.setdefault(src.path, []).append(node)
    for path in by_source:
        by_source[path].sort(key=lambda n: getattr(n, "lineno", 0))

    def _source_sort_key(p: Path) -> tuple[int, str]:
        parts = p.parts
        is_generated = any(part == "generated" for part in parts)
        return (1 if is_generated else 0, str(p))

    ordered_sources = sorted(by_source.keys(), key=_source_sort_key)
    rendered_defs: list[str] = []
    for path in ordered_sources:
        for node in by_source[path]:
            rendered_defs.append(ast.unparse(node))

    parts = [_HEADER]
    if rendered_imports:
        parts.append("\n".join(rendered_imports))
    parts.append("\n\n".join(rendered_defs))
    return "\n\n\n".join(p for p in parts if p) + "\n"
