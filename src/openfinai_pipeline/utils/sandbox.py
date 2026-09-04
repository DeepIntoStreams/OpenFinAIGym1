"""Static checks and timeout-bounded subprocess tests for generated code.

The subprocess contains interpreter state but is not an OS security boundary.
"""

import ast
import importlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from openfinai_pipeline.utils.logging import log_detail

logger = logging.getLogger(__name__)

# Modules blocked in generated task code; HTTP clients remain allowed.
_BLOCKED_MODULES = frozenset(
    [
        "subprocess",
        "multiprocessing",
        "socket",
        "shutil",
        "ctypes",
        "importlib",
    ]
)

# ``os`` paths are allowed, but destructive and shell calls are not.
_BLOCKED_CALLS = frozenset(
    [
        ("os", "system"),
        ("os", "popen"),
        ("os", "execv"),
        ("os", "execve"),
        ("shutil", "rmtree"),
        ("shutil", "copytree"),
    ]
)

_VALIDATED_IMPORT_ROOTS = ("openfingym.", "openfinai_pipeline.")

# Runtime package installation breaks reproducibility and bloats reviewer logs.
_PKG_MANAGER_TOKENS = frozenset(
    [
        "pip",
        "pip3",
        "conda",
        "mamba",
        "micromamba",
        "apt",
        "apt-get",
        "brew",
        "npm",
        "yum",
        "dnf",
        "pacman",
        "zypper",
        "pkg",
        "choco",
        "winget",
    ]
)

# Package-manager commands passed to a shell executor.
_PKG_MANAGER_SHELL_RE = re.compile(
    r"\b(pip3?|conda|mamba|micromamba|apt(?:-get)?|brew|npm|yum|dnf|pacman|"
    r"zypper|pkg|choco|winget)\b\s+(install|uninstall|update|upgrade|add|"
    r"remove|env\s+create|env\s+update)",
    re.IGNORECASE,
)


def _argv_list_pkg_manager(list_node: ast.List) -> Optional[str]:
    """Find a direct or ``python -m`` package manager in literal argv."""
    string_tokens: list[str] = []
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            string_tokens.append(elt.value)
        else:
            # Preserve positions across dynamic argv elements.
            string_tokens.append("")
    if string_tokens and string_tokens[0] in _PKG_MANAGER_TOKENS:
        return string_tokens[0]
    for i, tok in enumerate(string_tokens):
        if tok == "-m" and i + 1 < len(string_tokens):
            nxt = string_tokens[i + 1]
            if nxt in _PKG_MANAGER_TOKENS:
                return nxt
    return None


def _is_shell_executor_call(call: ast.Call) -> bool:
    """Return whether a call executes a string through a shell."""
    func = call.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return False
    pair = (func.value.id, func.attr)
    if pair == ("os", "system") or pair == ("os", "popen"):
        return True
    if func.value.id == "subprocess" and func.attr in (
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
    ):
        for kw in call.keywords:
            if (
                kw.arg == "shell"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                return True
    return False


def _check_call_for_package_install(call: ast.Call) -> Optional[str]:
    """Return a violation string if ``call`` looks like a runtime pkg install."""
    if not call.args:
        return None
    first = call.args[0]
    # Inspect argv even when passed through a user-defined wrapper.
    if isinstance(first, ast.List):
        offender = _argv_list_pkg_manager(first)
        if offender is not None:
            return (
                f"Line {call.lineno}: disallowed runtime package install via "
                f"argv containing '{offender}' (Phase 2 download scripts must "
                f"use only packages already installed in the conda env)"
            )
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        if _is_shell_executor_call(call) and _PKG_MANAGER_SHELL_RE.search(first.value):
            return (
                f"Line {call.lineno}: disallowed runtime package install "
                f"command in string passed to a shell executor"
            )
    return None


class Sandbox:
    """AST checks plus timeout-bounded subprocess testing.

    Generated code is not executed in the orchestrator process, but the child
    still shares the user's OS privileges and accessible resources.
    """

    def __init__(self, timeout_seconds: int = 120) -> None:
        self.timeout_seconds = timeout_seconds
        self._module_exports: dict[str, Optional[Set[str]]] = {}

    def syntax_check(self, code: str) -> Tuple[bool, str]:
        """Return success or a formatted Python syntax error."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"

    def scan_dangerous_imports(self, code: str) -> List[str]:
        """Report blocked imports, dynamic execution, and dangerous calls."""
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _BLOCKED_MODULES:
                        violations.append(
                            f"Line {node.lineno}: blocked import '{alias.name}'"
                        )

            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in _BLOCKED_MODULES:
                    violations.append(
                        f"Line {node.lineno}: blocked 'from {node.module} import ...'"
                    )

            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
                    violations.append(
                        f"Line {node.lineno}: disallowed call to '{func.id}()'"
                    )
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        pair = (func.value.id, func.attr)
                        if pair in _BLOCKED_CALLS:
                            violations.append(
                                (
                                    f"Line {node.lineno}: disallowed call "
                                    f"'{func.value.id}.{func.attr}()'"
                                )
                            )

        return violations

    def scan_phase2_download_violations(self, code: str) -> List[str]:
        """Reject runtime package management in phase-two download scripts.

        This narrower pass permits process, archive, and network modules needed
        for downloads; :meth:`scan_dangerous_imports` applies stricter rules to
        generated task code.
        """
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "pip" or mod.startswith("pip."):
                    violations.append(
                        f"Line {node.lineno}: disallowed runtime package "
                        f"management import 'from {mod} import ...'"
                    )
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name == "pip" or name.startswith("pip."):
                        violations.append(
                            f"Line {node.lineno}: disallowed runtime package "
                            f"management import 'import {name}'"
                        )
                continue
            if isinstance(node, ast.Call):
                v = _check_call_for_package_install(node)
                if v is not None:
                    violations.append(v)

        return violations

    def _resolve_module_exports(self, module_path: str) -> Optional[Set[str]]:
        """Import a module and return its public names, or None."""
        if module_path in self._module_exports:
            return self._module_exports[module_path]
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            self._module_exports[module_path] = None
            return None
        if hasattr(mod, "__all__"):
            exports: Set[str] = set(mod.__all__)
        else:
            # Exclude names merely imported from another module.
            import types

            exports = set()
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                obj = getattr(mod, name, None)
                if isinstance(obj, type) or isinstance(obj, types.FunctionType):
                    defined_in = getattr(obj, "__module__", "")
                    if defined_in == module_path:
                        exports.add(name)
                        continue
                if not isinstance(obj, types.ModuleType):
                    defined_in = getattr(obj, "__module__", "")
                    if defined_in == module_path:
                        exports.add(name)
        self._module_exports[module_path] = exports
        return exports

    @staticmethod
    def _is_generated_task_module(module_path: str) -> bool:
        """True for imports from a generated task's own package (not yet installed)."""
        parts = module_path.split(".")
        if (
            len(parts) >= 3
            and parts[0] == "openfingym"
            and parts[1] == "tasks"
            and parts[2] != "base"
        ):
            return True
        if len(parts) >= 4 and parts[0] == "openfinai_pipeline" and parts[1] == "benchmark":
            return True
        return False

    def validate_openfingym_imports(self, code: str) -> List[str]:
        """Report unresolved OpenFinGym and pipeline imports."""
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_path = node.module or ""
                if not any(module_path.startswith(r) for r in _VALIDATED_IMPORT_ROOTS):
                    continue
                if self._is_generated_task_module(module_path):
                    continue

                available = self._resolve_module_exports(module_path)
                if available is None:
                    violations.append(
                        f"Line {node.lineno}: module '{module_path}' does not exist"
                    )
                    continue

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if alias.name in available:
                        continue
                    sub = f"{module_path}.{alias.name}"
                    if self._resolve_module_exports(sub) is not None:
                        continue
                    sorted_avail = ", ".join(sorted(available))
                    violations.append(
                        f"Line {node.lineno}: '{module_path}' does not export "
                        f"'{alias.name}'. Available: {sorted_avail}"
                    )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not any(
                        alias.name.startswith(r) for r in _VALIDATED_IMPORT_ROOTS
                    ):
                        continue
                    if self._is_generated_task_module(alias.name):
                        continue
                    if self._resolve_module_exports(alias.name) is None:
                        violations.append(
                            f"Line {node.lineno}: module '{alias.name}' does not exist"
                        )

        return violations

    def run_tests(
        self, staging_dir: Path, task_id: str | None = None
    ) -> Tuple[bool, str]:
        """Run the staged test file in a timeout-bounded pytest subprocess.

        Returns ``(passed, combined_output)``. ``task_id`` is used only for
        logging; the child runs in ``staging_dir`` with repository imports.
        """
        test_file = staging_dir / "test_task.py"
        if not test_file.exists():
            return False, f"test_task.py not found in {staging_dir}"

        if task_id is None:
            task_id = staging_dir.name

        # Include source and reward trees without requiring installation.
        repo_root = Path(__file__).resolve().parents[3]
        pipeline_root = repo_root / "src"
        rewards_root = repo_root / "data" / "knowledge_base" / "rewards"

        env = os.environ.copy()
        path_entries = [str(repo_root), str(pipeline_root), str(rewards_root)]
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            path_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(path_entries)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file.name,
            "-v",
            "--tb=short",
            "--no-header",
        ]

        logger.debug("Running tests: %s (task_id=%s)", " ".join(cmd), task_id)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(staging_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            if passed:
                log_detail(logger, "Tests PASSED in %s.", staging_dir)
            else:
                log_detail(
                    logger,
                    "Tests FAILED in %s (exit code %d).",
                    staging_dir,
                    result.returncode,
                )
            return passed, output

        except subprocess.TimeoutExpired:
            msg = (
                f"Test execution timed out after {self.timeout_seconds}s. "
                "The task may have an infinite loop or slow data download."
            )
            logger.warning(msg)
            return False, msg
