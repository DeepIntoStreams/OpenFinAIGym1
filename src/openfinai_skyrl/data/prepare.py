"""Unified SFT dataset preparation: ATIF trajectories -> task-split DatasetDict."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from openfinai_skyrl.data.splitting import (
    DEFAULT_TEST_TASKS,
    CorpusSplitManifest,
    build_train_only_corpus,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        required=True,
        help="single: one (system,user,assistant<python>) row per trial. "
        "multi: one row per trial, full history (system + all user/assistant turns).",
    )
    parser.add_argument(
        "--jobs-dir",
        default="jobs",
        help="Harbor jobs root (or a single trial directory).",
    )
    parser.add_argument(
        "--job-prefix",
        default=None,
        help="Optional substring filter — only include jobs whose dir starts with this prefix.",
    )
    parser.add_argument(
        "--tasks-root",
        default="tasks",
        help="Task directory used to canonicalise task names.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[3] / "data" / "run_output" / "experiments-sft"),
        help="Experiment output root. Splits land under <root>/data, reports under <root>/reports.",
    )
    parser.add_argument(
        "--system-prompt-path",
        default=None,
        help=(
            "Optional domain-prompt file used to RECOMPOSE the system "
            "message for every SFT row. Default reads each trial's "
            "trajectory system step verbatim (canonical path)."
        ),
    )
    # Rewards are loss-scaled (lower = better). Both knobs drop rows with
    # reward=None. Use --reward-max for an absolute ceiling; use
    # --top-k-lowest-reward for a per-task ranking cap.
    parser.add_argument(
        "--reward-max",
        type=float,
        default=None,
        help="Keep rows with reward <= REWARD_MAX (loss-scaled: lower=better).",
    )
    parser.add_argument(
        "--top-k-lowest-reward",
        type=int,
        default=None,
        help="Per task, keep only the N rows with the smallest reward; ties by trial_name.",
    )
    parser.add_argument(
        "--test-tasks",
        nargs="*",
        default=None,
        help="Override DEFAULT_TEST_TASKS.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional yaml supplying defaults (currently: top_k_lowest_reward).",
    )
    return parser.parse_args(argv)


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return raw if isinstance(raw, dict) else {}


def _iter_trial_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Jobs directory does not exist: {root}")
    if (root / "agent").exists():
        return [root]
    trials: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and (path / "agent").exists():
            trials.append(path)
    return trials


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_trajectory(trial_dir: Path) -> dict | None:
    return _read_json(trial_dir / "agent" / "trajectory.json")


def _read_conversation(trial_dir: Path) -> list[dict] | None:
    payload = _read_json(trial_dir / "agent" / "conversation.json")
    return payload if isinstance(payload, list) else None


# Two response-parser contracts: ReActFinAgent uses <python>...</python> tags,
# SingleShotLLMAgent uses fenced markdown. Extractor tries tag first, falls
# back to fence so single-shot trajectories aren't dropped.
_PYTHON_TAG = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL | re.IGNORECASE)
_PYTHON_FENCE_LANG = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_PYTHON_FENCE_BARE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def _extract_python_blocks(text: str) -> list[str]:
    """Return all ``<python>`` blocks; falls back to ``python`` / bare fences."""
    if not isinstance(text, str) or not text.strip():
        return []
    tag_hits = [match.strip() for match in _PYTHON_TAG.findall(text) if match.strip()]
    if tag_hits:
        return tag_hits
    fence_hits = [match.strip() for match in _PYTHON_FENCE_LANG.findall(text) if match.strip()]
    if fence_hits:
        return fence_hits
    return [match.strip() for match in _PYTHON_FENCE_BARE.findall(text) if match.strip()]


def _content_has_code_block(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return ("<python>" in text) or ("```" in text)


def _looks_like_acceptance_observation(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lower()
    return (
        "candidate accepted" in normalized
        or "accepted and saved to /workspace/train.py" in normalized
        or "file replaced successfully" in normalized
    )


def _first_user_message(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return None


def conversation_to_messages(conversation: list[dict]) -> list[dict]:
    """Convert raw ``conversation.json`` to chat messages; ensures trailing assistant."""
    messages: list[dict] = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": content})

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages


def atif_to_messages(trajectory: dict) -> list[dict]:
    """Convert an ATIF v1.6 trajectory to chat messages.

    Agent step's raw ``message`` is used verbatim (both OpenFinGym agents
    store the full ``<reasoning>...</reasoning>\\n<python>...</python>``
    there; we deliberately don't re-render the parsed duplicates).
    Observations are wrapped in ``<information>...</information>`` to
    match ReActFinAgent's inference-time envelope. Ensures trailing
    assistant so loss is computable.
    """
    steps = trajectory.get("steps") or []
    messages: list[dict] = []

    for step in steps:
        source = step.get("source")
        message = step.get("message", "")
        if isinstance(message, list):
            text_parts = [
                p.get("text", "") for p in message if isinstance(p, dict) and p.get("type") == "text"
            ]
            message = "\n".join(t for t in text_parts if t)

        if source == "system":
            if isinstance(message, str) and message.strip():
                messages.append({"role": "system", "content": message.strip()})
            continue

        if source == "user":
            if message and message.strip():
                messages.append({"role": "user", "content": message.strip()})
            continue

        if source == "agent":
            # Use raw message verbatim; composing from reasoning_content +
            # tool_calls produces triple-rendered content with a stray
            # <tool_call> envelope neither agent parses.
            assistant_content = message.strip() if isinstance(message, str) else ""
            if not assistant_content:
                # Fallback for non-OpenFinGym producers that leave `message` empty.
                parts: list[str] = []
                reasoning = step.get("reasoning_content")
                if reasoning:
                    parts.append(f"<reasoning>{reasoning}</reasoning>")
                for tc in step.get("tool_calls") or []:
                    args = tc.get("arguments") if isinstance(tc, dict) else None
                    code = args.get("code") if isinstance(args, dict) else None
                    if isinstance(code, str) and code.strip():
                        parts.append(f"<python>\n{code.strip()}\n</python>")
                assistant_content = "\n".join(parts).strip()
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            obs = step.get("observation") or {}
            results = obs.get("results") or []
            if results:
                obs_texts: list[str] = []
                for r in results:
                    c = r.get("content")
                    if isinstance(c, str):
                        obs_texts.append(c)
                    elif isinstance(c, list):
                        for p in c:
                            if isinstance(p, dict) and p.get("type") == "text":
                                obs_texts.append(p.get("text", ""))
                obs_text = "\n".join(t for t in obs_texts if t and t.strip()).strip()
                if obs_text:
                    if "<information>" not in obs_text:
                        obs_text = f"<information>\n{obs_text}\n</information>"
                    messages.append({"role": "user", "content": obs_text})

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages


def extract_single_turn_final_code_from_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Reduce a multi-turn message list to a single (user, assistant<python>) pair.

    Picks the last ``<python>`` block confirmed accepted by a following
    ``<information>`` user message, else the last ``<python>`` block.
    Returns None if no ``<python>`` block exists.
    """
    prompt = _first_user_message(messages)
    if not prompt:
        return None

    assistant_turns = 0
    python_turns: list[dict[str, Any]] = []
    answer_turns = 0

    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_turns += 1
        content = str(message.get("content", ""))
        if "<answer>" in content:
            answer_turns += 1
        python_blocks = _extract_python_blocks(content)
        if not python_blocks:
            continue

        next_user_content: str | None = None
        for later in messages[idx + 1 :]:
            if later.get("role") == "user":
                next_user_content = str(later.get("content", ""))
                break

        python_turns.append(
            {
                "assistant_turn_index": assistant_turns - 1,
                "python_source": python_blocks[-1],
                "full_content": content,
                "next_user_content": next_user_content,
                "accepted": _looks_like_acceptance_observation(next_user_content or ""),
            }
        )

    if not python_turns:
        return None

    accepted_turns = [turn for turn in python_turns if turn["accepted"]]
    if accepted_turns:
        selected = accepted_turns[-1]
        selection_rule = "last_accepted_python"
        extraction_status = "accepted"
    else:
        selected = python_turns[-1]
        selection_rule = "fallback_last_python"
        extraction_status = "fallback_last_python"

    # Wrap in a fenced block when bare so the SFT target matches
    # SingleShotLLMAgent's parser (fence only, no <python> tag).
    assistant_content = str(selected["full_content"]).strip()
    if not _content_has_code_block(assistant_content):
        assistant_content = f"```python\n{selected['python_source']}\n```"
    return {
        "user_content": prompt,
        "assistant_content": assistant_content,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "extraction_status": extraction_status,
        "python_turn_index": int(selected["assistant_turn_index"]),
        "selection_rule": selection_rule,
        "assistant_turns_total": assistant_turns,
        "assistant_python_turns": len(python_turns),
        "assistant_answer_turns": answer_turns,
    }


def extract_single_turn_final_code_from_conversation(
    conversation: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return extract_single_turn_final_code_from_messages(conversation_to_messages(conversation))


def extract_single_turn_final_code_from_trajectory(
    trajectory: dict[str, Any],
) -> dict[str, Any] | None:
    """ATIF-trajectory variant; uses the raw ``observation`` text for acceptance
    detection, which is more reliable than post-processed message text.
    """
    steps = trajectory.get("steps") or []
    prompt: str | None = None
    assistant_turns = 0
    python_turns: list[dict[str, Any]] = []
    answer_turns = 0

    for step in steps:
        source = step.get("source")
        message = step.get("message", "")
        if isinstance(message, list):
            text_parts = [
                p.get("text", "") for p in message if isinstance(p, dict) and p.get("type") == "text"
            ]
            message = "\n".join(t for t in text_parts if t)

        if source == "user" and prompt is None:
            content = str(message).strip()
            if content:
                prompt = content
            continue

        if source != "agent":
            continue

        assistant_turns += 1
        content = str(message)
        if "<answer>" in content:
            answer_turns += 1
        python_blocks = _extract_python_blocks(content)
        if not python_blocks:
            # Fallback for non-OpenFinGym producers with code only in tool_calls.
            for tc in step.get("tool_calls") or []:
                args = tc.get("arguments") if isinstance(tc, dict) else None
                if isinstance(args, dict):
                    code = args.get("code")
                    if isinstance(code, str) and code.strip():
                        python_blocks = [code.strip()]
                        break
        if not python_blocks:
            continue

        obs = step.get("observation") or {}
        results = obs.get("results") or []
        obs_text = "\n".join(
            str(result.get("content", "")).strip()
            for result in results
            if isinstance(result, dict) and str(result.get("content", "")).strip()
        )
        # Use a fenced wrapper when reconstructing from parsed views
        # (SingleShotLLMAgent's parser only accepts fences).
        if _content_has_code_block(content):
            full_content = content.strip()
        else:
            parts: list[str] = []
            reasoning = step.get("reasoning_content")
            if reasoning:
                parts.append(f"<reasoning>{reasoning}</reasoning>")
            if content.strip():
                parts.append(content.strip())
            parts.append(f"```python\n{python_blocks[-1]}\n```")
            full_content = "\n".join(parts).strip()
        python_turns.append(
            {
                "assistant_turn_index": assistant_turns - 1,
                "python_source": python_blocks[-1],
                "full_content": full_content,
                "accepted": _looks_like_acceptance_observation(obs_text),
            }
        )

    if not prompt or not python_turns:
        return None

    accepted_turns = [turn for turn in python_turns if turn["accepted"]]
    if accepted_turns:
        selected = accepted_turns[-1]
        selection_rule = "last_accepted_python"
        extraction_status = "accepted"
    else:
        selected = python_turns[-1]
        selection_rule = "fallback_last_python"
        extraction_status = "fallback_last_python"

    assistant_content = str(selected["full_content"]).strip()
    return {
        "user_content": prompt,
        "assistant_content": assistant_content,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "extraction_status": extraction_status,
        "python_turn_index": int(selected["assistant_turn_index"]),
        "selection_rule": selection_rule,
        "assistant_turns_total": assistant_turns,
        "assistant_python_turns": len(python_turns),
        "assistant_answer_turns": answer_turns,
    }


def _mode_prompt_addendum(mode: str) -> str:
    """Return MODE_PROMPT_ADDENDUM from the matching agent class."""
    if mode == "single":
        from openfinai_harbor.agents.single_shot_llm import SingleShotLLMAgent
        return SingleShotLLMAgent.MODE_PROMPT_ADDENDUM or ""
    if mode == "multi":
        from openfinai_harbor.agents.react_fin_agent import ReActFinAgent
        return ReActFinAgent.MODE_PROMPT_ADDENDUM or ""
    raise ValueError(f"Unknown mode: {mode!r}")


def _compose_system_prompt(domain_prompt: str, mode: str) -> str:
    """Mirror BaseFinAgent._compose_system_prompt so prompts byte-match the agent."""
    domain = domain_prompt.rstrip()
    addendum = _mode_prompt_addendum(mode)
    if not addendum:
        return domain
    return f"{domain}\n\n{addendum.lstrip()}"


def _compose_default_system_prompt(mode: str) -> str:
    """Last-resort default = DEFAULT_DOMAIN_PROMPT + MODE_PROMPT_ADDENDUM."""
    from openfinai_harbor.agents.base import BaseFinAgent
    return _compose_system_prompt(BaseFinAgent.DEFAULT_DOMAIN_PROMPT, mode)


def _load_override_system_prompt(system_prompt_path: Path | str, *, mode: str) -> str:
    """Read a caller-supplied domain-prompt file and compose with the mode addendum."""
    path = Path(system_prompt_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"--system-prompt-path={path} not found. Drop the flag to use the "
            f"system prompt recorded in each trial's trajectory (the canonical "
            f"path)."
        ) from exc
    if not text.strip():
        raise ValueError(f"system prompt file is empty: {path}")
    return _compose_system_prompt(text, mode)


def _system_prompt_from_trial(
    trajectory: dict | None,
    conversation: list[dict] | None,
) -> str | None:
    """Return the trial's recorded system message verbatim, or None."""
    if trajectory is not None:
        for step in trajectory.get("steps") or []:
            if step.get("source") != "system":
                continue
            msg = step.get("message", "")
            if isinstance(msg, list):
                msg = "\n".join(
                    p.get("text", "")
                    for p in msg
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    if conversation:
        for m in conversation:
            if m.get("role") != "system":
                continue
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


@dataclass
class TrialLoadResult:
    messages: list[dict[str, str]]
    task_name: str
    agent_name: str
    model_name: str | None
    reward: float | None
    source_kind: str
    job_name: str
    trial_name: str
    extraction_status: str | None = None
    python_turn_index: int | None = None
    selection_rule: str | None = None
    assistant_turns_total: int = 0
    assistant_python_turns: int = 0
    assistant_answer_turns: int = 0


def load_messages_from_trial(
    trial_dir: Path,
    *,
    mode: str,
    override_system_prompt: str | None = None,
) -> TrialLoadResult | None:
    """Load a single trial and return its SFT messages.

    System prompt resolution order: override > trajectory > composed default.
    """
    trajectory = _read_trajectory(trial_dir)
    conversation: list[dict] | None = None
    agent_name = "react-fin-agent"
    model_name: str | None = None

    if trajectory is not None:
        agent_info = trajectory.get("agent") or {}
        agent_name = agent_info.get("name", agent_name)
        model_name = agent_info.get("model_name")
        source_kind = "trajectory.json"
    else:
        conversation = _read_conversation(trial_dir)
        if conversation is None:
            return None
        source_kind = "conversation.json"

    if override_system_prompt is not None:
        system_prompt = override_system_prompt
    else:
        sp = _system_prompt_from_trial(trajectory, conversation)
        system_prompt = sp if sp else _compose_default_system_prompt(mode)

    # Lazy import to avoid a data/eval circular import at module load.
    from openfinai_skyrl.eval.common import read_reward
    reward = read_reward(trial_dir)
    # config.json's task path handles both __<uid>-suffix and <utc>/ layouts.
    task_name: str | None = None
    cfg = _read_json(trial_dir / "config.json")
    if isinstance(cfg, dict):
        task_cfg = cfg.get("task")
        if isinstance(task_cfg, dict):
            task_path = task_cfg.get("path")
            if isinstance(task_path, str) and task_path.strip():
                task_name = Path(task_path).name
    if not task_name:
        task_name = trial_dir.name.rsplit("__", 1)[0] if "__" in trial_dir.name else trial_dir.name

    extraction_status: str | None = None
    python_turn_index: int | None = None
    selection_rule: str | None = None
    assistant_turns_total = 0
    assistant_python_turns = 0
    assistant_answer_turns = 0

    if mode == "multi":
        if trajectory is not None:
            messages = atif_to_messages(trajectory)
        else:
            assert conversation is not None
            messages = conversation_to_messages(conversation)
        if not messages:
            return None
        # Drop pre-existing system rows; we prepend our resolved one ourselves.
        body = [m for m in messages if m.get("role") != "system"]
        if not body or body[-1].get("role") != "assistant":
            return None
        final_messages = [{"role": "system", "content": system_prompt}] + body
    elif mode == "single":
        if trajectory is not None:
            extracted = extract_single_turn_final_code_from_trajectory(trajectory)
        else:
            assert conversation is not None
            extracted = extract_single_turn_final_code_from_conversation(conversation or [])
        if extracted is None:
            return None
        extraction_status = str(extracted["extraction_status"])
        python_turn_index = int(extracted["python_turn_index"])
        selection_rule = str(extracted["selection_rule"])
        assistant_turns_total = int(extracted["assistant_turns_total"])
        assistant_python_turns = int(extracted["assistant_python_turns"])
        assistant_answer_turns = int(extracted["assistant_answer_turns"])
        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extracted["user_content"]},
            {"role": "assistant", "content": extracted["assistant_content"]},
        ]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return TrialLoadResult(
        messages=final_messages,
        task_name=task_name,
        agent_name=agent_name,
        model_name=model_name,
        reward=reward,
        source_kind=source_kind,
        job_name=trial_dir.parent.name,
        trial_name=trial_dir.name,
        extraction_status=extraction_status,
        python_turn_index=python_turn_index,
        selection_rule=selection_rule,
        assistant_turns_total=assistant_turns_total,
        assistant_python_turns=assistant_python_turns,
        assistant_answer_turns=assistant_answer_turns,
    )


def _example_summary(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    role_counts: dict[str, int] = {}
    for m in messages:
        role = str(m.get("role", "?"))
        role_counts[role] = role_counts.get(role, 0) + 1
    preview_lengths = [len(str(m.get("content", ""))) for m in messages]
    return {
        "n_messages": len(messages),
        "role_counts": role_counts,
        "content_lengths_chars": preview_lengths,
    }


def _build_sanity_fields(
    *,
    mode: str,
    dataset: DatasetDict,
    manifest: CorpusSplitManifest,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return diagnostic fields (sample row, extraction counts, etc.) to be
    merged into ``dataset_manifest.json``. Replaces the old reports/ dir
    (``dataset_sanity.{json,md}``); a single combined file under
    ``<experiment>/data/dataset_manifest.json`` is the new source of truth.
    """
    sample_row: dict[str, Any] | None = None
    if len(dataset.get("train", [])) > 0:
        sample_row = dict(dataset["train"][0])

    extraction_counts: dict[str, int] = {}
    if mode == "single":
        for row in rows:
            key = str(row.get("extraction_status") or "unknown")
            extraction_counts[key] = extraction_counts.get(key, 0) + 1

    sample_summary = _example_summary(sample_row) if sample_row else None
    sample_first_message = None
    if sample_row and sample_row.get("messages"):
        first = sample_row["messages"][0]
        sample_first_message = {
            "role": first.get("role"),
            "content_preview": str(first.get("content", ""))[:200],
            "content_chars": len(str(first.get("content", ""))),
        }
    sample_last_assistant = None
    if sample_row and sample_row.get("messages"):
        for m in reversed(sample_row["messages"]):
            if m.get("role") == "assistant":
                content_str = str(m.get("content", ""))
                sample_last_assistant = {
                    "content_preview": content_str[:300],
                    "content_chars": len(content_str),
                    "has_python_tag": "<python>" in content_str,
                    "has_fenced_python": "```python" in content_str,
                }
                break

    return {
        "extraction_status_counts": extraction_counts,
        "sample_row_summary": sample_summary,
        "sample_first_message": sample_first_message,
        "sample_last_assistant": sample_last_assistant,
    }


@dataclass
class DatasetManifest:
    mode: str
    created_at_utc: str
    jobs_dir: str
    job_prefix: str | None
    tasks_root: str
    system_prompt_path: str
    system_prompt_chars: int
    n_rows_total: int
    row_counts: dict[str, int]
    rows_by_task: dict[str, int]
    train_tasks: list[str]
    protected_test_tasks: list[str]
    n_trials_seen: int
    n_trials_kept: int
    n_filtered_by_reward: int
    n_missing_source: int
    n_no_messages: int
    n_missing_reward: int = 0
    n_zero_reward_sentinel: int = 0
    n_dropped_test_task_rows: int = 0
    dropped_test_task_breakdown: dict[str, int] = field(default_factory=dict)
    top_k_lowest_reward: int | None = None
    rows_filtered_by_top_k: int = 0
    top_k_kept_per_task: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _detect_known_prompt_file(
    rows: list[dict[str, Any]], repo_root: Path
) -> str | None:
    """If every row carries the same system prompt and it starts with the
    content of exactly one file under ``examples/prompts/``, return that
    file's repo-relative path. Otherwise return None.

    Matches on ``file_content.rstrip() + '\\n\\n'`` so the detection survives
    drift in the in-script ``MODE_PROMPT_ADDENDUM`` between collect time and
    prepare time. When multiple candidates match (e.g. variants that extend
    a shared header), the longest matching content wins.
    """
    if not rows:
        return None
    sample = rows[0].get("messages") or []
    if not sample or sample[0].get("role") != "system":
        return None
    traj_sp = sample[0].get("content")
    if not isinstance(traj_sp, str) or not traj_sp:
        return None
    for r in rows[1:]:
        msgs = r.get("messages") or []
        if not msgs or msgs[0].get("role") != "system":
            return None
        if msgs[0].get("content") != traj_sp:
            return None
    prompts_dir = repo_root / "examples" / "prompts"
    if not prompts_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for prompt_file in sorted(prompts_dir.glob("*.txt")):
        try:
            text = prompt_file.read_text(encoding="utf-8")
        except OSError:
            continue
        rstripped = text.rstrip()
        if not rstripped:
            continue
        if traj_sp.startswith(rstripped + "\n\n"):
            candidates.append((len(rstripped), prompt_file))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    try:
        rel = best.relative_to(repo_root)
    except ValueError:
        rel = best
    return str(rel)


def _split_dir_name(mode: str) -> str:
    return f"dataset_{mode}_task_split"


def prepare(
    *,
    mode: str,
    jobs_dir: str | Path,
    output_root: str | Path,
    tasks_root: str | Path = "tasks",
    job_prefix: str | None = None,
    system_prompt_path: str | Path | None = None,
    reward_max: float | None = None,
    test_tasks: list[str] | None = None,
    top_k_lowest_reward: int | None = None,
) -> dict[str, Any]:
    """Build the SFT DatasetDict for ``mode`` and write all artefacts."""
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unknown mode: {mode!r}")

    jobs_path = Path(jobs_dir).resolve()
    output_root = Path(output_root).resolve()
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    override_system_prompt: str | None = None
    resolved_prompt_path: Path | None = None
    if system_prompt_path is not None:
        override_system_prompt = _load_override_system_prompt(
            system_prompt_path, mode=mode
        )
        resolved_prompt_path = Path(system_prompt_path)

    trial_dirs = _iter_trial_dirs(jobs_path)
    if job_prefix:
        trial_dirs = [t for t in trial_dirs if t.parent.name.startswith(job_prefix)]
    if not trial_dirs:
        raise RuntimeError(
            f"No trial directories under {jobs_path} matching prefix={job_prefix!r}"
        )

    rows: list[dict[str, Any]] = []
    n_trials_seen = 0
    n_missing_source = 0
    n_no_messages = 0
    n_filtered = 0
    n_missing_reward = 0
    n_zero_reward_sentinel = 0

    for trial_dir in trial_dirs:
        n_trials_seen += 1
        result = load_messages_from_trial(
            trial_dir, mode=mode, override_system_prompt=override_system_prompt
        )
        if result is None:
            if not (trial_dir / "agent" / "trajectory.json").exists() and not (
                trial_dir / "agent" / "conversation.json"
            ).exists():
                n_missing_source += 1
            else:
                n_no_messages += 1
            continue
        # Drop verifier failures even without optional reward filters: None
        # means no reward was written, and non-positive values are sentinels.
        if result.reward is None:
            n_missing_reward += 1
            continue
        if result.reward <= 0.0:
            n_zero_reward_sentinel += 1
            continue
        # Optional ceiling on the loss-scaled reward.
        if reward_max is not None and result.reward > reward_max:
            n_filtered += 1
            continue

        row = {
            "messages": result.messages,
            "task": result.task_name,
            "reward": result.reward,
            "agent": result.agent_name,
            "model": result.model_name,
            "job_name": result.job_name,
            "trial_name": result.trial_name,
            "source": result.source_kind,
        }
        if mode == "single":
            row.update(
                {
                    "extraction_status": result.extraction_status,
                    "python_turn_index": result.python_turn_index,
                    "selection_rule": result.selection_rule,
                    "assistant_turns_total": result.assistant_turns_total,
                    "assistant_python_turns": result.assistant_python_turns,
                    "assistant_answer_turns": result.assistant_answer_turns,
                }
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("No usable rows after extraction")

    # Per-task top-k-lowest-reward: drop reward=None rows, sort by
    # (reward, trial_name) ascending for stable tie-break, keep first N.
    rows_filtered_by_top_k = 0
    top_k_kept_per_task: dict[str, int] = {}
    if top_k_lowest_reward is not None:
        if top_k_lowest_reward < 0:
            raise ValueError(
                f"--top-k-lowest-reward must be >= 0; got {top_k_lowest_reward}"
            )
        by_task: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_task.setdefault(str(row.get("task")), []).append(row)
        filtered_rows: list[dict[str, Any]] = []
        n_before = len(rows)
        for task_name, task_rows in by_task.items():
            ranked = sorted(
                (r for r in task_rows if r.get("reward") is not None),
                key=lambda r: (float(r["reward"]), str(r.get("trial_name", ""))),
            )
            kept = ranked[:top_k_lowest_reward]
            top_k_kept_per_task[task_name] = len(kept)
            filtered_rows.extend(kept)
        rows = filtered_rows
        rows_filtered_by_top_k = n_before - len(rows)
        if not rows:
            raise RuntimeError(
                f"No rows survived --top-k-lowest-reward={top_k_lowest_reward} filter"
            )

    test_task_names = list(test_tasks) if test_tasks else list(DEFAULT_TEST_TASKS)
    dataset, split_manifest = build_train_only_corpus(
        rows=rows,
        tasks_root=tasks_root,
        test_tasks=test_task_names,
    )

    dataset_dir = data_root / _split_dir_name(mode)
    if dataset_dir.exists():
        # Remove first so old per-split dirs don't survive a re-run.
        import shutil

        shutil.rmtree(dataset_dir)
    dataset.save_to_disk(str(dataset_dir))

    for split_name, split_ds in dataset.items():
        jsonl_path = dataset_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in split_ds:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if override_system_prompt is not None:
        manifest_prompt_path = str(resolved_prompt_path.resolve()) if resolved_prompt_path else ""
        manifest_prompt_chars = len(override_system_prompt)
    else:
        first_row_messages = rows[0]["messages"] if rows else []
        manifest_prompt_chars = (
            len(first_row_messages[0]["content"])
            if first_row_messages and first_row_messages[0].get("role") == "system"
            else 0
        )
        # If every trajectory shares one prompt that originated from a known
        # file under examples/prompts/, record the path so eval-time can
        # auto-feed it instead of falling back to the agent's in-script
        # default (which is rarely what was used during collection).
        detected = _detect_known_prompt_file(rows, _REPO_ROOT)
        manifest_prompt_path = detected if detected else "<trajectory-recorded>"

    manifest = DatasetManifest(
        mode=mode,
        created_at_utc=_utc_timestamp(),
        jobs_dir=str(jobs_path),
        job_prefix=job_prefix,
        tasks_root=str(Path(tasks_root).resolve()),
        system_prompt_path=manifest_prompt_path,
        system_prompt_chars=manifest_prompt_chars,
        n_rows_total=split_manifest.row_counts.get("train", 0),
        row_counts=split_manifest.row_counts,
        rows_by_task=split_manifest.rows_by_task,
        train_tasks=split_manifest.train_tasks,
        protected_test_tasks=split_manifest.protected_test_tasks,
        n_trials_seen=n_trials_seen,
        n_trials_kept=split_manifest.row_counts.get("train", 0),
        n_filtered_by_reward=n_filtered,
        n_missing_source=n_missing_source,
        n_no_messages=n_no_messages,
        n_missing_reward=n_missing_reward,
        n_zero_reward_sentinel=n_zero_reward_sentinel,
        n_dropped_test_task_rows=split_manifest.n_dropped_test_task_rows,
        dropped_test_task_breakdown=split_manifest.dropped_test_task_breakdown,
        top_k_lowest_reward=top_k_lowest_reward,
        rows_filtered_by_top_k=rows_filtered_by_top_k,
        top_k_kept_per_task=top_k_kept_per_task,
    )
    # Merge the diagnostic sanity fields (sample row, extraction counts) into
    # the manifest so a single JSON under <experiment>/data/ is the source
    # of truth. The old <experiment>/reports/ dir is gone.
    manifest_dict = manifest.to_dict()
    manifest_dict.update(
        _build_sanity_fields(
            mode=mode,
            dataset=dataset,
            manifest=split_manifest,
            rows=rows,
        )
    )
    manifest_path = data_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "row_counts": split_manifest.row_counts,
        "n_rows_total": len(rows),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # CLI flags win over yaml; default sentinel is None.
    top_k = args.top_k_lowest_reward
    if args.config is not None:
        cfg = _load_yaml_config(args.config)
        if top_k is None and "top_k_lowest_reward" in cfg:
            yaml_val = cfg["top_k_lowest_reward"]
            top_k = int(yaml_val) if yaml_val is not None else None

    result = prepare(
        mode=args.mode,
        jobs_dir=args.jobs_dir,
        output_root=args.output_root,
        tasks_root=args.tasks_root,
        job_prefix=args.job_prefix,
        system_prompt_path=args.system_prompt_path,
        reward_max=args.reward_max,
        test_tasks=args.test_tasks,
        top_k_lowest_reward=top_k,
    )
    print(f"[prepare] mode={args.mode} rows={result['n_rows_total']}")
    print(f"[prepare] row_counts={result['row_counts']}")
    print(f"[prepare] dataset_dir={result['dataset_dir']}")
    print(f"[prepare] manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
