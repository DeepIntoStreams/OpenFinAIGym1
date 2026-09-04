"""Trajectory and dataset preparation utilities.

* :mod:`.prepare` — unified single/multi-turn dataset prep CLI.
* :mod:`.splitting` — train-only corpus assembly (5 protected test tasks
  excluded from the SFT corpus; reserved for harbor-based eval).
* :mod:`.visualize` — diagnostic plots (training loss, per-task reward).
"""

from openfinai_skyrl.data.prepare import (
    atif_to_messages,
    conversation_to_messages,
    extract_single_turn_final_code_from_conversation,
    extract_single_turn_final_code_from_messages,
    extract_single_turn_final_code_from_trajectory,
    load_messages_from_trial,
    prepare,
)
from openfinai_skyrl.data.splitting import (
    DEFAULT_TEST_TASKS,
    CorpusSplitManifest,
    available_task_names,
    build_train_only_corpus,
    normalize_exported_task_name,
)

__all__ = [
    "DEFAULT_TEST_TASKS",
    "CorpusSplitManifest",
    "atif_to_messages",
    "available_task_names",
    "build_train_only_corpus",
    "conversation_to_messages",
    "extract_single_turn_final_code_from_conversation",
    "extract_single_turn_final_code_from_messages",
    "extract_single_turn_final_code_from_trajectory",
    "load_messages_from_trial",
    "normalize_exported_task_name",
    "prepare",
]
