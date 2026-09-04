import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from openfinai_pipeline.schemas import ScopeDefinition


class RetryConfig(BaseModel):
    max_retries: int = 3


class RunConfig(BaseModel):
    since_days: int = 30
    max_papers_per_scope: int = 200
    max_accepts_per_scope: int | None = None


class ScopesConfig(BaseModel):
    path: str = "config/scopes.yaml"


class ArxivConfig(BaseModel):
    page_size: int = 100
    max_pages: int = 5
    enable_time_slicing: bool = True
    time_slice_days: int = 30
    sort_by: str = "submittedDate"
    request_interval_sec: float = 3.0
    timeout_sec: float = 30.0


class SemanticScholarConfig(BaseModel):
    enabled: bool = True
    batch_size: int = 500
    timeout_sec: float = 30.0
    cache_path: str = "data/.cache/semantic_scholar_cache.json"


class CrossrefConfig(BaseModel):
    enabled: bool = True
    rows: int = 3
    timeout_sec: float = 25.0
    min_title_similarity: float = 0.82
    mailto: str | None = None


class JudgeConfig(BaseModel):
    threshold_default: float = 7.0
    max_llm_calls_per_scope: int = 100
    llm_judge_fraction: float = 0.8
    prefilter_enabled: bool = True
    prefilter_llm_score_min: float = 5.5
    prefilter_llm_confidence_min: float = 0.5
    ranking_citation_boost: float = 0.5
    ranking_recency_weight: float = 0.1
    ranking_recency_half_life_days: int = 365


class ProviderConfig(BaseModel):
    model: str
    timeout_sec: float = 600.0


class PhaseLLMOverride(BaseModel):
    """Per-phase LLM override; null fields inherit from global ``llm.*``.

    A phase's effective provider is ``override.provider or llm.provider``; its
    effective model is ``override.model or llm.providers[effective_provider].model``.
    The override only carries provider/model — temperature, max_tokens, retry,
    and tool_max_turns remain global.
    """

    provider: str | None = None
    model: str | None = None


class PerPhaseLLMConfig(BaseModel):
    """Optional per-phase LLM provider/model overrides.

    Phase ids match the construction pipeline stages:
    ``scrape`` (Phase 1), ``summarize`` (Phase 1b), ``triage`` (Phase 1c —
    trading-paper triage), ``dataset`` (Phase 2), ``reward`` (Phase 3),
    ``task`` (Phase 4 task assembly), ``loader`` (Phase 4 inner per-task
    ``load.py`` codegen).

    Triage defaults to ``gpt-5.4`` (not the global ``gpt-5.4-mini``) because the
    call extracts a structured config from a full-paper PDF — quality on
    multi-field structured extraction is meaningfully worse on mini.
    """

    scrape: PhaseLLMOverride = Field(default_factory=PhaseLLMOverride)
    summarize: PhaseLLMOverride = Field(default_factory=PhaseLLMOverride)
    triage: PhaseLLMOverride = Field(
        default_factory=lambda: PhaseLLMOverride(model="gpt-5.4")
    )
    dataset: PhaseLLMOverride = Field(default_factory=PhaseLLMOverride)
    reward: PhaseLLMOverride = Field(default_factory=PhaseLLMOverride)
    task: PhaseLLMOverride = Field(default_factory=PhaseLLMOverride)
    loader: PhaseLLMOverride = Field(default_factory=PhaseLLMOverride)


PHASE_NAMES: tuple[str, ...] = (
    "scrape",
    "summarize",
    "triage",
    "dataset",
    "reward",
    "task",
    "loader",
)


class LLMConfig(BaseModel):
    provider: str = "openai"
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 16000
    tool_max_turns: int = 10
    retry: RetryConfig = Field(default_factory=RetryConfig)
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            "openai": ProviderConfig(model="gpt-5.4-mini"),
            "openrouter": ProviderConfig(model="openai/gpt-5.4-mini"),
            "anthropic": ProviderConfig(model="claude-sonnet-4-6"),
            "bedrock": ProviderConfig(
                model="us.anthropic.claude-sonnet-4-6-v1:0"
            ),
            "ollama": ProviderConfig(model="qwen2.5:7b-instruct"),
        }
    )
    per_phase: PerPhaseLLMConfig = Field(default_factory=PerPhaseLLMConfig)


class DownloadConfig(BaseModel):
    enabled: bool = True
    max_mb: int = 30
    timeout_sec: float = 40.0


class StorageConfig(BaseModel):
    artifact_dir: str = "data/pipeline_output/logs"


class PapersConfig(BaseModel):
    root_dir: str = "data/pipeline_output/papers"


class DatasetsConfig(BaseModel):
    root_dir: str = "data/pipeline_output/datasets"
    curated_root_dir: str = "data/knowledge_base/datasets"


class RewardsConfig(BaseModel):
    root_dir: str = "data/knowledge_base/rewards/"
    generated_dir: str = "data/pipeline_output/rewards"


class StagingConfig(BaseModel):
    root_dir: str = "data/pipeline_output/.staging"


class ImplementationConfig(BaseModel):
    sandbox_timeout_sec: int = 120
    staging_dir: str = "data/pipeline_output/.staging/tasks"
    construction_log: str = "tasks/generated/construction_log.jsonl"
    tasks_install_dir: str = "tasks/generated"


class BenchmarkConfig(BaseModel):
    execute_download: bool = True
    download_timeout_sec: int = 300
    # Phase 2 budget: low because each retry burns a real network call and
    # download scripts are usually one-shot once auth + URL are right.
    dataset_generation_rounds: int = Field(default=2, ge=1)
    # Phase 3 budget: higher because rewards usually need a reviewer round
    # to surface signature/contract issues missed in round 1.
    reward_generation_rounds: int = Field(default=3, ge=1)
    # Phase 4 evaluator + task.py generate→sandbox-test→fix retry budget.
    task_generation_rounds: int = Field(default=3, ge=1)
    # Phase 4 per-task load.py codegen loop (round 1 attempt, round 2 review,
    # round 3 final convergence on the leakage-aware contract).
    loader_generation_rounds: int = Field(default=3, ge=1)
    implementation: ImplementationConfig = Field(default_factory=ImplementationConfig)


class LoggingConfig(BaseModel):
    dir: str = "data/pipeline_output/logs"


class CollectorsConfig(BaseModel):
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    semantic_scholar: SemanticScholarConfig = Field(
        default_factory=SemanticScholarConfig
    )
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig)


class SummaryConfig(BaseModel):
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "FIN_PIPELINE_SUMMARY_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    reranker_model: str = Field(
        default_factory=lambda: os.getenv(
            "FIN_PIPELINE_SUMMARY_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    )
    candidate_top_k: int = Field(default=12, ge=1)
    semantic_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    heuristic_weight: float = Field(default=0.15, ge=0.0, le=1.0)


# Realtime paper-trading evaluation


class ResolverConfig(BaseModel):
    """Default knobs for :class:`ResolverService`.

    Constructed directly by ResolverService when no config is passed;
    not currently bound to any AppConfig field — kept here so callers
    that want a non-default resolver pass a ``ResolverConfig(...)``
    explicitly rather than reaching into a YAML block.
    """

    poll_interval_sec: int = 60
    batch_size: int = 100
    grace_period_sec: int = 30


class TradingConfig(BaseModel):
    """Configuration for paper trading simulation.

    Used by :class:`RealtimeTradingTask` and optionally by offline
    :class:`TradingTask` subclasses that want cost modeling. Per-task
    knobs are supplied via each curated bundle's ``task.toml`` under
    ``[curated.default_config]``; this class is not bound to any
    AppConfig field.
    """

    slippage_pct: float = 0.001  # 0.1% = 10 basis points
    transaction_cost_pct: float = 0.0  # percentage of notional per trade
    execution_mode: str = "internal_paper"  # "internal_paper" | "alpaca_paper"
    # True => AlpacaPaperExecutor closes residual positions + cancels open
    # orders at construction. False => raises on dirty account so PnL can't
    # be contaminated by prior-session carry. SimulatedExecutor ignores this.
    flatten_on_start: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict()

    run: RunConfig = Field(default_factory=RunConfig)
    scopes: ScopesConfig = Field(default_factory=ScopesConfig)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    papers: PapersConfig = Field(default_factory=PapersConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    rewards: RewardsConfig = Field(default_factory=RewardsConfig)
    staging: StagingConfig = Field(default_factory=StagingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)


_DEFAULT_BASE = Path(__file__).resolve().parents[2] / "config" / "base.yaml"
_DEFAULT_SCOPES = Path(__file__).resolve().parents[2] / "config" / "scopes.yaml"
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_app_config(config_path: str | None = None) -> AppConfig:
    from omegaconf import OmegaConf

    env_path = os.getenv("FIN_PIPELINE_CONFIG")
    path = _resolve_input_file(config_path or env_path, default=_DEFAULT_BASE)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    cfg = AppConfig.model_validate(raw)
    data_root = _resolve_data_root_override()
    if data_root is not None:
        _apply_data_root_override(cfg, data_root)
    _normalize_config_paths(cfg)
    return cfg


def load_scope_definitions(scopes_path: str | None = None) -> list[ScopeDefinition]:
    from omegaconf import OmegaConf

    env_path = os.getenv("FIN_PIPELINE_SCOPES")
    path = _resolve_input_file(scopes_path or env_path, default=_DEFAULT_SCOPES)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    scopes = raw.get("scopes", [])
    return [ScopeDefinition.model_validate(item) for item in scopes]


def filter_scopes(
    scopes: list[ScopeDefinition], ids: list[str] | None = None
) -> list[ScopeDefinition]:
    if not ids:
        return [scope for scope in scopes if scope.enabled]
    wanted = set(ids)
    return [scope for scope in scopes if scope.id in wanted and scope.enabled]


def _resolve_data_root_override() -> Path | None:
    raw = os.getenv("FIN_PIPELINE_DATA_ROOT", "").strip()
    if not raw:
        return None
    return _resolve_output_path(raw)


def _apply_data_root_override(cfg: AppConfig, data_root: Path) -> None:
    cfg.collectors.semantic_scholar.cache_path = _rebase_data_path(
        cfg.collectors.semantic_scholar.cache_path, data_root
    )
    cfg.storage.artifact_dir = _rebase_data_path(cfg.storage.artifact_dir, data_root)
    cfg.papers.root_dir = _rebase_data_path(cfg.papers.root_dir, data_root)
    cfg.datasets.root_dir = _rebase_data_path(cfg.datasets.root_dir, data_root)
    cfg.datasets.curated_root_dir = _rebase_data_path(cfg.datasets.curated_root_dir, data_root)
    cfg.rewards.root_dir = _rebase_data_path(cfg.rewards.root_dir, data_root)
    cfg.rewards.generated_dir = _rebase_data_path(cfg.rewards.generated_dir, data_root)
    cfg.staging.root_dir = _rebase_data_path(cfg.staging.root_dir, data_root)
    cfg.benchmark.implementation.staging_dir = _rebase_data_path(
        cfg.benchmark.implementation.staging_dir, data_root
    )
    cfg.benchmark.implementation.construction_log = _rebase_data_path(
        cfg.benchmark.implementation.construction_log, data_root
    )
    cfg.benchmark.implementation.tasks_install_dir = _rebase_data_path(
        cfg.benchmark.implementation.tasks_install_dir, data_root
    )
    cfg.logging.dir = _rebase_data_path(cfg.logging.dir, data_root)


def _normalize_config_paths(cfg: AppConfig) -> None:
    cfg.scopes.path = str(_resolve_input_file(cfg.scopes.path, default=_DEFAULT_SCOPES))
    cfg.collectors.semantic_scholar.cache_path = str(
        _resolve_output_path(cfg.collectors.semantic_scholar.cache_path)
    )
    cfg.storage.artifact_dir = str(_resolve_output_path(cfg.storage.artifact_dir))
    cfg.papers.root_dir = str(_resolve_output_path(cfg.papers.root_dir))
    cfg.datasets.root_dir = str(_resolve_output_path(cfg.datasets.root_dir))
    cfg.datasets.curated_root_dir = str(_resolve_output_path(cfg.datasets.curated_root_dir))
    cfg.rewards.root_dir = str(_resolve_output_path(cfg.rewards.root_dir))
    cfg.rewards.generated_dir = str(
        _resolve_output_path(cfg.rewards.generated_dir)
    )
    cfg.staging.root_dir = str(_resolve_output_path(cfg.staging.root_dir))
    cfg.benchmark.implementation.staging_dir = str(
        _resolve_output_path(cfg.benchmark.implementation.staging_dir)
    )
    cfg.benchmark.implementation.construction_log = str(
        _resolve_output_path(cfg.benchmark.implementation.construction_log)
    )
    cfg.benchmark.implementation.tasks_install_dir = str(
        _resolve_output_path(cfg.benchmark.implementation.tasks_install_dir)
    )
    cfg.logging.dir = str(_resolve_output_path(cfg.logging.dir))


def _resolve_input_file(value: str | Path | None, *, default: Path) -> Path:
    if value is None:
        return default.resolve()
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw
    candidates = [
        (Path.cwd() / raw),
        (_PIPELINE_ROOT / raw),
        (_PROJECT_ROOT / raw),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_output_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw

    first = raw.parts[0] if raw.parts else ""
    if first == "config":
        return (_PIPELINE_ROOT / raw).resolve()
    if first in {"data", "harbor"}:
        return (_PROJECT_ROOT / raw).resolve()
    if (Path.cwd() / raw).exists():
        return (Path.cwd() / raw).resolve()
    if (_PIPELINE_ROOT / raw).exists():
        return (_PIPELINE_ROOT / raw).resolve()
    return (_PROJECT_ROOT / raw).resolve()


def _rebase_data_path(value: str, data_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    parts = path.parts
    if not parts or parts[0] != "data":
        return value
    return str(data_root.joinpath(*parts[1:]))


def resolve_phase_llm(
    base: LLMConfig,
    phase: str,
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    cli_max_tokens: int | None = None,
    cli_temperature: float | None = None,
    cli_llm_max_retries: int | None = None,
    cli_timeout_sec: float | None = None,
    cli_tool_max_turns: int | None = None,
) -> LLMConfig:
    """Resolve effective LLM config for a phase.

    Precedence (highest wins): CLI args > ``base.per_phase.<phase>`` > global
    ``base.*``. Returns a deep copy with the resolved values applied; the
    input ``base`` is not mutated. ``provider`` and ``model`` are the only
    fields that vary per phase — every other field is purely CLI > global.
    """
    if phase not in PHASE_NAMES:
        raise ValueError(
            f"Unknown phase '{phase}'. Expected one of: {sorted(PHASE_NAMES)}"
        )
    cfg = base.model_copy(deep=True)
    override = getattr(cfg.per_phase, phase)

    if cli_provider is not None:
        cfg.provider = cli_provider
    elif override.provider is not None:
        cfg.provider = override.provider

    if cli_temperature is not None:
        cfg.temperature = cli_temperature
    if cli_max_tokens is not None:
        cfg.max_tokens = cli_max_tokens
    if cli_llm_max_retries is not None:
        cfg.retry.max_retries = cli_llm_max_retries
    if cli_tool_max_turns is not None:
        cfg.tool_max_turns = cli_tool_max_turns

    selected = cfg.provider
    if selected not in cfg.providers:
        bootstrap_model = (
            cli_model
            if cli_model is not None
            else override.model
        )
        if not bootstrap_model:
            raise ValueError(
                f"Unknown provider '{selected}' for phase '{phase}'. "
                "Pass --model since no default is known for this provider."
            )
        cfg.providers[selected] = ProviderConfig(
            model=bootstrap_model,
            timeout_sec=cli_timeout_sec if cli_timeout_sec is not None else 600.0,
        )

    if cli_model is not None:
        cfg.providers[selected].model = cli_model
    elif override.model is not None:
        cfg.providers[selected].model = override.model

    if cli_timeout_sec is not None:
        cfg.providers[selected].timeout_sec = cli_timeout_sec

    return cfg
