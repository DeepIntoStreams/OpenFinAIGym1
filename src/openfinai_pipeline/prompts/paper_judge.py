from openfinai_pipeline.papers.schemas import PaperRecord, ScopeDefinition
from openfinai_pipeline.summary import strip_low_value_sections


def _scope_context(scope: ScopeDefinition) -> str:
    query_block = "\n".join(f"- {q}" for q in scope.queries[:10]) or "- (none)"
    categories = ", ".join(scope.categories) or "(none)"
    return (
        f"Scope name: {scope.name}\n"
        f"Scope description: {scope.description}\n"
        f"Scope categories: {categories}\n"
        f"Scope queries:\n{query_block}"
    )


def build_prefilter_prompt(scope: ScopeDefinition, paper: PaperRecord) -> str:
    return f"""
You are a strict prefilter for financial-ML paper screening.
Use only title + abstract evidence. Do not use assumptions beyond the given text.

{_scope_context(scope)}

Paper title: {paper.title}
Paper abstract: {paper.abstract}
Paper categories: {', '.join(paper.categories) or '(none)'}

Decision policy:
- Accept only if there is **STRONG EVIDENCE** of in-scope ML work.
- Reject obvious surveys, position papers, theory-only papers, or clearly off-scope domains.
- When uncertain or evidence is weak, reject by default.

Task:
1) Write detailed `reasons` grounded in title/abstract words. Be specific about what evidence supports relevance or irrelevance, and how it relates to the scope definition, queries, and categories. Decide whether this paper has **STRONG EVIDENCE** of relevance while writing.
2) Decide `label`: accepted/rejected, consistent with `reasons`.
3) Estimate `relevance_score_0_10`.
4) Estimate `confidence_0_1`.

Output JSON only with keys:
- reasons: string
- label: "accepted" | "rejected"
- relevance_score_0_10: number in [0, 10]
- confidence_0_1: number in [0, 1]
""".strip()


def prefilter_output_schema() -> dict:
    return {
        "title": "PrefilterDecision",
        "description": "Initial paper screening decision with relevance score and confidence assessment from abstract.",
        "type": "object",
        "properties": {
            "reasons": {"type": "string"},
            "label": {"type": "string", "enum": ["accepted", "rejected"]},
            "relevance_score_0_10": {"type": "number", "minimum": 0, "maximum": 10},
            "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["reasons", "label", "relevance_score_0_10", "confidence_0_1"],
        "additionalProperties": False,
    }


def build_sift_judge_prompt(
    scope: ScopeDefinition,
    paper: PaperRecord,
    fulltext_excerpt: str | None = None,
) -> str:
    excerpt = strip_low_value_sections(fulltext_excerpt or "").strip()[:40000] or "(fulltext excerpt unavailable)"
    return f"""
You are judging whether the paper is suitable for research task construction, based on the provided fulltext excerpt and scope context. Do not invent details.

{_scope_context(scope)}

Paper title: {paper.title}
Paper fulltext excerpt:
{excerpt}

Hard acceptance rule:
- Accept only if ALL THREE conditions are satisfied:
  (1) the paper has STRONG EVIDENCE of relevance to this scope, especially in terms of experiments, datasets, and evaluation metrics; AND
  (2) there is STRONG EVIDENCE of detailed setups for ALL three: experiments, datasets, and evaluation metrics; AND
  (3) at least one benchmark-relevant dataset appears downloadable.
- Reject papers that are off-topic, surveys, position papers, or theory-only work.
- If evidence is weak, incomplete, or ambiguous, reject by default.

Dataset downloadability: treat the following as positive evidence.
- the dataset is from an open-data body or public exchange API (e.g. FRED, IMF, World Bank, ECB, BIS, exchange public REST endpoints)
- the dataset is a commonly used public benchmark (e.g. ACL18, CIKM18, KDD17, LOBSTER samples)
- the dataset is a standard public-market series — prices, returns, volumes, OHLCV, or standard macro indicators for publicly traded instruments (equities, indices, futures, FX, crypto, rates, commodities). The canonical values of such series are vendor-independent: a paid or terminal vendor named in the paper (e.g. Bloomberg, Refinitiv, WRDS, CRSP, MetaTrader, ricequant, JoinQuant, Pinnacle CLC) does NOT block downloadability as long as the same instrument identifier(s), period, and frequency are obtainable from a free, scriptable, no-interactive-login source. Representative free families include yfinance for global equities/indices/FX/futures front-month, public exchange REST or ccxt for crypto, akshare/tushare/baostock for Chinese markets, FRED/IMF/World Bank for macro, and Frankfurter/exchangerate.host for FX; this list is illustrative, not exhaustive.
- the authors directly open-source the dataset, or the paper contains a repository that lays out procedures to download the data
- Hugging Face, Kaggle, or well-known academic datasets with clear naming and source context, if the excerpt gives enough clues about accessibility
- any other strong evidence that the dataset is reproducibly downloadable without a paywall or interactive-auth barrier
- If more than one dataset is used, the downloadability condition is satisfied when at least one benchmark-relevant dataset appears downloadable from the provided evidence.

Dataset downloadability: reject if any of the following holds.
- the dataset content itself (not merely the vendor) is paper-specific or otherwise non-reconstructible from public sources: hand-curated labels, proprietary signal/factor libraries, broker-internal order flow, full-depth limit-order-book or tick-by-tick data behind paid feeds, or vendor-licensed historical universes such as historical index constituents that the paper does not include
- the data is described too vaguely to identify the canonical entity — instruments, period, and frequency are all unspecified or only hinted at
- after considering vendor-substitutable public-market series, downloadability remains genuinely ambiguous

Evidence quality requirements:
- evidence.experiments must be a detailed, concrete string covering the experimental setup, compared methods or baselines, splits or backtest windows, and evaluation protocol when present.
- evidence.datasets must be a detailed string with specific dataset names, source clues, whether the data appears public or private, and why at least one dataset appears downloadable or not downloadable.
- evidence.metrics must be a detailed string with specific metric names and enough detail to tell what is measured and how the metric is used.

Task:
1) Provide detailed `evidence` strings for experiments, datasets, and metrics — these are factual observations from the excerpt.
2) Provide `reasons` grounded in the evidence above. Be specific about what evidence supports acceptance or rejection, especially for experiments, datasets, metrics, and dataset downloadability.
3) Decide `label`: accepted/rejected, consistent with the evidence and reasons.
4) Give `score_0_10` and `confidence_0_1`.

Output JSON only with keys:
- evidence: object with keys:
  - experiments: str
  - datasets: str
  - metrics: str
- reasons: string
- label: "accepted" | "rejected"
- score_0_10: number in [0, 10]
- confidence_0_1: number in [0, 1]
""".strip()


def sift_output_schema() -> dict:
    return {
        "title": "SiftJudgement",
        "description": "In-depth paper evaluation for task construction, extracting structured evidence on experiments, datasets, and downloadability.",
        "type": "object",
        "properties": {
            "evidence": {
                "type": "object",
                "properties": {
                    "experiments": {"type": "string"},
                    "datasets": {"type": "string"},
                    "metrics": {"type": "string"},
                },
                "required": ["experiments", "datasets", "metrics"],
                "additionalProperties": False,
            },
            "reasons": {"type": "string"},
            "label": {"type": "string", "enum": ["accepted", "rejected"]},
            "score_0_10": {"type": "number", "minimum": 0, "maximum": 10},
            "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["evidence", "reasons", "label", "score_0_10", "confidence_0_1"],
        "additionalProperties": False,
    }
