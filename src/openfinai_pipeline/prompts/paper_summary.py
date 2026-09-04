from openfinai_pipeline.papers.schemas import PaperRecord, ScopeDefinition
from openfinai_pipeline.prompts.paper_judge import _scope_context


def build_paper_summary_prompt(
    scope: ScopeDefinition,
    paper: PaperRecord,
    fulltext_excerpt: str | None = None,
) -> str:
    excerpt = (fulltext_excerpt or "").strip() or "(fulltext excerpt unavailable)"
    return f"""
You are extracting a detailed Machine Learning task summary from one paper.
Use only the supplied paper metadata and fulltext excerpt. Do not invent details.

{_scope_context(scope)}

Paper title: {paper.title}
Paper fulltext excerpt:
{excerpt}

Context
- You are producing the structured paper summary that will directly feed downstream dataset and reward construction.
- Downstream, this summary is used to: download real data or generate synthetic data for ML experiments, construct evaluation code, and construct well-specified benchmark tasks.
- Therefore, datasets, metrics, and experiments are not just descriptive. They must preserve the concrete implementation details needed for those downstream steps.
- Use only explicit evidence from the supplied excerpt. When details are partial, preserve every concrete clue instead of collapsing to vague summaries.

Objective
- Extract one precise ML task summary, detailed experimental evidence, all relevant datasets, and all relevant evaluation metrics supported by the excerpt.
- Include both real datasets and synthetic datasets. Any generated or simulated data, including synthetic data derived from real seed data, must be labeled as `synthetic`.
- Ensure the extracted summary covers not only datasets and metrics, but also the main method/algorithm and the implementation or experimental setup details when the excerpt supports them.
- Prioritize details that let a developer later reproduce the data path, write evaluation logic, and implement the resulting task with minimal guesswork.
- Also classify the benchmark interaction protocol with `task_family`, choosing exactly one of:
  - `forecasting`
  - `generative`
  - `trading`

Task extraction rules
- `task_family` is set by the metrics in the paper's HEADLINE REPORTED RESULTS — not by model name, task name, prose, or scope.
  - `forecasting`: model outputs vs ground truth — MSE, RMSE, MAE, MAPE, R2, IC, RankIC, AUC, F1, Brier, accuracy.
  - `generative`: distributional quality of samples — FID, KID, W1.
  - `trading`: trade actions / equity curves — PnL, total return, Sharpe, Sortino, Calmar, IR, drawdown, win rate, turnover.
- Decision: scan that metric list once. ANY forecasting-list metric present → `forecasting`; else all metrics on the trading list → `trading`. Ignore model-selection criteria (AIC, BIC, training loss) and intermediate component scores (e.g. an inner classifier's Brier/precision when the headline reports a PnL backtest).
- For `trading`, ensure `experiments` surfaces backtest start date, end date, asset universe, and bar resolution when stated.
- task_name must be a short, precise identifier for the ML task — specific enough that a developer
  reading just the name knows the prediction target, asset class, and task type.
  The task_name drives folder names, class names, and module paths downstream, so compactness is critical.
  Naming rules:
  - Use underscores to separate components: asset/market, data modality, and task type.
    "BTC_DailyClose_Forecasting", "MultiCrypto_Correlation_Forecasting", "ETH_OHLCV_PortfolioOptimization",
    "LOB_MidPrice_Prediction", "CryptoSentiment_TrendClassification", "FX_Volatility_Forecasting",
    "BTC_HFT_Trading", "USEquities_StatArb_Trading".
  - The name must describe the TASK (what to predict/optimize), not the METHOD (how the paper solves it).
    BAD:  "VMD_LSTM_Bitcoin_Forecasting" (method in name) → GOOD: "BTC_DailyClose_Forecasting"
    BAD:  "XGBoost_Crypto_Prediction" (method in name) → GOOD: "MultiCrypto_PriceReturn_Forecasting"
    BAD:  "The paper studies univariate time series forecasting" (sentence, not identifier) → GOOD: "BTC_UnivariatePriceForecasting"
    BAD:  "Transformer_Multivariate_Crypto" (method in name) → GOOD: "MultiCrypto_Multivariate_Forecasting"
  - Do NOT embed date ranges, sample sizes, model names, or author names. Those belong in ml_task_summary only.
  - Keep names under 60 characters. Names must never be empty or whitespace-only.
- Produce a concise but informative ml_task_summary of the paper's ML task and contribution. Make it as specific as the excerpt allows about asset class or market, target, prediction horizon, forecasting objective, core method, and implementation setup.
- experiments must be detailed. Clearly describe the experimental setup, model or baseline comparisons, splits, backtest design, evaluation protocol, and implementation or hyperparameter details when present.
- If the excerpt contains a GitHub link, dataset landing page, download URL, code repository, appendix code pointer, notebook link, or other implementation pointer, place dataset-relevant links in the relevant `datasets` descriptions, place implementation- or evaluation-related links in `experiments` and the relevant `metrics` descriptions, and include every directly relevant URL in the top-level `links` array.

Link extraction rules
- `links` must contain unique URL strings closely and directly linked to the current paper's reproducibility record.
- Include only URLs that help reproduce the current paper's experiments or data path: official/current-paper code repositories, dataset landing pages, direct download URLs, API or documentation pages for paper-used data, notebooks, supplementary artifact links, and paper-used Hugging Face, Kaggle, Zenodo, Google Drive, S3, or similar data artifact links.
- Do NOT include links merely because they appear in references or related work.
- Do NOT include baseline model repositories, unrelated software libraries, generic provider homepages, citation links, publisher pages, arXiv/DOI links, or other papers' project pages unless that link directly hosts reproducibility artifacts for the current paper.
- If no directly relevant reproducibility links are supported by the excerpt, output an empty array.

Dataset extraction rules
- dataset name must precisely identify the data source, asset, and data type — specific enough that a developer reading just the name knows where to download from and what data to expect.
  The dataset name and description will be the sole input used to generate a download script downstream, so precision is critical.
  Naming rules:
  - The name must include the data provider/source and the primary asset or instrument:
    "YahooFinance_BTC_DailyOHLCV", "Binance_ETH_DailyOHLCV", "FI2010_LOB", "Kaggle_CryptoSentiment", "FRED_GlobalM2", "CryptoCompare_BTC_DailyOHLCV".
  - For well-known benchmark datasets, use their established name: "FI2010_LOB", "ACL18", "CIKM18", "KDD17".
  - Do NOT embed date ranges, full ticker lists, sample sizes, or model names in the dataset name. Those belong in the description only.
    BAD:  "Yahoo Finance BTC-USD daily closing prices (2017-08-11 to 2025-06-13)" → GOOD: "YahooFinance_BTC_DailyClose"
    BAD:  "Binance OHLC and metadata for 62 cryptocurrencies (including BTC, ETH, BNB, XRP, ADA)" → GOOD: "Binance_MultiCrypto_DailyOHLC"
    BAD:  "Historical cryptocurrency price data for Bitcoin (BTC), Ethereum (ETH), Dogecoin (DOGE), and Litecoin (LTC) from global and localized exchanges" → GOOD: "MultiExchange_MultiCrypto_HistoricalPrices"
  - If the paper does not specify the data source clearly, use the most specific name possible from available clues. Do not invent a source. If the paper is truly vague (e.g., "Bitcoin price data" with no source), note this ambiguity in the description.
  - Keep names under 60 characters. Names must never be empty or whitespace-only.
- Each dataset entry must include:
  - `dataset_kind`: exactly one of `real` or `synthetic`
  - `description`: the main human-readable dataset summary
- Real dataset rules:
  - Use `dataset_kind="real"` for externally sourced or downloadable data.
  - `description` must be as detailed and concrete as possible because it is the primary reference for generating a download script downstream.
  - Include every relevant detail supported by the paper, especially the exact source, release context, provider, and how to fetch, download, or access the data.
  - Include all specifics excluded from the name here: exact date ranges, full ticker/instrument lists, frequency, number of assets, download URLs, API endpoints, repository links, notebook links, GitHub links, repo names, appendix references, code pointers, and access constraints.
  - Capture repository names, API/provider/exchange clues, download procedures, access constraints, asset universe, ticker/instrument coverage, date range, frequency, dataset structure, files/tables, splits, labels/targets, features/covariates, and how the paper uses the dataset.
  - Also include preprocessing, cleaning, filtering, normalization, feature engineering, label construction, and any stated statistical properties such as class balance, sparsity, missingness, volatility, distributional characteristics, or summary statistics.
  - If a GitHub link, dataset landing page, or direct download link is present, include it in the relevant dataset `description`.
- Synthetic dataset rules:
  - Use `dataset_kind="synthetic"` for any simulated or generated data, including synthetic data built from real seed data.
  - `description` must clearly state what the synthetic dataset is, how it is used in the paper, and every supported generation detail needed to reproduce it.
  - For synthetic datasets, pack all synthesis clues into `description`: simulator/process name, distributions, parameters, constraints, horizons, calibration method, seeds if stated, sequence lengths, entity or asset counts, preprocessing, filtering, and how the generation depends on any real seed data.
  - Also include notebook links, GitHub links, repo names, appendix references, and code pointers when present.
  - If the synthetic pipeline depends on real seed data and the excerpt gives a source or download link for that seed data, include those links in the relevant dataset `description` and reference them in `experiments`.
- Include ALL relevant datasets supported by the excerpt, not just the most prominent one.

Metric extraction rules
- metrics name must uniquely identify the mathematical formula or scoring function being computed — precise enough that a developer reading just the name knows exactly what code to write.
  The metric name and description will be the sole input used to generate a PyTorch evaluation function downstream, so precision is critical.
  Naming rules:
  - Use CamelCase or a standard abbreviation that maps to exactly one formula:
    "MSE" (mean of squared errors), "RMSE" (root mean squared error), "MAE" (mean absolute error), "MAPE" (mean absolute percentage error), "R2" (coefficient of determination), "SharpeRatio" (mean return / std of returns), "MaxDrawdown" (largest peak-to-trough decline), "F1Score" (harmonic mean of precision and recall), "MacroF1Score" (F1 macro-averaged across classes), "AUC_ROC" (area under ROC curve), "CRPS" (continuous ranked probability score), "InformationRatio" (excess return / tracking error), "DirectionalAccuracy" (fraction of correct direction-of-change predictions), "PermutationEntropy" (Shannon entropy of ordinal patterns), "WinklerScore" (interval forecast scoring rule).
  - If a metric is a meaningful variant of a standard metric, the name must capture the variant: "MinMaxRMSE" (RMSE divided by target range), "AnnualizedSharpeRatio", "WeightedF1Score".
  - Do NOT put paper-specific context in the name. Dataset names, forecast horizons, scaling factors, model names, asset names, and sample sizes belong in the description only.
  - If the paper uses vague language ("forecasting accuracy", "prediction quality", "model performance"), do NOT use that phrase as the metric name. Instead, identify the specific formula from the paper's methodology or results section and use its canonical name. If no formula can be determined, omit the metric entirely.
  - If the paper describes a concept that is not a computable quantity (e.g., "computational efficiency", "model interpretability"), omit it — every metric entry must correspond to a concrete formula that can be implemented as code.
  - Names must never be empty or whitespace-only. Keep names under 40 characters.
- metrics description must be as detailed and concrete as possible — the description is the primary reference for generating the metric as a PyTorch evaluation function downstream.
  The same level of detail is expected for metrics as for datasets. Include every relevant detail supported by the paper, especially the exact mathematical definition, the formula, what inputs it takes (predictions and targets for forecasting, or generated and real distributions for generative), and how it is computed or applied in the experiment or backtest.
  Include all paper-specific context here that was excluded from the name: scaling factors, horizons, thresholds, dataset-specific conditions, aggregation modes, and variant details.
  Capture metric variants, thresholds, horizons, aggregation, ranking/directional settings, optimization versus reporting use, higher-is-better or lower-is-better interpretation, normalization/averaging/weighting, confidence or statistical significance details, and any economic or practical interpretation discussed in the paper.
- If the paper gives only partial details, extract every explicit clue instead of collapsing to generic descriptions.

Output contract
Output JSON only with keys, in this exact order:
- ml_task_summary: string
- experiments: string
- links: array of strings
- datasets: array of objects with keys:
  - name: string
  - description: string
  - dataset_kind: "real" | "synthetic"
- metrics: array of objects with keys:
  - name: string
  - description: string
- task_family: string
- task_name: string
""".strip()


def paper_summary_schema() -> dict:
    return {
        "title": "PaperSummaryExtract",
        "description": "Complete extraction of ML task specification, datasets, metrics, and experiments from paper fulltext.",
        "type": "object",
        "properties": {
            "ml_task_summary": {"type": "string"},
            "experiments": {"type": "string"},
            "links": {
                "type": "array",
                "items": {"type": "string"},
            },
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "dataset_kind": {"type": "string", "enum": ["real", "synthetic"]},
                    },
                    "required": ["name", "description", "dataset_kind"],
                    "additionalProperties": False,
                },
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False,
                },
            },
            "task_family": {
                "type": "string",
                "enum": [
                    "forecasting",
                    "generative",
                    "trading",
                ],
            },
            "task_name": {"type": "string"},
        },
        "required": ["ml_task_summary", "experiments", "links", "datasets", "metrics", "task_family", "task_name"],
        "additionalProperties": False,
    }
