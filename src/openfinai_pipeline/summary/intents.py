from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalIntent:
    name: str
    terms: tuple[str, ...]
    max_chunks: int
    section_terms: tuple[str, ...] = ()
    query_text: str = ""


_INTENTS: tuple[RetrievalIntent, ...] = (
    RetrievalIntent(
        name="task_overview",
        terms=("task", "forecast", "forecasting", "prediction", "predict", "target", "horizon", "return", "volatility", "classification", "generation", "portfolio", "trading", "asset", "market"),
        max_chunks=2,
        section_terms=("abstract", "introduction", "problem", "task"),
        query_text="task objective target prediction horizon asset class market modality forecasting classification generation trading portfolio",
    ),
    RetrievalIntent(
        name="algorithm_method",
        terms=("method", "methods", "model", "models", "algorithm", "architecture", "objective", "loss", "optimization", "training", "inference", "transformer", "lstm", "gan", "diffusion", "vae"),
        max_chunks=2,
        section_terms=("method", "model", "algorithm", "approach"),
        query_text="main method algorithm model architecture objective loss training inference approach",
    ),
    RetrievalIntent(
        name="experiments",
        terms=("experiment", "experiments", "baseline", "baselines", "ablation", "split", "backtest", "window", "validation", "test", "protocol", "comparison", "benchmark", "compared"),
        max_chunks=2,
        section_terms=("experiments", "evaluation", "results", "backtest"),
        query_text="experimental setup baselines ablations train validation test splits backtest evaluation protocol comparison",
    ),
    RetrievalIntent(
        name="real_dataset",
        terms=("dataset", "data source", "download", "repository", "github", "notebook", "appendix", "supplementary", "benchmark", "kaggle", "huggingface", "fred", "yahoo", "binance", "api", "endpoint", "url", "ticker", "exchange", "frequency", "feature", "label", "target", "csv", "table"),
        max_chunks=3,
        section_terms=("data", "dataset", "appendix", "supplementary"),
        query_text="dataset source provider download benchmark appendix repository code notebook real data access details endpoint url ticker exchange frequency features labels targets",
    ),
    RetrievalIntent(
        name="synthetic_dataset",
        terms=("synthetic", "simulate", "simulation", "simulator", "generator", "sampling", "calibration", "parameter", "seed", "monte carlo", "stochastic", "brownian", "copula", "bootstrap", "scenario", "appendix"),
        max_chunks=3,
        section_terms=("simulation", "synthetic", "appendix", "supplementary"),
        query_text="synthetic dataset simulation generator calibration parameters seed monte carlo stochastic brownian copula bootstrap scenario appendix supplementary generation details",
    ),
    RetrievalIntent(
        name="implementation_details",
        terms=("implementation", "experimental setup", "hyperparameter", "hyperparameters", "batch size", "learning rate", "epochs", "hardware", "runtime", "optimizer", "dropout", "hidden size", "appendix", "supplementary"),
        max_chunks=2,
        section_terms=("implementation", "experimental setup", "training", "appendix", "supplementary"),
        query_text="implementation details experimental setup hyperparameters batch size learning rate epochs hardware runtime appendix supplementary",
    ),
    RetrievalIntent(
        name="metrics",
        terms=("metric", "metrics", "evaluation", "benchmark", "loss", "accuracy", "error", "sharpe", "precision", "recall", "appendix", "mse", "rmse", "mae", "mape", "r2", "crps", "mmd", "fid", "auc", "f1", "formula", "equation", "defined as", "measured by"),
        max_chunks=2,
        section_terms=("evaluation", "experiment", "results", "appendix"),
        query_text="evaluation metrics benchmark loss accuracy error sharpe precision recall results appendix formula equation defined as measured by",
    ),
    RetrievalIntent(
        name="code_links",
        terms=("github", "repository", "repo", "code", "notebook", "implementation", "supplementary", "appendix"),
        max_chunks=1,
        section_terms=("implementation", "appendix", "supplementary"),
        query_text="github repository repo code notebook implementation supplementary appendix",
    ),
)


def retrieval_intents() -> tuple[RetrievalIntent, ...]:
    return _INTENTS
