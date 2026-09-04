# Curated Task–Paper Matrix

Reference sheet for the **61 curated offline tasks** shipped with OpenFinGym — 48 forecasting (`1_*`) and 13 market-generation (`2_*`). For each task it records the source paper, the supervised contract, the metrics the verifier emits, and how the scalar score is aggregated.

Everything below was regenerated directly from the task bundles (`manifest.json`, `eval.py`, `metrics.py`, `instruction.md`) rather than transcribed by hand, so the metric names and scoring formulas match what the verifier actually computes.

## Scoring convention

Uniform across all 61 tasks:

- The verifier emits a dict of named metrics plus `total_loss` and `reward`.
- **`reward` is always exactly equal to `total_loss`.** No task averages, re-weights, or rescales the metric dict a second time.
- `total_loss` is **lower-is-better** for every task, and so is `reward`. A task's *scored* metrics are the ones entering `total_loss`; every other emitted metric is a **diagnostic** that is reported but does not affect the score.
- Aggregation is unweighted unless the family below states weights. Weights are not normalised across metrics of different magnitude, so where one scored metric is orders of magnitude larger than the others it dominates `total_loss` — see the note under F9/F10.

## Scoring families

| family | `total_loss` | tasks |
|---|---|---|
| **F1** | `0.35·mae + 0.25·rmse + 0.20·avg_sharpe_diff + 0.10·avg_var_diff + 0.10·avg_es_diff` | 34 |
| **F2** | `mean(mse, rmse, mae)` | 7 |
| **F3** | `mean(rmse, mae)` | 2 |
| **F4** | `avg_mafe` | 1 |
| **F5** | `rmse_nd` | 1 |
| **F6** | `0.35·mse_loss + 0.25·r2_loss + 0.20·dm_loss + 0.20·sharpe_loss` | 1 |
| **F7** | `0.30·roos2_loss + 0.20·accuracy_loss + 0.15·macro_f1_loss + 0.20·sharpe_loss + 0.10·drawdown_loss + 0.05·max_1d_loss` | 1 |
| **F8** | `mean(mse/span², mae/span, |picp − 0.95| / 0.95, mpiw/span)` | 1 |
| **F9** | `mean(scored metrics)` — unweighted mean of the task's scored metrics | 10 |
| **F10** | `sum(scored metrics)` — **sum**, not mean | 3 |

> **Magnitude caveat (F9, F10).** These families average or sum metrics that are on very different scales. `cross_correlation_score` in particular is an *unnormalised sum* over a full time×feature correlation matrix, so it is O(10²) while its siblings are O(10⁻³–10⁰); in `2_us5stock_conditional_generation` it accounts for ~98% of `total_loss`. Read those scores as dominated by their largest component rather than as an even blend.

## Metric glossary

Definitions are shared across tasks; a task's row lists only which of these it emits.

| metric | definition |
|---|---|
| `acf_loss` | discrepancy in temporal autocorrelation structure between generated and reference windows. |
| `american_put_pricing_error` | difference in American put option pricing implied by the sampled paths. |
| `annualized_return_pct` | yearly-average annualized return of the long-short portfolio induced by the predictions. |
| `annualized_volatility_pct` | yearly-average annualized volatility of that long-short portfolio. |
| `autocorrelation` | temporal autocorrelation distance. |
| `autocorrelation_l1` | L1 discrepancy in temporal autocorrelation structure. |
| `autocorrelation_lag_1` | discrepancy in within-channel autocorrelation at lag 1. |
| `autocorrelation_lag_5` | discrepancy in within-channel autocorrelation at lag 5. |
| `autocorrelation_score` | autocorrelation distance on generated paths. |
| `average` | alias for the packaged aggregate score (equal to `total_loss`). |
| `avg_es_diff` | difference in Expected Shortfall implied by generated or predicted paths. |
| `avg_mafe` | average mean absolute forecast error across all four horizons. |
| `avg_mape` | average mean absolute percentage error across all four horizons. |
| `avg_mse` | average mean squared error across all four horizons. |
| `avg_r2` | average coefficient of determination across all four horizons. |
| `avg_sharpe_diff` | difference in Sharpe ratio implied by generated or predicted paths. |
| `avg_var_diff` | difference in Value-at-Risk implied by generated or predicted paths. |
| `correlation` | cross-channel correlation distance. |
| `correlation_l1` | L1 discrepancy in cross-asset correlation structure. |
| `cov_loss` | discrepancy in cross-currency covariance structure between generated and reference windows. |
| `covariance` | covariance-structure distance. |
| `cross_corr` | discrepancy in cross-currency correlation structure between generated and reference windows. |
| `cross_correlation_lag_0` | discrepancy in cross-channel correlation at lag 0. |
| `cross_correlation_lag_5` | discrepancy in cross-channel correlation at lag 5. |
| `cross_correlation_score` | cross-channel correlation distance on generated paths. |
| `daily_return_bps` | yearly-average mean daily portfolio return in basis points. |
| `discriminative_score` | distance measured by a train/test discriminative classifier. |
| `dm_zero` | Diebold-Mariano statistic against a zero forecast. |
| `down_accuracy_pct` | yearly-average accuracy on rows whose realized return is negative. |
| `es_loss` | average absolute difference in 95% expected shortfall between generated and reference returns across time steps and channels. |
| `generalization_dq` | out-of-sample generalization error for the dynamic tail-focused strategy family. |
| `generalization_ds` | out-of-sample generalization error for the static tail-focused strategy family. |
| `hist_loss` | discrepancy in the marginal value histograms between generated and reference windows. |
| `kupiec_rejection_alpha_0_05` | rejection rate of the Kupiec tail-coverage test at alpha `0.05`. |
| `long_short_sharpe` | annualized Sharpe ratio of the long-short portfolio induced by the predictions. |
| `macro_f1_pct` | yearly-average macro-F1 for the up/down sign classification implied by the predictions. |
| `mae` | mean absolute error. |
| `mafe_h1` | mean absolute forecast error for the 1-day horizon across the 8 indices. |
| `mafe_h15` | mean absolute forecast error for the 15-day horizon across the 8 indices. |
| `mafe_h21` | mean absolute forecast error for the 21-day horizon across the 8 indices. |
| `mafe_h5` | mean absolute forecast error for the 5-day horizon across the 8 indices. |
| `mape` | mean absolute percentage error. |
| `mape_h1` | mean absolute percentage error for the 1-day horizon across the 8 indices. |
| `mape_h15` | mean absolute percentage error for the 15-day horizon across the 8 indices. |
| `mape_h21` | mean absolute percentage error for the 21-day horizon across the 8 indices. |
| `mape_h5` | mean absolute percentage error for the 5-day horizon across the 8 indices. |
| `marginal` | marginal-distribution distance. |
| `marginal_distribution` | discrepancy in the per-time-step marginal distributions of the generated channels. |
| `marginal_score` | marginal-distribution distance. |
| `marginal_score_1plus` | marginal-distribution distance. |
| `max_1day_drawdown_pct` | yearly-average worst single-day portfolio loss in percent. |
| `max_drawdown_pct` | yearly-average worst peak-to-trough portfolio drawdown in percent. |
| `mpiw` | mean prediction-interval width. |
| `mse` | mean squared error. |
| `mse_h1` | mean squared error for the 1-day horizon across the 8 indices. |
| `mse_h15` | mean squared error for the 15-day horizon across the 8 indices. |
| `mse_h21` | mean squared error for the 21-day horizon across the 8 indices. |
| `mse_h5` | mean squared error for the 5-day horizon across the 8 indices. |
| `nw_tstat` | Newey-West t-statistic of the overall long-short portfolio return series. |
| `onnd` | outgoing nearest-neighbor distance. |
| `overall_accuracy_pct` | yearly-average sign accuracy over all rows. |
| `picp` | prediction-interval coverage probability. |
| `predictive_score` | distance measured by a predictive proxy task. |
| `r2` | coefficient of determination. |
| `r2_h1` | coefficient of determination for the 1-day horizon across the 8 indices. |
| `r2_h15` | coefficient of determination for the 15-day horizon across the 8 indices. |
| `r2_h21` | coefficient of determination for the 21-day horizon across the 8 indices. |
| `r2_h5` | coefficient of determination for the 5-day horizon across the 8 indices. |
| `r2_zero` | coefficient of determination versus a zero baseline. |
| `raw_annualized_return_pct` | overall-sample annualized return of the long-short portfolio. |
| `raw_annualized_volatility_pct` | overall-sample annualized volatility of the long-short portfolio. |
| `raw_daily_return_bps` | overall-sample mean daily portfolio return in basis points. |
| `raw_down_accuracy_pct` | overall-sample accuracy on rows whose realized return is negative. |
| `raw_macro_f1_pct` | overall-sample macro-F1 for the up/down sign classification implied by the predictions. |
| `raw_max_1day_drawdown_pct` | overall-sample worst single-day portfolio loss in percent. |
| `raw_max_drawdown_pct` | overall-sample worst peak-to-trough portfolio drawdown in percent. |
| `raw_overall_accuracy_pct` | overall-sample sign accuracy over all rows. |
| `raw_return_kurtosis` | overall-sample kurtosis of portfolio returns. |
| `raw_return_skewness` | overall-sample skewness of portfolio returns. |
| `raw_roos2_pct` | overall-sample out-of-sample R-squared. |
| `raw_sharpe_ratio` | overall-sample annualized Sharpe ratio of the long-short portfolio. |
| `raw_up_accuracy_pct` | overall-sample accuracy on rows whose realized return is positive. |
| `re_alpha_0_01` | relative tail-risk error at alpha `0.01`. |
| `re_alpha_0_05` | relative tail-risk error at alpha `0.05`. |
| `re_alpha_0_10` | relative tail-risk error at alpha `0.10`. |
| `return_kurtosis` | yearly-average kurtosis of portfolio returns. |
| `return_skewness` | yearly-average skewness of portfolio returns. |
| `rmse` | root mean squared error. |
| `rmse_nd` | RMSE after the task's normalized rescaling step. |
| `roos2_pct` | yearly-average out-of-sample R-squared. |
| `score_test_rejection_alpha_0_05` | rejection rate of the score-based tail test at alpha `0.05`. |
| `sharpe_ratio` | yearly-average annualized Sharpe ratio of the long-short portfolio. |
| `sig_mmd` | signature-kernel maximum mean discrepancy between the generated and real path distributions. |
| `sig_w1_distance` | signature Wasserstein-1 distance. |
| `smape` | symmetric mean absolute percentage error. |
| `up_accuracy_pct` | yearly-average accuracy on rows whose realized return is positive. |
| `var_loss` | average absolute difference in 95% value-at-risk between generated and reference returns across time steps and channels. |

## Source papers

| key | citation | title | link | tasks |
|---|---|---|---|---|
| **MOSAIC** | Bao et al., 2026 | *MOSAIC: Modular Orchestration for Structured Agentic Intelligence and Composition* | [anon-4open](https://anonymous.4open.science/r/MOSAIC-8778/) | 44 |
| **TS-Agent** | Ang et al., 2025 | *Structured Agentic Workflows for Financial Time-Series Modelling with LLMs and Reflective Feedback* | [arXiv](https://arxiv.org/abs/2508.13915) | 5 |
| **PCF-GAN** | Lou et al., 2023 | *PCF-GAN: Generating Sequential Data via the Characteristic Function of Measures on the Path Space* | [arXiv](https://arxiv.org/abs/2305.12511) | 3 |
| **HRPD** | Tao et al., 2024 | *High Rank Path Development: an Approach to Learning the Filtration of Stochastic Processes* | [arXiv](https://arxiv.org/abs/2405.14913) | 2 |
| **EAP-ML** | Gu et al., 2020 | *Empirical Asset Pricing via Machine Learning* | `N/A` | 1 |
| **DGNN-Vol** | Kumar et al., 2024 | *Dynamic Graph Neural Networks for Enhanced Volatility Prediction in Financial Markets* | [arXiv](https://arxiv.org/abs/2410.16858) | 1 |
| **GRU-TPE** | Dinda, 2024 | *Gated Recurrent Neural Network with TPE Bayesian Optimization for Enhancing Stock Index Prediction Accuracy* | [arXiv](https://arxiv.org/abs/2406.02604) | 1 |
| **TSFM** | Rahimikia et al., 2025 | *Re(Visiting) Time Series Foundation Models in Finance* | [arXiv](https://arxiv.org/abs/2511.18578) | 1 |
| **YieldCurve** | Richman & Scognamiglio, 2024 | *Multiple Yield Curve Modeling and Forecasting Using Deep Learning* | [ASTIN Bulletin](https://www.cambridge.org/core/product/identifier/S0515036124000266/type/journal_article) | 1 |
| **Tail-GAN** | Cont et al., 2026 | *Tail-GAN: Learning to Simulate Tail Risk Scenarios* | [arXiv](https://arxiv.org/abs/2203.01664) | 1 |
| **TimeGAN** | Yoon et al., 2019 | *Time-series Generative Adversarial Networks* | [NeurIPS](https://papers.neurips.cc/paper/8789-time-series-generative-adversarial-networks) | 1 |

## Forecasting tasks (48)

`contract` is the per-sample supervised map (lookback × channels → horizon × channels); `required output` is the exact array the agent must write.

| task_folder | task name | freq | source | contract | required output | scored metrics | diagnostics | family |
|---|---|---|---|---|---|---|---|---|
| `1_commodity_logreturn_forecasting` | Commodity Log-Return Forecasting | daily | MOSAIC | `60×17 → 10×17` | `(398, 10, 17)` | `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff`, `mae`, `rmse` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant1_logreturn_forecasting` | Commodity Variant 1 Forecasting | daily | MOSAIC | `30×6 → 5×6` | `(801, 5, 6)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant2_logreturn_forecasting` | Commodity Variant 2 Forecasting | daily | MOSAIC | `60×7 → 10×7` | `(398, 10, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant3_logreturn_forecasting` | Commodity Variant 3 Forecasting | daily | MOSAIC | `90×8 → 20×8` | `(197, 20, 8)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant4_logreturn_forecasting` | Commodity Variant 4 Forecasting | daily | MOSAIC | `20×6 → 2×6` | `(410, 2, 6)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant5_logreturn_forecasting` | Commodity Variant 5 Forecasting | daily | MOSAIC | `45×9 → 15×9` | `(264, 15, 9)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant6_logreturn_forecasting` | Commodity Variant 6 Forecasting | daily | MOSAIC | `60×7 → 10×7` | `(399, 10, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_commodity_variant7_logreturn_forecasting` | Commodity Variant 7 Forecasting | daily | MOSAIC | `40×7 → 5×7` | `(801, 5, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_forecasting` | Crypto Forecasting | hourly | MOSAIC | `72×20 → 3×20` | `(439, 3, 20)` | `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff`, `mae`, `rmse` | `average`, `mape`, `smape` | F1 |
| `1_crypto_logreturn_forecasting` | Crypto Log-Return Forecasting | hourly | MOSAIC | `72×12 → 12×12` | `(865, 12, 12)` | `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff`, `mae`, `rmse` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant1_logreturn_forecasting` | Crypto Variant 1 Forecasting | hourly | MOSAIC | `24×6 → 1×6` | `(2604, 1, 6)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant2_logreturn_forecasting` | Crypto Variant 2 Forecasting | hourly | MOSAIC | `48×7 → 6×7` | `(1300, 6, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant3_logreturn_forecasting` | Crypto Variant 3 Forecasting | hourly | MOSAIC | `72×6 → 12×6` | `(649, 12, 6)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant4_logreturn_forecasting` | Crypto Variant 4 Forecasting | hourly | MOSAIC | `168×9 → 24×9` | `(323, 24, 9)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant5_logreturn_forecasting` | Crypto Variant 5 Forecasting | hourly | MOSAIC | `12×7 → 1×7` | `(2604, 1, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant6_logreturn_forecasting` | Crypto Variant 6 Forecasting | hourly | MOSAIC | `168×8 → 24×8` | `(216, 24, 8)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_crypto_variant7_logreturn_forecasting` | Crypto Variant 7 Forecasting | hourly | MOSAIC | `96×7 → 8×7` | `(650, 8, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_excess_return_forecasting` | Monthly Equity Excess-Return Forecasting | monthly | EAP-ML | `42 features/row → 1` (tabular, no lookback tensor) | `(7200,)` | `mse`, `r2_zero`, `dm_zero`, `long_short_sharpe` | — | F6 |
| `1_equity_logreturn_forecasting` | Equity Log-Return Forecasting | daily | MOSAIC | `40×8 → 10×8` | `(447, 10, 8)` | `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff`, `mae`, `rmse` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant1_logreturn_forecasting` | Equity Variant 1 Forecasting | daily | MOSAIC | `30×5 → 5×5` | `(898, 5, 5)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant2_logreturn_forecasting` | Equity Variant 2 Forecasting | daily | MOSAIC | `60×7 → 10×7` | `(447, 10, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant3_logreturn_forecasting` | Equity Variant 3 Forecasting | daily | MOSAIC | `20×8 → 1×8` | `(902, 1, 8)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant4_logreturn_forecasting` | Equity Variant 4 Forecasting | daily | MOSAIC | `90×9 → 20×9` | `(177, 20, 9)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant5_logreturn_forecasting` | Equity Variant 5 Forecasting | daily | MOSAIC | `40×9 → 3×9` | `(900, 3, 9)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant6_logreturn_forecasting` | Equity Variant 6 Forecasting | daily | MOSAIC | `30×8 → 5×8` | `(898, 5, 8)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_equity_variant7_logreturn_forecasting` | Equity Variant 7 Forecasting | daily | MOSAIC | `50×6 → 10×6` | `(447, 10, 6)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_exchange_rate_forecasting` | Exchange Rate Forecasting | daily | TS-Agent | `60×8 → 20×8` | `(741, 20, 8)` | `rmse`, `mae` | `average`, `mape`, `smape` | F3 |
| `1_fx_variant1_logreturn_forecasting` | FX Variant 1 Forecasting | daily | MOSAIC | `30×6 → 5×6` | `(821, 5, 6)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_fx_variant2_logreturn_forecasting` | FX Variant 2 Forecasting | daily | MOSAIC | `60×7 → 10×7` | `(408, 10, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_fx_variant3_forecasting` | FX Variant 3 Forecasting | daily | MOSAIC | `10×8 → 1×8` | `(550, 1, 8)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `average`, `mape`, `mse`, `smape` | F1 |
| `1_fx_variant4_logreturn_forecasting` | FX Variant 4 Forecasting | daily | MOSAIC | `90×8 → 20×8` | `(162, 20, 8)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_fx_variant5_logreturn_forecasting` | FX Variant 5 Forecasting | daily | MOSAIC | `45×9 → 3×9` | `(548, 3, 9)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_fx_variant6_forecasting` | FX Variant 6 Forecasting | daily | MOSAIC | `20×7 → 2×7` | `(824, 2, 7)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `average`, `mape`, `mse`, `smape` | F1 |
| `1_fx_variant7_logreturn_forecasting` | FX Variant 7 Forecasting | daily | MOSAIC | `60×10 → 15×10` | `(163, 15, 10)` | `rmse`, `mae`, `avg_sharpe_diff`, `avg_var_diff`, `avg_es_diff` | `mape`, `mse`, `smape` | F1 |
| `1_global_index_volatility_forecasting` | Global Index Volatility Graph Forecasting | daily | DGNN-Vol | `21×8 → 8×4` (+ corr & DY-spillover graphs) | `(894, 8, 4)` | `avg_mafe` | 19 diagnostic keys — see glossary | F4 |
| `1_lob_forecasting` | LOB Forecasting | high-frequency LOB bars | MOSAIC | `72×14 → 3×14` | `(79, 3, 14)` | `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff`, `mae`, `rmse` | `average`, `mape`, `smape` | F1 |
| `1_nifty50_multifactor_forecasting` | NIFTY 50 Multifactor Forecasting | daily | GRU-TPE | `30×8 → 1` (scalar next-day close) | `(729,)` | `rmse_nd` | `mape`, `r2`, `rmse` | F5 |
| `1_stock10_forecasting` | Stock-10 Forecasting | business-day | MOSAIC | `60×10 → 3×10` | `(63, 3, 10)` | `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff`, `mae`, `rmse` | `average`, `mape`, `smape` | F1 |
| `1_stock_OHLCV_forecasting` | Stock OHLCV Forecasting | daily | TS-Agent | `60×5 → 10×5` | `(360, 10, 5)` | `mae`, `rmse` | `mape`, `smape` | F3 |
| `1_treasury_variant1_forecasting` | Treasury Variant 1 Forecasting | daily | MOSAIC | `20×6 → 5×6` | `(982, 5, 6)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_treasury_variant2_forecasting` | Treasury Variant 2 Forecasting | daily | MOSAIC | `60×6 → 10×6` | `(196, 10, 6)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_treasury_variant3_forecasting` | Treasury Variant 3 Forecasting | daily | MOSAIC | `30×10 → 1×10` | `(657, 1, 10)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_treasury_variant4_forecasting` | Treasury Variant 4 Forecasting | daily | MOSAIC | `40×7 → 5×7` | `(491, 5, 7)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_treasury_variant5_forecasting` | Treasury Variant 5 Forecasting | daily | MOSAIC | `40×7 → 5×7` | `(491, 5, 7)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_treasury_variant6_forecasting` | Treasury Variant 6 Forecasting | daily | MOSAIC | `15×6 → 3×6` | `(984, 3, 6)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_treasury_variant7_forecasting` | Treasury Variant 7 Forecasting | daily | MOSAIC | `5×7 → 1×7` | `(986, 1, 7)` | `mse`, `rmse`, `mae` | `average`, `mape`, `smape` | F2 |
| `1_tsfm_forecasting` | TSFM Daily Equity Excess-Return Forecasting | daily | TSFM | `252×1 → 1` per name, yearly walk-forward | `(163719,)` | `roos2_pct`, `overall_accuracy_pct`, `macro_f1_pct`, `sharpe_ratio`, `max_drawdown_pct`, `max_1day_drawdown_pct` | 23 diagnostic keys — see glossary | F7 |
| `1_yieldcurve_forecasting` | Multiple Yield Curve Forecasting | monthly | YieldCurve | `10×150 → 3×150` (lower/central/upper curves) | `(408, 3, 150)` | `mse`, `mae`, `picp`, `mpiw` | — | F8 |

## Generation tasks (13)

Unconditional tasks generate a window per sample; conditional tasks (`*_conditional_*`) generate K continuations per context — K is visible in the `required output` shape.

| task_folder | task name | freq | source | contract | required output | scored metrics | diagnostics | family |
|---|---|---|---|---|---|---|---|---|
| `2_crypto_generation` | Crypto Generation | hourly | MOSAIC | `60×20` (unconditional) | `(1319, 60, 20)` | `marginal`, `correlation`, `autocorrelation`, `covariance` | `average`, `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff` | F9 |
| `2_crypto_logreturn_generation` | Crypto Log-Return Generation | hourly | TS-Agent | `24×10` (unconditional) | `(876, 24, 10)` | `var_loss`, `es_loss` | — | F10 |
| `2_exchange_generation` | Exchange Generation | daily | TS-Agent | `7×8` (unconditional) | `(253, 7, 8)` | `hist_loss`, `cross_corr`, `acf_loss`, `cov_loss` | — | F10 |
| `2_fbm_conditional_generation` | fBM Conditional Generation | synthetic | HRPD | `5×3` (unconditional) | `(10000, 5, 3)` | `autocorrelation_score`, `cross_correlation_score`, `sig_w1_distance`, `marginal_score`, `onnd` | — | F9 |
| `2_lob_generation` | LOB Generation | high-frequency LOB bars | MOSAIC | `60×14` (unconditional) | `(238, 60, 14)` | `marginal`, `correlation`, `autocorrelation`, `covariance` | `average` | F9 |
| `2_ou_generation` | OU Generation | synthetic | PCF-GAN | `64×1` (unconditional) | `(1000, 64, 1)` | `discriminative_score`, `predictive_score`, `sig_mmd` | — | F9 |
| `2_roughvol_generation` | Rough Volatility Generation | synthetic | PCF-GAN | `200×2` (unconditional) | `(2000, 200, 2)` | `discriminative_score`, `predictive_score`, `sig_mmd`, `autocorrelation_lag_1`, `autocorrelation_lag_5`, `cross_correlation_lag_0`, `cross_correlation_lag_5`, `marginal_distribution` | — | F9 |
| `2_stock_generation` | Stock Generation | business-day | MOSAIC | `60×10` (unconditional) | `(190, 60, 10)` | `marginal`, `correlation`, `autocorrelation`, `covariance` | `average`, `avg_es_diff`, `avg_sharpe_diff`, `avg_var_diff` | F9 |
| `2_stock_logreturn_generation` | Stock Log-Return Generation | daily | TS-Agent | `5×8` (unconditional) | `(159, 5, 8)` | `hist_loss`, `cross_corr`, `acf_loss`, `cov_loss` | — | F10 |
| `2_stock_ohlcv_generation` | Stock OHLCV Generation | daily | PCF-GAN | `20×5` (unconditional) | `(230, 20, 5)` | `discriminative_score`, `predictive_score`, `sig_mmd` | — | F9 |
| `2_tailgan_generation` | Tail-Risk Scenario Generation | synthetic | Tail-GAN | `100×5` (unconditional) | `(5000, 100, 5)` | `re_alpha_0_01`, `re_alpha_0_05`, `re_alpha_0_10`, `correlation_l1`, `autocorrelation_l1`, `score_test_rejection_alpha_0_05`, `kupiec_rejection_alpha_0_05`, `generalization_dq`, `generalization_ds` | — | F9 |
| `2_timegan_stock_generation` | TimeGAN Stock Generation | daily windows | TimeGAN | `24×6` (unconditional) | `(3661, 24, 6)` | `discriminative_score`, `predictive_score` | — | F9 |
| `2_us5stock_conditional_generation` | US5 Stock Conditional Generation | daily | HRPD | `5×5`, K=128 (conditional) | `(276, 128, 5, 5)` | `autocorrelation_score`, `cross_correlation_score`, `sig_w1_distance`, `american_put_pricing_error`, `marginal_score_1plus`, `onnd` | — | F9 |

## Source-paper fidelity and verification

`dataset_verified` / `metrics_verified`: whether the packaged data and the implemented metric family were checked against the source paper. `yes` = matches; `partial` = the metric family matches but some source-side detail (significance tests, bootstrap CIs, author code) is not reproduced.

| task_folder | dataset | metrics | source problem |
|---|---|---|---|
| `1_commodity_logreturn_forecasting` | `yes` | `yes` | Multivariate commodity return forecasting over fixed rolling windows, judged by both prediction accuracy and financial-risk realism. |
| `1_commodity_variant1_logreturn_forecasting` | `yes` | `yes` | This task involves short-term multi-horizon forecasting of precious metals and energy commodities using daily futures price data. The objective is to model cross-asset dependencies to predict future price movements within a volatile market setting. |
| `1_commodity_variant2_logreturn_forecasting` | `yes` | `yes` | This task involves medium-term multivariate forecasting of the energy and precious metals complex using commodity futures data. The objective is to model cross-asset dependencies and volatility clustering to predict future price movements. |
| `1_commodity_variant3_logreturn_forecasting` | `yes` | `yes` | This task involves multi-horizon forecasting of agricultural commodity futures to capture long-term price dynamics and harvest-related cycles. The objective is to predict future log returns across a portfolio of eight agricultural assets using a sliding window approach. |
| `1_commodity_variant4_logreturn_forecasting` | `yes` | `yes` | This task involves high-frequency regime-focused forecasting of commodity futures prices. The objective is to predict future log returns across a multivariate portfolio to capture short-term price sensitivity. |
| `1_commodity_variant5_logreturn_forecasting` | `yes` | `yes` | This task involves multi-variate time-series forecasting of commodity futures to capture inter-sector dependencies between energy and metal markets. The objective is to predict future price movements based on historical log-returns. |
| `1_commodity_variant6_logreturn_forecasting` | `yes` | `yes` | This task involves soft-commodity forecasting over a medium-term horizon to capture weather- and supply-driven cross-asset dynamics. The objective is to predict future log returns across seven agricultural and soft commodity contracts using a multivariate sliding-window setup. |
| `1_commodity_variant7_logreturn_forecasting` | `yes` | `yes` | This task involves forecasting industrial and energy-linked commodity futures to capture cyclical demand shifts and cross-sector spillovers. The objective is to predict future log returns across seven industrial commodity contracts using a multivariate time-series setup. |
| `1_crypto_forecasting` | `yes` | `yes` | Short-horizon multivariate crypto forecasting over rolling benchmark windows with both point and financial-risk diagnostics. |
| `1_crypto_logreturn_forecasting` | `yes` | `yes` | Multivariate hourly crypto return forecasting for downstream trading/control under combined prediction and risk criteria. |
| `1_crypto_variant1_logreturn_forecasting` | `yes` | `yes` | This task involves short-term hourly forecasting of major cryptocurrency assets to capture immediate market momentum. It is a multivariate time-series forecasting problem designed to model cross-asset dependencies in a high-frequency financial setting. |
| `1_crypto_variant2_logreturn_forecasting` | `yes` | `yes` | This task involves medium-term multivariate forecasting of DeFi and infrastructure assets to identify sector-specific trends and cross-asset dependencies within the crypto market. The objective is to predict future log returns for a subset of assets based on historical price patterns. |
| `1_crypto_variant3_logreturn_forecasting` | `yes` | `yes` | This task involves multi-step time-series forecasting for high-volatility crypto assets, specifically targeting meme coins and privacy tokens. The objective is to model complex market dynamics and tail risks using hourly log-return data. |
| `1_crypto_variant4_logreturn_forecasting` | `yes` | `yes` | This task involves multivariate time-series forecasting to capture systemic market beta across a panel of major cryptocurrencies. The objective is to predict future hourly log returns based on historical price patterns to assess cross-asset dependencies. |
| `1_crypto_variant5_logreturn_forecasting` | `yes` | `yes` | This task involves aggressive short-horizon forecasting of hourly crypto market returns to support high-frequency trading simulations. The objective is to predict the next-step price movement across a diversified portfolio of seven major digital assets. |
| `1_crypto_variant6_logreturn_forecasting` | `yes` | `yes` | This task involves multi-day forecasting of major crypto assets to support portfolio rebalancing under changing cross-asset dependencies. The objective is to predict future hourly log returns across eight liquid digital assets using long-context multivariate windows. |
| `1_crypto_variant7_logreturn_forecasting` | `yes` | `yes` | This task involves forecasting Layer-1, infrastructure, and DeFi-related crypto assets to capture cross-chain correlation structure. The objective is to predict future hourly log returns across seven digital assets using a multivariate forecasting setup. |
| `1_equity_excess_return_forecasting` | `partial` | `yes` | Monthly stock-level excess-return prediction using firm characteristics and macro signals, judged by statistical and economic portfolio value. |
| `1_equity_logreturn_forecasting` | `yes` | `yes` | Multivariate daily equity return forecasting over fixed windows for downstream trading use. |
| `1_equity_variant1_logreturn_forecasting` | `yes` | `yes` | This task involves multi-step forecasting of 5-day log returns for a portfolio of energy sector equities. The objective is to leverage historical price dynamics to predict future returns, capturing volatility and sector-specific trends. |
| `1_equity_variant2_logreturn_forecasting` | `yes` | `yes` | Predict 10-day price movements for large-cap financial institutions to assess credit cycle sensitivity |
| `1_equity_variant3_logreturn_forecasting` | `yes` | `yes` | This task involves forecasting 1-day log returns for a portfolio of defensive consumer staples to model low-volatility market regimes. It falls under the multivariate time-series forecasting family. |
| `1_equity_variant4_logreturn_forecasting` | `yes` | `yes` | This task involves multi-asset time-series forecasting for high-growth technology stocks to capture momentum and regime shifts. The objective is to predict future price movements using a multivariate panel of equity data. |
| `1_equity_variant5_logreturn_forecasting` | `yes` | `yes` | This task involves forecasting 3-day log returns for a selection of industrial and materials sector equities to analyze cyclical economic sensitivity. It is a multivariate time-series forecasting problem designed to capture co-movement patterns across sensitive industrial assets. |
| `1_equity_variant6_logreturn_forecasting` | `yes` | `yes` | This task involves sector-focused forecasting of healthcare and pharmaceutical equities to capture shared volatility and defensive-market co-movement. The objective is to predict future daily log returns across eight healthcare-related stocks using a multivariate sliding-window setup. |
| `1_equity_variant7_logreturn_forecasting` | `yes` | `yes` | This task involves forecasting a diversified basket of retail and leisure equities to capture consumer-cycle co-movement across names. The objective is to predict future daily log returns across six consumer-facing stocks over a 10-day horizon. |
| `1_exchange_rate_forecasting` | `partial` | `yes` | Multivariate exchange-rate forecasting over fixed daily windows under a direct point-forecast metric bundle. |
| `1_fx_variant1_logreturn_forecasting` | `yes` | `yes` | This task involves multi-horizon forecasting of G10 currency pair returns to evaluate predictive accuracy and tail risk management. The objective is to model the dynamics of major foreign exchange rates using a multivariate time-series approach. |
| `1_fx_variant2_logreturn_forecasting` | `yes` | `yes` | This task involves predicting emerging market foreign exchange (FX) volatility regimes using a multivariate forecasting approach. The objective is to capture non-stationary patterns and tail events in high-volatility currency pairs over a 10-day horizon. |
| `1_fx_variant3_forecasting` | `yes` | `yes` | This task involves short-term intraday-style daily forecasting of high-liquidity foreign exchange rates. The objective is to predict the next-day close price for a portfolio of eight major currency pairs using a sliding window approach. |
| `1_fx_variant4_logreturn_forecasting` | `yes` | `yes` | This task involves long-horizon multivariate forecasting of commodity-linked foreign exchange rates to capture slow-moving global macro trends. The objective is to predict 20-day log-returns using a 90-day historical input window. |
| `1_fx_variant5_logreturn_forecasting` | `yes` | `yes` | This task involves multi-horizon forecasting of Asian currency bloc dynamics within the foreign exchange (fx) market. The objective is to leverage cross-asset dependencies to predict future price movements across a 3-day horizon. |
| `1_fx_variant6_forecasting` | `yes` | `yes` | This task involves forecasting a correlated basket of European and Swiss-linked FX pairs using short daily price-level windows. The objective is to predict the next two daily price steps across seven currency pairs while preserving cross-pair dependence. |
| `1_fx_variant7_logreturn_forecasting` | `yes` | `yes` | This task involves forecasting high-volatility emerging-market and Asia-linked FX returns over a longer horizon. The objective is to predict future daily log returns across ten currency pairs while modeling regime-sensitive cross-asset dependence. |
| `1_global_index_volatility_forecasting` | `partial` | `partial` | Forecast multi-horizon realized-volatility proxies jointly across eight global equity indices using packaged correlation and spillover graphs. |
| `1_lob_forecasting` | `yes` | `yes` | Short-horizon multivariate forecasting over LOB-derived features, with emphasis on price-channel forecast quality and financial-risk diagnostics. |
| `1_nifty50_multifactor_forecasting` | `partial` | `yes` | Forecast the next NIFTY 50 close from a daily multivariate feature window spanning market, macro, commodity, and technical signals. |
| `1_stock10_forecasting` | `yes` | `yes` | Multivariate business-day stock forecasting over rolling benchmark windows with point and financial-risk evaluation. |
| `1_stock_OHLCV_forecasting` | `yes` | `yes` | Multistep stock forecasting under a fixed supervised benchmark with direct point-forecast scoring. |
| `1_treasury_variant1_forecasting` | `yes` | `yes` | This task involves multi-horizon forecasting of short-term U.S. Treasury yields to analyze monetary policy sensitivity. The objective is to predict the evolution of the short end of the yield curve using a multivariate time-series approach. |
| `1_treasury_variant2_forecasting` | `yes` | `yes` | This task involves multi-horizon forecasting of the U.S. Treasury yield curve to analyze long-term interest rate trends and risk premia. It focuses on the evolution of intermediate and long-term maturities to capture structural shifts in the yield curve. |
| `1_treasury_variant3_forecasting` | `yes` | `yes` | This task involves multivariate full-curve forecasting of US Treasury yields to capture cross-maturity spillover dynamics. The objective is to model the evolution of the entire yield curve surface using 10 distinct maturities. |
| `1_treasury_variant4_forecasting` | `yes` | `yes` | This task involves forecasting the evolution of the U.S. Treasury yield curve to capture interest rate dynamics within the post-2008 financial regime. It is a multivariate time-series forecasting problem focused on predicting future yield levels across seven specific maturities. |
| `1_treasury_variant5_forecasting` | `yes` | `yes` | This task involves forecasting the dynamics of the U.S. Treasury yield curve, specifically focusing on the pre-2008 economic regime. The objective is to predict future yield levels across multiple maturities to capture term structure evolution. |
| `1_treasury_variant6_forecasting` | `yes` | `yes` | This task involves intermediate-segment Treasury curve forecasting with emphasis on 2y-10y spread dynamics and cross-maturity spillovers. The objective is to predict future yield levels across six U.S. Treasury maturities using a multivariate sliding-window setup. |
| `1_treasury_variant7_forecasting` | `yes` | `yes` | This task involves short-window forecasting of the short end of the U.S. Treasury curve to capture fast-moving policy-sensitive rate dynamics. The objective is to predict the next daily yield level across seven short-to-intermediate maturities using multivariate context windows. |
| `1_tsfm_forecasting` | `partial` | `yes` | One-day-ahead stock-level excess-return forecasting under yearly expanding-window retraining, with both statistical and portfolio evaluation. |
| `1_yieldcurve_forecasting` | `yes` | `yes` | Probabilistic one-step-ahead forecasting of full monthly yield curves across multiple countries. |
| `2_crypto_generation` | `yes` | `yes` | Unconditional multivariate crypto-window generation judged by distributional similarity and financial-risk alignment. |
| `2_crypto_logreturn_generation` | `yes` | `yes` | Unconditional crypto sequence generation judged by tail-risk realism rather than pointwise reconstruction. |
| `2_exchange_generation` | `yes` | `yes` | Unconditional multivariate exchange-rate sequence generation under a stylized-fact evaluator. |
| `2_fbm_conditional_generation` | `yes` | `partial` | Generate realistic conditional future paths for a non-Markovian multivariate process under a strict reproducible metric subset. |
| `2_lob_generation` | `yes` | `yes` | Unconditional LOB window generation scored by multivariate statistical similarity across the full feature set. |
| `2_ou_generation` | `yes` | `yes` | Generate scalar OU sample paths whose law matches the held-out synthetic process. |
| `2_roughvol_generation` | `yes` | `yes` | Generate realistic rough-volatility paths whose marginals, temporal dependence, and joint path geometry match the held-out synthetic process. |
| `2_stock_generation` | `yes` | `yes` | Unconditional multivariate stock-window generation judged by statistical and financial realism. |
| `2_stock_logreturn_generation` | `yes` | `yes` | Unconditional stock-sequence generation judged by multivariate stylized-fact realism rather than supervised prediction error. |
| `2_stock_ohlcv_generation` | `yes` | `yes` | Generate normalized 20-day OHLCV windows whose law matches held-out stock windows. |
| `2_tailgan_generation` | `partial` | `partial` | Generate financial scenarios that preserve tail-risk calibration and dependence structure for downstream risk analysis. |
| `2_timegan_stock_generation` | `yes` | `yes` | Generate realistic multivariate stock windows that preserve temporal dynamics under the official post-hoc evaluation family. |
| `2_us5stock_conditional_generation` | `yes` | `partial` | Generate conditional future return paths that preserve cross-asset dependence and short-maturity option-pricing behavior under a strict reproducible metric subset. |

## Packaged data

| task_folder | packaged tensors / splits |
|---|---|
| `1_commodity_logreturn_forecasting` | Packaged tensors: `train (1843, 60, 17) -> (1843, 10, 17)`, `val (398, 60, 17) -> (398, 10, 17)`, `test (398, 60, 17) -> (398, 10, 17)`; 17 daily commodity log-return channels; chronological `70/15/15` split with stride `2`; target is the next `10` daily log-return steps for all channels. |
| `1_commodity_variant1_logreturn_forecasting` | Packaged tensors: `train (3720, 30, 6) -> (3720, 5, 6)`, `val (800, 30, 6) -> (800, 5, 6)`, `test (801, 30, 6) -> (801, 5, 6)`; assets/features: `gold`, `silver`, `platinum`, `palladium`, `copper`, `crude_oil_wti`; lookback `30`, horizon `5`; value semantics `log_return`. |
| `1_commodity_variant2_logreturn_forecasting` | Packaged tensors: `train (1843, 60, 7) -> (1843, 10, 7)`, `val (398, 60, 7) -> (398, 10, 7)`, `test (398, 60, 7) -> (398, 10, 7)`; assets/features: `crude_oil_wti`, `gasoline`, `heating_oil`, `natural_gas`, `gold`, `copper`, `silver`; lookback `60`, horizon `10`; value semantics `log_return`. |
| `1_commodity_variant3_logreturn_forecasting` | Packaged tensors: `train (912, 90, 8) -> (912, 20, 8)`, `val (197, 90, 8) -> (197, 20, 8)`, `test (197, 90, 8) -> (197, 20, 8)`; assets/features: `corn`, `wheat`, `soybeans`, `kc_wheat`, `cocoa`, `coffee`, `sugar`, `cotton`; lookback `90`, horizon `20`; value semantics `log_return`. |
| `1_commodity_variant4_logreturn_forecasting` | Packaged tensors: `train (1897, 20, 6) -> (1897, 2, 6)`, `val (410, 20, 6) -> (410, 2, 6)`, `test (410, 20, 6) -> (410, 2, 6)`; assets/features: `gold`, `silver`, `crude_oil_wti`, `natural_gas`, `copper`, `soybeans`; lookback `20`, horizon `2`; value semantics `log_return`. |
| `1_commodity_variant5_logreturn_forecasting` | Packaged tensors: `train (1232, 45, 9) -> (1232, 15, 9)`, `val (264, 45, 9) -> (264, 15, 9)`, `test (264, 45, 9) -> (264, 15, 9)`; assets/features: `gold`, `silver`, `platinum`, `palladium`, `copper`, `crude_oil_wti`, `gasoline`, `heating_oil`, `natural_gas`; lookback `45`, horizon `15`; value semantics `log_return`. |
| `1_commodity_variant6_logreturn_forecasting` | Packaged tensors: `train (1843, 60, 7) -> (1843, 10, 7)`, `val (398, 60, 7) -> (398, 10, 7)`, `test (399, 60, 7) -> (399, 10, 7)`; assets/features: `cocoa`, `coffee`, `sugar`, `cotton`, `corn`, `wheat`, `soybeans`; lookback `60`, horizon `10`; value semantics `log_return`. |
| `1_commodity_variant7_logreturn_forecasting` | Packaged tensors: `train (3710, 40, 7) -> (3710, 5, 7)`, `val (800, 40, 7) -> (800, 5, 7)`, `test (801, 40, 7) -> (801, 5, 7)`; assets/features: `copper`, `platinum`, `palladium`, `crude_oil_wti`, `heating_oil`, `natural_gas`, `cotton`; lookback `40`, horizon `5`; value semantics `log_return`. |
| `1_crypto_forecasting` | Packaged split tensors over 20 hourly crypto series; lookback `72`, horizon `3`, train stride `12`, val/test stride `3`; `train (507, 72, 20) -> (507, 3, 20)`, `val (439, 72, 20) -> (439, 3, 20)`, `test (439, 72, 20) -> (439, 3, 20)`. |
| `1_crypto_logreturn_forecasting` | Packaged tensors: `train (4022, 72, 12) -> (4022, 12, 12)`, `val (864, 72, 12) -> (864, 12, 12)`, `test (865, 72, 12) -> (865, 12, 12)`; 12 hourly crypto log-return channels; lookback `72`, horizon `12`, stride `3`; target is the next `12` hourly log-return steps for all channels. |
| `1_crypto_variant1_logreturn_forecasting` | Packaged tensors: `train (12123, 24, 6) -> (12123, 1, 6)`, `val (2602, 24, 6) -> (2602, 1, 6)`, `test (2604, 24, 6) -> (2604, 1, 6)`; assets/features: `btc_usd`, `eth_usd`, `bnb_usd`, `xrp_usd`, `sol_usd`, `ada_usd`; lookback `24`, horizon `1`; value semantics `log_return`. |
| `1_crypto_variant2_logreturn_forecasting` | Packaged tensors: `train (6047, 48, 7) -> (6047, 6, 7)`, `val (1299, 48, 7) -> (1299, 6, 7)`, `test (1300, 48, 7) -> (1300, 6, 7)`; assets/features: `link_usd`, `fil_usd`, `aave_usd`, `avax_usd`, `dot_usd`, `atom_usd`, `hbar_usd`; lookback `48`, horizon `6`; value semantics `log_return`. |
| `1_crypto_variant3_logreturn_forecasting` | Packaged tensors: `train (3016, 72, 6) -> (3016, 12, 6)`, `val (648, 72, 6) -> (648, 12, 6)`, `test (649, 72, 6) -> (649, 12, 6)`; assets/features: `doge_usd`, `shib_usd`, `xmr_usd`, `zec_usd`, `ltc_usd`, `bch_usd`; lookback `72`, horizon `12`; value semantics `log_return`. |
| `1_crypto_variant4_logreturn_forecasting` | Packaged tensors: `train (1495, 168, 9) -> (1495, 24, 9)`, `val (323, 168, 9) -> (323, 24, 9)`, `test (323, 168, 9) -> (323, 24, 9)`; assets/features: `btc_usd`, `eth_usd`, `bnb_usd`, `xrp_usd`, `sol_usd`, `ada_usd`, `avax_usd`, `dot_usd`, `link_usd`; lookback `168`, horizon `24`; value semantics `log_return`. |
| `1_crypto_variant5_logreturn_forecasting` | Packaged tensors: `train (12135, 12, 7) -> (12135, 1, 7)`, `val (2602, 12, 7) -> (2602, 1, 7)`, `test (2604, 12, 7) -> (2604, 1, 7)`; assets/features: `btc_usd`, `eth_usd`, `sol_usd`, `bnb_usd`, `xrp_usd`, `trx_usd`, `algo_usd`; lookback `12`, horizon `1`; value semantics `log_return`. |
| `1_crypto_variant6_logreturn_forecasting` | Packaged tensors: `train (997, 168, 8) -> (997, 24, 8)`, `val (215, 168, 8) -> (215, 24, 8)`, `test (216, 168, 8) -> (216, 24, 8)`; assets/features: `btc_usd`, `eth_usd`, `ltc_usd`, `bch_usd`, `etc_usd`, `xmr_usd`, `zec_usd`, `link_usd`; lookback `168`, horizon `24`; value semantics `log_return`. |
| `1_crypto_variant7_logreturn_forecasting` | Packaged tensors: `train (3011, 96, 7) -> (3011, 8, 7)`, `val (649, 96, 7) -> (649, 8, 7)`, `test (650, 96, 7) -> (650, 8, 7)`; assets/features: `avax_usd`, `dot_usd`, `atom_usd`, `hbar_usd`, `algo_usd`, `fil_usd`, `aave_usd`; lookback `96`, horizon `8`; value semantics `log_return`. |
| `1_equity_excess_return_forecasting` | Built from repo-local `equity.csv (6007, 61)` plus `ust_daily_par_yields_bc.csv (6570, 11)` and packaged as `train_X (5760, 42)`, `train_Y (5760,)`, `val_X (2880, 42)`, `val_Y (2880,)`, `test_X (7200, 42)` with aligned `train_meta.csv`, `val_meta.csv`, `test_meta.csv`; each target month contributes exactly 60 asset rows. The 42-feature stack includes return/momentum (`ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`), rolling mean/volatility/downside-volatility, price-vs-moving-average and `drawdown_12m`, `beta_12m` and `idio_vol_12m`, market return features, risk-free and yield-curve level/slope/twist/change features, cross-sectional rank features, sector-relative features, and sector dummies. The target is `log(P_{t+1}/P_t) - rf_{t+1}`. Harbor preserves the paper’s core monthly excess-return row structure and causal timing, but narrows the cross-section, feature panel, and sample coverage. Target months are train `2004-05..2012-04`, validation `2012-05..2016-04`, test `2016-05..2026-04`, scored under fixed yearly walk-forward refits with a 48-month validation block. |
| `1_equity_logreturn_forecasting` | Packaged tensors: `train (2078, 40, 8) -> (2078, 10, 8)`, `val (446, 40, 8) -> (446, 10, 8)`, `test (447, 40, 8) -> (447, 10, 8)`; 8 daily equity log-return channels; lookback `40`, horizon `10`, stride `2`. |
| `1_equity_variant1_logreturn_forecasting` | Packaged tensors: `train (4170, 30, 5) -> (4170, 5, 5)`, `val (896, 30, 5) -> (896, 5, 5)`, `test (898, 30, 5) -> (898, 5, 5)`; assets/features: `xom`, `cvx`, `cop`, `slb`, `oxy`; lookback `30`, horizon `5`; value semantics `log_return`. |
| `1_equity_variant2_logreturn_forecasting` | Packaged tensors: `train (2068, 60, 7) -> (2068, 10, 7)`, `val (446, 60, 7) -> (446, 10, 7)`, `test (447, 60, 7) -> (447, 10, 7)`; assets/features: `bac`, `gs`, `wfc`, `ms`, `axp`, `c`, `blk`; lookback `60`, horizon `10`; value semantics `log_return`. |
| `1_equity_variant3_logreturn_forecasting` | Packaged tensors: `train (4184, 20, 8) -> (4184, 1, 8)`, `val (900, 20, 8) -> (900, 1, 8)`, `test (902, 20, 8) -> (902, 1, 8)`; assets/features: `pg`, `ko`, `pep`, `wmt`, `cost`, `mcd`, `cl`, `kmb`; lookback `20`, horizon `1`; value semantics `log_return`. |
| `1_equity_variant4_logreturn_forecasting` | Packaged tensors: `train (819, 90, 9) -> (819, 20, 9)`, `val (177, 90, 9) -> (177, 20, 9)`, `test (177, 90, 9) -> (177, 20, 9)`; assets/features: `orcl`, `csco`, `ibm`, `adbe`, `nflx`, `intc`, `qcom`, `txn`, `amd`; lookback `90`, horizon `20`; value semantics `log_return`. |
| `1_equity_variant5_logreturn_forecasting` | Packaged tensors: `train (4162, 40, 9) -> (4162, 3, 9)`, `val (898, 40, 9) -> (898, 3, 9)`, `test (900, 40, 9) -> (900, 3, 9)`; assets/features: `cat`, `ge`, `ba`, `hon`, `de`, `unp`, `ups`, `fdx`, `mmm`; lookback `40`, horizon `3`; value semantics `log_return`. |
| `1_equity_variant6_logreturn_forecasting` | Packaged tensors: `train (4170, 30, 8) -> (4170, 5, 8)`, `val (896, 30, 8) -> (896, 5, 8)`, `test (898, 30, 8) -> (898, 5, 8)`; assets/features: `lly`, `unh`, `mrk`, `pfe`, `abt`, `tmo`, `dhr`, `bmy`; lookback `30`, horizon `5`; value semantics `log_return`. |
| `1_equity_variant7_logreturn_forecasting` | Packaged tensors: `train (2073, 50, 6) -> (2073, 10, 6)`, `val (446, 50, 6) -> (446, 10, 6)`, `test (447, 50, 6) -> (447, 10, 6)`; assets/features: `low`, `sbux`, `tjx`, `hd`, `nike`, `dis`; lookback `50`, horizon `10`; value semantics `log_return`. |
| `1_exchange_rate_forecasting` | Packaged multivariate daily split tensors over 8 exchange-rate series; lookback `60`, horizon `20`; `train (5172, 60, 8) -> (5172, 20, 8)`, `val (1438, 60, 8) -> (1438, 20, 8)`, `test (741, 60, 8) -> (741, 20, 8)`; all 8 future series are forecast and scored. |
| `1_fx_variant1_logreturn_forecasting` | Packaged tensors: `train (3809, 30, 6) -> (3809, 5, 6)`, `val (819, 30, 6) -> (819, 5, 6)`, `test (821, 30, 6) -> (821, 5, 6)`; assets/features: `eur_usd`, `gbp_usd`, `usd_jpy`, `usd_chf`, `usd_cad`, `nzd_usd`; lookback `30`, horizon `5`; value semantics `log_return`. |
| `1_fx_variant2_logreturn_forecasting` | Packaged tensors: `train (1887, 60, 7) -> (1887, 10, 7)`, `val (407, 60, 7) -> (407, 10, 7)`, `test (408, 60, 7) -> (408, 10, 7)`; assets/features: `usd_inr`, `usd_mxn`, `usd_krw`, `usd_zar`, `usd_try`, `usd_cny`, `usd_sgd`; lookback `60`, horizon `10`; value semantics `log_return`. |
| `1_fx_variant3_forecasting` | Packaged tensors: `train (4383, 10, 8) -> (4383, 1, 8)`, `val (549, 10, 8) -> (549, 1, 8)`, `test (550, 10, 8) -> (550, 1, 8)`; assets/features: `eur_usd`, `gbp_usd`, `usd_jpy`, `usd_chf`, `usd_cad`, `usd_hkd`, `usd_sgd`, `nzd_usd`; lookback `10`, horizon `1`; value semantics `price_level`. |
| `1_fx_variant4_logreturn_forecasting` | Packaged tensors: `train (747, 90, 8) -> (747, 20, 8)`, `val (161, 90, 8) -> (161, 20, 8)`, `test (162, 90, 8) -> (162, 20, 8)`; assets/features: `usd_cad`, `usd_nok`, `nzd_usd`, `usd_zar`, `usd_mxn`, `eur_usd`, `gbp_usd`, `usd_chf`; lookback `90`, horizon `20`; value semantics `log_return`. |
| `1_fx_variant5_logreturn_forecasting` | Packaged tensors: `train (4345, 45, 9) -> (4345, 3, 9)`, `val (547, 45, 9) -> (547, 3, 9)`, `test (548, 45, 9) -> (548, 3, 9)`; assets/features: `usd_jpy`, `usd_cny`, `usd_krw`, `usd_sgd`, `usd_hkd`, `usd_inr`, `nzd_usd`, `gbp_usd`, `eur_usd`; lookback `45`, horizon `3`; value semantics `log_return`. |
| `1_fx_variant6_forecasting` | Packaged tensors: `train (3823, 20, 7) -> (3823, 2, 7)`, `val (822, 20, 7) -> (822, 2, 7)`, `test (824, 20, 7) -> (824, 2, 7)`; assets/features: `eur_usd`, `usd_chf`, `gbp_usd`, `usd_sek`, `usd_nok`, `usd_cad`, `usd_jpy`; lookback `20`, horizon `2`; value semantics `price_level`. |
| `1_fx_variant7_logreturn_forecasting` | Packaged tensors: `train (754, 60, 10) -> (754, 15, 10)`, `val (162, 60, 10) -> (162, 15, 10)`, `test (163, 60, 10) -> (163, 15, 10)`; assets/features: `usd_try`, `usd_zar`, `usd_mxn`, `usd_inr`, `usd_krw`, `usd_cny`, `usd_sgd`, `usd_hkd`, `usd_jpy`, `eur_usd`; lookback `60`, horizon `15`; value semantics `log_return`. |
| `1_global_index_volatility_forecasting` | Harbor packages Yahoo Finance adjusted-close histories for `^GSPC`, `^GDAXI`, `^FCHI`, `^FTSE`, `^NSEI`, `^N225`, `^KS11`, and `^HSI` into a common-date aligned wide panel `aligned_panel.csv (2993, 25)` and graph splits `train.pkl`, `val.pkl`, `test.pkl`. Each split stores `node_features (N, 21, 8, 1)`, `corr_adj (N, 8, 8)`, `spill_adj (N, 8, 8)`, and `targets (N, 8, 4)`; concretely, the Harbor bundles materialize train `(1605, 21, 8, 1) -> (1605, 8, 4)`, val `(411, 21, 8, 1) -> (411, 8, 4)`, and test `(894, 21, 8, 1) -> (894, 8, 4)`. The realized-volatility proxy is `rv_proxy_t = sqrt(mean(r_{t-k}^2))` over a 21-day window, each sample uses a node-history lookback `w = 21`, graph estimation window `63`, and the agent must emit `predictions.npy (894, 8, 4)`. |
| `1_lob_forecasting` | Packaged normalized split tensors over 14 LOB-derived channels; lookback `72`, horizon `3`, train stride `12`, val/test stride `3`; `train (86, 72, 14) -> (86, 3, 14)`, `val (79, 72, 14) -> (79, 3, 14)`, `test (79, 72, 14) -> (79, 3, 14)`; only scored subset is `open, high, low, close, vwap`, and those scored channels are inverse-transformed back to their original scale before evaluation. Packaged preprocessing is `log1p(abs(x))` on the six heavy-tailed count/volume fields followed by train-only channel-wise standardization. |
| `1_nifty50_multifactor_forecasting` | Harbor packages an aligned daily multivariate panel as `merged_panel.csv` and fixed tensors `train_X (2916, 30, 8)`, `train_Y (2916,)`, `test_X (729, 30, 8)`, `test_Y (729,)`, plus `dataset_metadata.json`. Each 30-step window contains `NIFTY50`, `S&P500`, `USDINR`, crude oil, gold, India VIX, `MACD`, and `RSI`; the target is next-day NIFTY 50 close. |
| `1_stock10_forecasting` | Packaged split tensors over 10 equity channels; lookback `60`, horizon `3`, train stride `1`, val/test stride `3`; `train (817, 60, 10) -> (817, 3, 10)`, `val (62, 60, 10) -> (62, 3, 10)`, `test (63, 60, 10) -> (63, 3, 10)`. |
| `1_stock_OHLCV_forecasting` | Packaged five-channel OHLCV split tensors over `Open`, `High`, `Low`, `Close`, `Volume`; lookback `60`, horizon `10`, stride `1`; `train (2450, 60, 5) -> (2450, 10, 5)`, `val (668, 60, 5) -> (668, 10, 5)`, `test (360, 60, 5) -> (360, 10, 5)`. The packaged benchmark uses bordered chronological windowing with nominal `70/20/10` split proportions and the fixed 5-channel raw-value representation. |
| `1_treasury_variant1_forecasting` | Packaged tensors: `train (4575, 20, 6) -> (4575, 5, 6)`, `val (981, 20, 6) -> (981, 5, 6)`, `test (982, 20, 6) -> (982, 5, 6)`; assets/features: `bc_3m`, `bc_6m`, `bc_1y`, `bc_2y`, `bc_3y`, `bc_5y`; lookback `20`, horizon `5`; value semantics `yield_level`. |
| `1_treasury_variant2_forecasting` | Packaged tensors: `train (906, 60, 6) -> (906, 10, 6)`, `val (196, 60, 6) -> (196, 10, 6)`, `test (196, 60, 6) -> (196, 10, 6)`; assets/features: `bc_5y`, `bc_7y`, `bc_10y`, `bc_20y`, `bc_30y`, `bc_3y`; lookback `60`, horizon `10`; value semantics `yield_level`. |
| `1_treasury_variant3_forecasting` | Packaged tensors: `train (5226, 30, 10) -> (5226, 1, 10)`, `val (657, 30, 10) -> (657, 1, 10)`, `test (657, 30, 10) -> (657, 1, 10)`; assets/features: `bc_3m`, `bc_6m`, `bc_1y`, `bc_2y`, `bc_3y`, `bc_5y`, `bc_7y`, `bc_10y`, `bc_20y`, `bc_30y`; lookback `30`, horizon `1`; value semantics `yield_level`. |
| `1_treasury_variant4_forecasting` | Packaged tensors: `train (2278, 40, 7) -> (2278, 5, 7)`, `val (491, 40, 7) -> (491, 5, 7)`, `test (491, 40, 7) -> (491, 5, 7)`; assets/features: `bc_2y`, `bc_3y`, `bc_5y`, `bc_7y`, `bc_10y`, `bc_20y`, `bc_30y`; lookback `40`, horizon `5`; value semantics `yield_level`. |
| `1_treasury_variant5_forecasting` | Packaged tensors: `train (2278, 40, 7) -> (2278, 5, 7)`, `val (491, 40, 7) -> (491, 5, 7)`, `test (491, 40, 7) -> (491, 5, 7)`; assets/features: `bc_2y`, `bc_3y`, `bc_5y`, `bc_7y`, `bc_10y`, `bc_20y`, `bc_30y`; lookback `40`, horizon `5`; value semantics `yield_level`. |
| `1_treasury_variant6_forecasting` | Packaged tensors: `train (4582, 15, 6) -> (4582, 3, 6)`, `val (983, 15, 6) -> (983, 3, 6)`, `test (984, 15, 6) -> (984, 3, 6)`; assets/features: `bc_2y`, `bc_3y`, `bc_5y`, `bc_7y`, `bc_10y`, `bc_20y`; lookback `15`, horizon `3`; value semantics `yield_level`. |
| `1_treasury_variant7_forecasting` | Packaged tensors: `train (4594, 5, 7) -> (4594, 1, 7)`, `val (985, 5, 7) -> (985, 1, 7)`, `test (986, 5, 7) -> (986, 1, 7)`; assets/features: `bc_3m`, `bc_6m`, `bc_1y`, `bc_2y`, `bc_3y`, `bc_5y`, `bc_7y`; lookback `5`, horizon `1`; value semantics `yield_level`. |
| `1_tsfm_forecasting` | Harbor packages one supervised row per asset-date from an open U.S. proxy universe: `train_X (56346, 252)`, `train_Y (56346,)`, `val_X (29188, 252)`, `val_Y (29188,)`, `test_X (163719, 252)` with aligned metadata columns `feature_start_date`, `feature_end_date`, `target_date`, `target_year`, `asset`, `sector`, `price_t`. Each feature row is the previous 252 daily excess returns for one stock, computed from daily log returns minus a dailyized 3-month Treasury rate and winsorized to `[-1, 1]`; the target is the stock's next-day excess return. Harbor preserves the paper’s core daily excess-return row definition, 252-day lookback, one-day horizon, winsorization logic, and yearly expanding evaluation style, but narrows the panel to an open U.S. proxy universe and excludes the paper’s broader cross-market / augmentation branches. The nominal universe has 60 names, but the fixed `price_t >= 5` screen leaves 59 active assets in train/validation and 60 in test. The yearly expanding schedule is fixed: train `2007-2010`, validation `2011-2012`, test `2013-2023`, with yearly refits defined by metadata. |
| `1_yieldcurve_forecasting` | Harbor packages official monthly multi-country term structures as `train.pkl` with `contexts (1734, 10, 150)`, `targets (1734, 150)`, `country_ids (1734,)` and `test.pkl` with `contexts (408, 10, 150)`, `targets (408, 150)`, `country_ids (408,)`, plus `scalers.pkl` and `dataset_metadata.json`. Predictions must be written as `predictions.npz` with `central/lower/upper`, each `(408, 150)`. |
| `2_crypto_generation` | Raw-CSV generation benchmark over `crypto.csv (8784, 21)` with one `time` column and 20 ordered asset channels `ARKMUSDT` through `ALPINEUSDT`; fixed split `6148 / 1317 / 1319`, `seq_len = 60`, unit stride, train-only normalization; resulting windows `train (6089, 60, 20)`, `val (1317, 60, 20)`, `test (1319, 60, 20)`, with `test/features` represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(1319, 60, 20)` evaluation pool. Harbor reconstruction matches the TSAgentX source task exactly. |
| `2_crypto_logreturn_generation` | Packaged unconditional crypto return windows `crypto_train_data.pkl (7004, 24, 10)`, `crypto_val_data.pkl (875, 24, 10)`, `crypto_test_data.pkl (876, 24, 10)`, with `test/features` represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(876, 24, 10)` evaluation pool. These packaged tensors are byte-for-byte identical to the local TS-Agent source benchmark files. |
| `2_exchange_generation` | Packaged unconditional exchange-rate level windows `exchange_train_data.pkl (2021, 7, 8)`, `exchange_val_data.pkl (253, 7, 8)`, `exchange_test_data.pkl (253, 7, 8)`, with `test/features` represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(253, 7, 8)` evaluation pool. These packaged tensors are byte-for-byte identical to the local TS-Agent source benchmark files. |
| `2_fbm_conditional_generation` | Harbor packages the verified `H = 0.25` synthetic source as split dictionaries with `contexts` and `targets`: train `contexts (10000, 6, 3), targets (10000, 5, 3)`, test `contexts (10000, 6, 3), targets (10000, 5, 3)`. The full source train/test sets are preserved directly, and the agent scores `samples.npy (10000, 128, 5, 3)`. |
| `2_lob_generation` | Raw `lob.csv (1581, 15)` benchmark with one time column and 14 ordered feature channels `open, high, low, close, volume, vwap, buy_volume, sell_volume, n_trades, n_new_orders, n_cancels, price_range, net_order_flow, trade_imbalance`; heavy-tailed volume/count fields `volume, buy_volume, sell_volume, n_trades, n_new_orders, n_cancels` are transformed by `log1p(abs(x))`; chronological split `1106 / 237 / 238`; `seq_len = 60`, unit stride; windows `train (1047, 60, 14)`, `val (237, 60, 14)`, `test (238, 60, 14)`, with `test/features` represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(238, 60, 14)` evaluation pool. Harbor reconstruction matches the TSAgentX source task exactly. |
| `2_ou_generation` | Harbor rebuilds the exact local OU source `ou_timeseries.csv (10000, 64)` into tensors `train.pkl (8000, 64, 1)`, `val.pkl (1000, 64, 1)`, `test.pkl (1000, 64, 1)` and requires `samples.npy (1000, 64, 1)`. The packaged tensors are exactly equal to the source table after reshaping `t_0..t_63` into `(10000, 64, 1)` and slicing sequentially `8000 / 1000 / 1000`; `test/features` is represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(1000, 64, 1)` evaluation pool. |
| `2_roughvol_generation` | Harbor packages the exact repo-local rough-volatility source tensor `rough_volatility_paths.npy (20000, 200, 2)` as `train.pkl (16000, 200, 2)`, `val.pkl (2000, 200, 2)`, `test.pkl (2000, 200, 2)` with channels `log_price`, `log_volatility`; the agent emits `samples.npy (2000, 200, 2)`. Harbor adds a validation slice by splitting the paper’s 20k-path synthetic corpus into deterministic `16000 / 2000 / 2000`, and `test/features` is represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(2000, 200, 2)` evaluation pool. |
| `2_stock_generation` | Raw `stock_10.csv (1257, 11)` benchmark with one time column and 10 ordered stock channels `aapl, amzn, goog, jnj, jpm, meta, msft, nvda, tsla, v`; chronological split `879 / 188 / 190`; `seq_len = 60`, unit stride, train-only normalization; windows `train (820, 60, 10)`, `val (188, 60, 10)`, `test (190, 60, 10)`, with `test/features` represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(190, 60, 10)` evaluation pool. Harbor reconstruction matches the TSAgentX source task exactly. |
| `2_stock_logreturn_generation` | Packaged unconditional stock log-return windows `stock_train_data.pkl (1267, 5, 8)`, `stock_val_data.pkl (158, 5, 8)`, `stock_test_data.pkl (159, 5, 8)`. These packaged tensors are byte-for-byte identical to the local TS-Agent source benchmark files. |
| `2_stock_ohlcv_generation` | Harbor redownloads the paper’s ten-stock Yahoo Finance OHLCV panel and packages pooled single-stock windows as `train.pkl (900, 20, 5)` and `test.pkl (230, 20, 5)` with feature order `open, close, high, low, volume`. Each ticker is min-max normalized per channel over its full 2013-01-01 to 2021-12-31 history, then the full history is truncated into non-overlapping 20-day windows before a chronological `80% / 20%` train/test split on windows. This keeps all complete windows and is closer to the paper than the earlier row-first split; `test/features` is represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(230, 20, 5)` evaluation pool. The agent emits `samples.npy (230, 20, 5)`. |
| `2_tailgan_generation` | Harbor fixes the synthetic branch as an unconditional scenario-generation benchmark with `train.pkl (50000, 100, 5)`, `val.pkl (5000, 100, 5)`, and `test.pkl (5000, 100, 5)`. Each sample is a 5-channel return scenario of length 100; there is no conditioning context, and the agent must emit `samples.npy (5000, 100, 5)`. Harbor does not include the paper's separate real intraday equity branch, and `test/features` is represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(5000, 100, 5)` evaluation pool. |
| `2_timegan_stock_generation` | Harbor downloads the official repo raw file `stock_data.csv` exactly, applies the repo preprocessing recipe, and packages `train.pkl (3661, 24, 6)` and `test.pkl (3661, 24, 6)` from the same in-distribution stock-window pool, plus `dataset_metadata.json`. The agent must emit `samples.npy (3661, 24, 6)`, and `test/features` is represented in Harbor as an unconditional placeholder while the held-out verifier reference carries the true `(3661, 24, 6)` evaluation pool. |
| `2_us5stock_conditional_generation` | Harbor packages verified five-stock daily log-return windows as split dictionaries: train `contexts (1103, 5, 5), targets (1103, 5, 5)`, test `contexts (276, 5, 5), targets (276, 5, 5)`, with ticker order `AAPL, LMT, JPM, AMZN, PG`. The underlying source matches the paper’s 2010-01-01 to 2020-12-31, window-length-10, stride-2, `80% / 20%` train/test construction directly, and the agent scores `samples.npy (276, 128, 5, 5)`. |

