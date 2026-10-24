# ASTROVET Data Dictionary

**Version:** 1.0.1  
**Last Updated:** 2026-10-21  

---

## 1. Candidate Record (`CandidateRecord`)

| Field | Type | Unit | Description |
|---|---|---|---|
| `candidate_id` | `str` | None | Canonical compound key: `<tic_id>-<planet_num>-S<sector_run>` (e.g. `390651552-1-S23`) |
| `tic_id` | `int` | None | Target star identifier in TESS Input Catalog |
| `planet_num` | `int` | None | SPOC integer transit candidate number |
| `sector_run` | `str` | None | TESS observing sector or multi-sector span (e.g. `23`, `1-67`) |
| `period_days` | `float` | days | Orbital period |
| `epoch_bjd_tdb` | `float` | days | Transit mid-point in BTJD (BJD - 2457000.0) |
| `duration_hours`| `float` | hours | Transit duration (contact I to IV) |
| `depth_ppm` | `float` | ppm | Transit depth in parts per million ($10^{-6}$) |
| `gaia_ruwe` | `float` | None | Gaia DR3 Renormalized Unit Weight Error |
| `label_raw` | `str` | None | Raw disposition string from upstream catalog |
| `label_normalized` | `Disposition` | None | Normalized disposition enum (`CONFIRMED_PLANET`, `PLANET_CANDIDATE`, `FALSE_POSITIVE`, `FALSE_ALARM`, `AMBIGUOUS_UNRESOLVED`) |

---

## 2. Upstream Baseline Prediction (`BaselinePrediction`)

| Field | Type | Range / Nullability | Description |
|---|---|---|---|
| `upstream_model` | `str` | Non-null string | Model identifier: `ExoMiner++ on TESS 2-min S1-S67` (Release tag `v1.0.0`, commit `d877a23`) |
| `release_tag` | `str` | String (`v1.0.0`) | Verified NASA software release tag |
| `release_commit` | `str` | String (`d877a23`) | Verified Git commit hash |
| `score` | `Optional[float]` | `[0.0, 1.0]` or `None` | Overall ExoMiner confidence score; `None` when `status=NOT_FOUND` (never fabricated as 0.50) |
| `score_pc` | `Optional[float]` | `[0.0, 1.0]` or `None` | Score for Planet Candidate class |
| `score_afp` | `Optional[float]` | `[0.0, 1.0]` or `None` | Score for Astrophysical False Positive class |
| `score_ntp` | `Optional[float]` | `[0.0, 1.0]` or `None` | Score for Non-Transiting Phenomenon (false alarm) class |
| `score_uncertainty` | `Optional[float]` | `[0.0, 1.0]` or `None` | Standard deviation across ensemble models; `None` when `status=NOT_FOUND` |
| `status` | `str` | `SUCCESS`, `NOT_FOUND`, `ERROR` | Baseline query status |
| `execution_mode` | `str` | `CATALOG`, `SYNTHETIC_FIXTURE`, `CONTAINER_PROCESS` | Execution mode: `CATALOG` lookup against published tables, `SYNTHETIC_FIXTURE`, or `CONTAINER_PROCESS` (Podman/Docker containerized execution) |
| `is_synthetic` | `bool` | `True` / `False` | Flag indicating synthetic demonstration or mock test fixture |
| `checkpoint_overlap_status` | `str` | `KNOWN_DISJOINT`, `KNOWN_TRAIN`, `UNKNOWN` | Checkpoint training data audit status |
| `raw_output` | `dict` | Dictionary | Raw catalog record or error dictionary |

---

## 3. Physics Diagnostics (`PhysicsDiagnostics`)

| Field | Type | Unit | Description |
|---|---|---|---|
| `transit_depth_fitted_ppm` | `float` | ppm | Fitted depth from approximate transit template |
| `transit_duration_fitted_hours` | `float` | hours | Fitted duration (contact I to IV) |
| `transit_snr` | `float` | None | Fitted depth / depth uncertainty |
| `fit_chi2_reduced` | `float` | None | Goodness of fit (reduced chi-squared of template residuals) |
| `odd_even_depth_diff_ppm` | `float` | ppm | Depth difference between odd and even transits |
| `odd_even_t_statistic` | `float` | None | Significance of odd-even difference |
| `odd_even_consistent` | `bool` | None | True if odd and even depths are statistically indistinguishable |
| `secondary_depth_ppm` | `float` | ppm | Flux drop detected in phase window around 0.5 |
| `secondary_eclipse_detected` | `bool` | None | True if secondary depth is significant ($>3\sigma$ and $>8\%$ of primary) |
| `transit_count_observed` | `int` | None | Number of distinct transit events observed |
| `transit_depth_scatter_ppm` | `float` | ppm | Standard deviation of individual transit depths across events |
| `transit_timing_scatter_minutes`| `float` | minutes | Standard deviation of individual transit timing offsets |
| `plausibility_all_passed` | `bool` | None | Validates $P>0$, $T_{dur}<P$, and physically bounded geometry |
| `modality_availability` | `dict` | None | Explicit boolean availability map for each diagnostic branch |
