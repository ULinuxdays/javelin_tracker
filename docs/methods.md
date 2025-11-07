## Javelin Tracker Methods Note (v0.1.0)

_Last updated: 2025-02-14. Cite this document with the main software citation in `CITATION.cff`._

### 1. Overview

Javelin Tracker is a reproducible command-line pipeline for capturing throwing sessions, computing workload metrics, and generating decision-support reports for coaches and researchers. This note enumerates the core algorithms, data transformations, and reference literature so the software can be cited as a transparent “methods paper” alongside traditional publications.

### 2. Data ingestion & schema

- **Capture** – The CLI `log` command accepts ISO&nbsp;8601 dates, best throw distance, per-throw series, RPE (1–10), duration (minutes), notes, tags, `athlete`, and optional `team`. Inputs are validated via the `Session` dataclass: distances must be positive floats, RPE values are clamped to `[1, 10]`, and durations must be non-negative real numbers.
- **Storage** – Sessions persist in JSON (`data/sessions.json` by default). Each record stores the multi-athlete fields plus free-text notes; exports append hashed identifiers for anonymised sharing. Environment variables `JAVELIN_TRACKER_DATA_DIR` or `JAVELIN_TRACKER_SESSIONS_FILE` redirect storage for secure deployments.
- **Normalization** – `sessions_to_dataframe` converts the JSON list into a pandas `DataFrame`, ensuring `date` is timezone-naïve UTC, calculating `throws_count`, attaching `athlete`/`team`, and dropping malformed entries. This is the canonical in-memory representation for all downstream routines.

### 3. Session workload (session-RPE)

We implement the session-RPE method (Foster et&nbsp;al., 2001):

\[
\text{session load}_i = \text{RPE}_i \times \text{duration}_i \text{ (minutes)}
\]

Loads are expressed in arbitrary units (AU). When no duration is supplied, the load defaults to zero but the record remains accessible for throw analytics.

### 4. Throw performance metrics

- **Best throw** – Direct user input or, if absent, the maximum of the provided throw series. Throws are validated individually to guard against string/float coercion errors.
- **Average throw** – Added during export as \(\bar{x}_i = \frac{1}{n}\sum_{j=1}^{n} \text{throw}_{ij}\) for researchers interested in central tendency.
- **Throw counts** – Stored alongside each record to quantify weekly or monthly volume.

### 5. Acute:Chronic Workload Ratio (Rolling)

Following Haddad et&nbsp;al. (2017) and Science for Sport guidance, the rolling ratio compares a 7-day acute load against a 28-day chronic baseline:

\[
\text{ACWR}_{\text{rolling}}(t) = \frac{\sum_{d=t-6}^{t} \text{load}_d}{\frac{1}{28}\sum_{d=t-27}^{t} \text{load}_d}
\]

The denominator uses the mean (not sum) of the chronic window to avoid unit mismatches. The calculation requires at least 4 weeks of history before stabilising.

### 6. ACWR via Exponentially Weighted Moving Averages

To emphasize recency while preserving historical context (Williams et&nbsp;al., 2017), we maintain exponentially weighted means with spans of 7 and 28 days:

\[
\begin{aligned}
\text{EWMA}_{7}(t) &= \alpha_{7}\cdot \text{load}(t) + (1-\alpha_{7})\cdot \text{EWMA}_{7}(t-1) \\
\text{EWMA}_{28}(t) &= \alpha_{28}\cdot \text{load}(t) + (1-\alpha_{28})\cdot \text{EWMA}_{28}(t-1)
\end{aligned}
\qquad \text{where }\alpha_{N} = \tfrac{2}{N+1}
\]

\[
\text{ACWR}_{\text{EWMA}}(t) = \frac{\text{EWMA}_{7}(t)}{\text{EWMA}_{28}(t)}
\]

The EWMA approach smooths noisy load spikes yet reacts faster than a simple moving average.

### 7. Risk stratification

We categorise risk based on consensus bands (Gabbett, 2016; Carey et&nbsp;al., 2018):

- **LOW**: \(0.8 \leq \text{ACWR} \leq 1.3\)
- **MODERATE**: \(1.3 < \text{ACWR} \leq 1.5\)
- **HIGH**: \(\text{ACWR} > 1.5\) or \(\text{ACWR} < 0.8\)

Daily records inherit the most severe risk flag from either rolling or EWMA ratios. Summaries surface all high-risk dates so practitioners can audit spikes.

### 8. Weekly & monthly aggregation

- **Weekly pipeline** – Sessions are grouped by ISO week label. We aggregate session counts, best throw, total load, throw volume, and mean RPE. Weekly ACWR metrics derive from the metrics tables described above. Personal bests are tracked per group to highlight standout periods.
- **Monthly view** – For `summary --by month`, weekly aggregates roll up into calendar months while preserving weighted averages (e.g., load is summed, RPE is averaged by throw counts).
- **Reports** – The PDF generator pairs tabular stats with matplotlib plots of session load and a conceptual ACWR-risk curve to support coaching meetings.

### 9. Open data exports & metadata

`javelin export --to <path>` now emits four synchronized artifacts:

1. CSV – tabular data compatible with spreadsheets.
2. Parquet – columnar format for analytics stacks.
3. JSON – full-fidelity records (ISO dates, floats, text) for interchange.
4. Metadata JSON – run metadata containing the application version, export timestamp (UTC), row count, schema, and environment pointers.

Each exported row also embeds the `app_version` and `exported_at` fields so downstream analyses can trace algorithm versions that generated the data. This satisfies FAIR-style provenance expectations and enables reproduction even if algorithms change in future releases.

### 10. Versioning, releases, and citation

- Semantic versions live in `pyproject.toml`, `CITATION.cff`, and the `CHANGELOG.md`.
- Tagged releases on GitHub should be mirrored to Zenodo to mint a DOI; include the DOI in future `CITATION.cff` updates.
- When publishing academic work, cite the software using the metadata in `CITATION.cff` and reference this methods note for algorithmic detail.

### References

- Carey, D. L., et&nbsp;al. (2018). *The acute:chronic workload ratio predicts injury*. **British Journal of Sports Medicine**, 52(5), 328–334.  
- Foster, C., et&nbsp;al. (2001). *A new approach to monitoring exercise training*. **Journal of Strength and Conditioning Research**, 15(1), 109–115.  
- Gabbett, T. J. (2016). *The training–injury prevention paradox*. **British Journal of Sports Medicine**, 50(5), 273–280.  
- Haddad, M., et&nbsp;al. (2017). *Session-RPE method for monitoring training load*. **Frontiers in Physiology**, 8, 142.  
- Science for Sport. (2023). *Acute:Chronic Workload Ratio (ACWR)*. https://www.scienceforsport.com/acute-chronic-workload-ratio/  
- Williams, S., et&nbsp;al. (2017). *Monitoring what matters: A systematic process for selecting training-load metrics*. **International Journal of Sports Physiology and Performance**, 12(S2), S2-26–S2-34.
