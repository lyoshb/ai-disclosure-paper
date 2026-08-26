# Talk or Deployment? AI Disclosure and the Limits of Text-Based Adoption Measures

Replication code and data for Balabanov (2026).

**Paper (SSRN):** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7339038  
**Paper (Zenodo):** https://doi.org/10.5281/zenodo.22066569  
**Data:** https://doi.org/10.5281/zenodo.22066664  
**Author:** Oleksii Balabanov ([ORCID](https://orcid.org/0009-0009-9039-3699))

## Summary

The share of S&P 500 firms mentioning artificial intelligence in their annual report rose from 38.6% in fiscal 2018 to 96.7% in fiscal 2025. This paper asks whether that increase measures adoption or disclosure.

Using 9,177 paragraphs extracted from Form 10-K filings, each AI mention is classified as describing a deployed system, an intention, legal risk language, governance content, or a residual category. The classification is validated against 250 manually labelled paragraphs.

Three findings: the increase is driven by firms that began disclosing AI after 2022 and describe little deployment; among earlier adopters, deployment is flat while risk language grew fivefold; and no text-based measure predicts firm performance.

## Data sources

All source data are public.

| Source | Content |
|---|---|
| SEC EDGAR submissions API | Form 10-K filings, 2018–2025 |
| SEC XBRL Company Facts API | Assets, net income, revenue, liabilities |
| Wikipedia | S&P 500 index constituents |

Raw filings are not included in this repository: they total several gigabytes and are freely available from EDGAR. The scripts download and cache them locally.

## Pipeline

Scripts are listed in execution order. Run from the repository root.

| Script | Purpose | Output |
|---|---|---|
| `sp500.py` | Match index constituents to SEC identifiers | `sp500_cik.csv` |
| `pipeline.py` | Core functions: filing retrieval, HTML parsing, caching | — |
| `harvest.py` | Download filings, extract AI paragraphs | `paragraphs_all.csv` |
| `panel.py` | Firm-year counts including zeros | `panel_counts.csv` |
| `financials.py` | Financial variables from XBRL | `financials.csv` |
| `merge_panel.py` | Merge text and financial data | `panel_final.csv` |
| `validate_deepseek.py` | Classify the 250 manually labelled paragraphs | `labels_deepseek.csv` |
| `classify_all.py` | Classify the full corpus | `labels_full.csv` |
| `build_measures.py` | Aggregate to firm-year measures | `measures.csv` |
| `regressions.py` | Main specifications | — |
| `robustness.py` | Placebo splits, sector exclusion, alternatives | — |
| `make_tables.py` | Regression tables | `table4_main.csv`, `table5_robustness.csv` |

`pipeline.py` is imported by other scripts and is not run directly.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The SEC API requires a User-Agent header identifying the requester. Edit the `HEADERS` constant in `pipeline.py` with your name and email before running.

Classification requires an API key, supplied through an environment variable:

```bash
export DEEPSEEK_API_KEY="your_key"
```

## Classification

Paragraphs are classified by `deepseek-chat` at temperature 0. The full prompt is reproduced in Appendix A of the paper and in `classify_all.py`.

Validation results, from 250 paragraphs labelled manually by the author:

| Model | 5-category κ | Binary κ |
|---|---|---|
| Llama-3.3-70B | 0.573 | 0.702 |
| DeepSeek-chat | 0.582 | 0.658 |
| GLM-4-Air | 0.531 | 0.663 |
| GLM-4.6 | 0.532 | 0.641 |
| GLM-4-Flash | 0.402 | 0.663 |
| Manual, intra-rater | — | 0.776 |

The final row is the author's agreement with own labels on a blind relabelling of 20 paragraphs, and serves as an upper bound on achievable agreement.

## Repository structure

```
code/       analysis scripts
data/       processed panels, labels, output tables
figures/    figures as they appear in the paper
```

## Notes on reproduction

Running the full pipeline downloads roughly 4,000 filings and takes several hours. Filings are cached in `cache/`, so interrupted runs resume where they stopped. `classify_all.py` writes incrementally and skips already-classified paragraphs on restart.

Model names and availability change over time. Results reported in the paper were produced in August 2026.

## Citation

```
Balabanov, O. (2026). Talk or Deployment? AI Disclosure and the Limits
of Text-Based Adoption Measures. SSRN Working Paper 7339038.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7339038
```

Archived version with DOI:

```
Balabanov, O. (2026). Talk or Deployment? AI Disclosure and the Limits
of Text-Based Adoption Measures. Zenodo.
https://doi.org/10.5281/zenodo.22066569
```

## License

Code is released under the MIT License. Data are released under CC BY 4.0.
