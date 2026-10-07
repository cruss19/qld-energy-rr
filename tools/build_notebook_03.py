"""Build the finalized RR feature-contract notebook."""

from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cells = [
        nbf.v4.new_markdown_cell(
            "# 03 — Feature Development and Reference Contracts\n\n"
            "This notebook records the frozen public feature boundary used by the 2015–2020 "
            "reference and TCN evaluations. Development is confined to 2015–2019; 2020 is the "
            "fixed final gate. It reports completed contract evidence rather than proposing future work."
        ),
        nbf.v4.new_markdown_cell(
            "## Leakage controls\n\n"
            "- Targets are six demand changes at 5–30 minutes and never enter the feature tensors.\n"
            "- All fitted transforms use training-fold rows only.\n"
            "- AEMO histories and lags are causal; weather joins are backward-only.\n"
            "- Monthly SOI changes at the preserved 19:00 Brisbane availability boundary on the first day of the following month.\n"
            "- The deterministic 2021-01-01 holiday date is used only to form the 2020-12-31 day-before flag; no 2021 observation enters the RR.\n"
            "- `Population_2` and `DEMANDFORECAST` are forbidden learned inputs."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nimport yaml\n"
            "from IPython.display import display\n"
            "ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p / 'config' / 'public_feature_contract.yml').exists())\n"
            "contract = yaml.safe_load((ROOT / 'config' / 'public_feature_contract.yml').read_text())\n"
            "summary = json.loads((ROOT / 'outputs' / '03_feature_contract' / 'feature_preflight_summary.json').read_text())\n"
            "display(pd.DataFrame([summary]))"
        ),
        nbf.v4.new_markdown_cell(
            "## Frozen tensor routes\n\n"
            "The exact public model uses 505 five-minute states for each temporal branch. Demand has 7 channels; "
            "system has 16; climate has 2; and each of five regions has 10. At the forecast origin, 55 encoded "
            "calendar values and one annual statewide-population scalar join at late fusion."
        ),
        nbf.v4.new_code_cell(
            "feature_table = pd.read_csv(ROOT / 'data' / 'contracts' / 'rr_public_feature_contract_v1.csv')\n"
            "display(feature_table.groupby('context', dropna=False).size().rename('expanded_features').to_frame())\n"
            "display(feature_table)"
        ),
        nbf.v4.new_markdown_cell(
            "## Exact reconstruction evidence\n\n"
            "`tools/verify_rr_public_parity.py` compares the public artifact with the frozen reconstruction cache. "
            "All 96 shared frozen columns match across 629,273 public origins. The six terminal timestamps from "
            "23:30 through 23:55 on 2020-12-31 are intentionally absent because all six future targets cannot be observed."
        ),
        nbf.v4.new_markdown_cell(
            "## Reference execution status\n\n"
            "The five original and three probabilistic linear references are complete 30-minute endpoint models. "
            "The original Ridge uses 55 base predictors and its preserved sparse_cg implementation; it is not a "
            "flattened TCN tensor. AEMO is retained as the separate six-horizon external benchmark."
        ),
        nbf.v4.new_code_cell(
            "status = pd.read_csv(ROOT / 'outputs' / '10_final_2020_evaluation' / 'reference_execution_status.csv')\n"
            "display(status)"
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "qld-energy-rr", "language": "python", "name": "python3"}},
    )
    nbf.write(notebook, root / "notebooks/03_feature_development_and_reference_contracts.ipynb")


if __name__ == "__main__":
    main()
