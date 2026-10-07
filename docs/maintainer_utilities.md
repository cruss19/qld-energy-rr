# Maintainer utilities and reproducibility boundaries

The public repository separates commands that work in a clean clone from
publication-maintenance utilities that depend on undistributed historical
artifacts.

## Clean-clone checks

From the repository root:

```powershell
python -m pytest -m "not source_data" -q
```

After the registered provider data are present, the full local suite is:

```powershell
python -m pytest -q
```

## Maintainer-only utilities

The following tools are retained for auditability but are not clean-clone
commands:

- `tools/verify_rr_public_parity.py` requires the undistributed historical
  feature cache and verifies the public feature reconstruction against it.
- `tools/preflight_reference_experiments.py` requires the local processed
  reference frame and undistributed row-level reference predictions.
- `tools/run_reference_models.py` reports preserved reference-run status from
  undistributed prediction artifacts; it does not train models.
- `tools/build_reference_section.py` rebuilds publication tables and figures
  from undistributed row-level reference and model predictions.
- `tools/build_tcn_development_notebooks.py` rebuilds development evidence from
  undistributed completed run artifacts and never trains or scores 2020.
- `tools/prepare_reference_notebook_assets.py` migrates explicitly supplied
  historical source artifacts into the bounded 2015--2020 publication form.
- `tools/build_public_run_identity_manifest.py` rebuilds the sanitized public
  model-member identity manifest from undistributed final run records.

Generated public manifests disclose identities, hashes, counts, and validation
status without publishing provider data, checkpoints, row-level predictions,
machine-specific paths, or later private research artifacts.
