# Data directory

The RR rebuilds provider data locally. Complete third-party datasets are not
committed to the public repository.

## Directory contract

| Directory | Contents | Git policy |
|---|---|---|
| `raw/` | Immutable source extracts and API snapshots | Ignored |
| `external/` | Manually obtained provider workbooks, archives, and reference tables | Ignored |
| `interim/` | Validated source-specific tables before model-table assembly | Ignored |
| `processed/` | Reproducible modelling tables | Ignored |
| `manifests/` | Source registry, artifact identities, checksums, coverage, and permissions metadata | Tracked after review |
| `schemas/` | Machine-readable source contracts | Tracked |
| `fixtures/` | Small lawful or synthetic test fixtures only | Tracked |

## Safe acquisition controls

Data acquisition is hard-bounded to the RR period. No acquisition mode may
admit observations after 2020-12-31 or a versioned source released after
2020-12-31. Fixed ABS, ASGS and CER files are identified by exact byte size and
SHA-256 before they enter their canonical directories.

The AEMO P5MIN benchmark is acquired separately from dispatch demand. Monthly
`P5MIN_REGIONSOLUTION` archives are filtered to `QLD1`, reduced to the six
5–30 minute horizons, and fingerprinted month by month. Both forecast origins
and target timestamps are hard-capped at 2020-12-31 23:55. The dispatch table's
`DEMANDFORECAST` field is never substituted for this origin-vintage product.

Data acquisition uses two independent environment variables:

- `RR_ACQUISITION_MODE=verify|missing`
- `RR_ALLOW_NETWORK=0|1`

`verify` is the default. It performs no network activity and audits whatever is
already present. `missing` may obtain only absent automatic sources and never
overwrites an existing canonical artifact. There is no snapshot or force mode:
either could introduce a later source vintage into the RR.

Network activity is refused unless `RR_ALLOW_NETWORK=1` is also set. There is
no force-overwrite mode.

From the repository root:

```powershell
# Safe local audit (default)
python tools/acquire_rr_data.py --mode verify

# Obtain only missing automatic sources
$env:RR_ALLOW_NETWORK = "1"
python tools/acquire_rr_data.py --mode missing

```

ASGS 2016 and the CER files frozen in 2020
are exact pinned downloads declared in
`data/manifests/source_registry_2020.yml`. The missing-data command obtains only
those identities. It downloads to a temporary `.partial` file, verifies size
and SHA-256, and only then promotes the file into `data/external/`.

The final public `TCN_starNLL` contract also admits the Bureau of Meteorology
monthly SOI series through October 2020. Only the through-2020 subset is written
to `data/raw/bom/soi_monthly_through_2020.csv`; the two model channels apply the
declared M+2 availability rule. The pinned CER files are transformed into
`data/processed/SGU_Solar_monthly_2015_2020.parquet`, containing five regional
monthly cumulative-capacity series and no post-2020 observation.

Queensland calendar records remain pinned manual inputs. Register them after
placing the exact declared files in their canonical directories:

```powershell
python tools/register_manual_source.py `
  --source-id qld_public_holidays `
  --notes "Exact frozen Queensland records"
```

Valid manual source identifiers are:

- `qld_public_holidays`
- `qld_school_calendar`

Coverage and source-release dates come only from the locked registry and cannot
be supplied or widened at the command line. A supplied retrieval timestamp must
include `Z` or an explicit UTC offset; otherwise the registration time is
recorded and disclosed.

The command refuses undeclared files, checksum or size mismatches, empty source
locations, symlinks, and paths outside the RR. It appends checksum evidence and
immediately re-verifies every current file for that source.

Every acquired artifact is recorded in `data/manifests/artifacts.csv` with a
repository-relative path, provider, retrieval timestamp, byte size, SHA-256
checksum, temporal coverage, and acquisition mode.

After registration, verification reports `not present locally`, `incomplete,
unregistered, or contract mismatch`, or `registered and verified`. The complete
acquisition gate passes only when every required source reports `registered and
verified`.

## Validated preparation outputs

[`notebooks/02_data_validation_and_preparation.ipynb`](../notebooks/02_data_validation_and_preparation.ipynb)
creates five local, reproducible tables after the acquisition gate passes:

| Output | Purpose |
|---|---|
| `data/interim/aemo_qld_5min_validated_2015_2020.parquet` | Source-faithful, unique QLD1 physical dispatch series |
| `data/interim/regional_weather_hourly_validated_2015_2020.parquet` | Source-faithful hourly weather for the five declared regions |
| `data/interim/aemo_p5min_forecasts_2015_2020.parquet` | QLD1 AEMO origin-vintage point forecasts at 5, 10, 15, 20, 25 and 30 minutes |
| `data/processed/rr_demand_calendar_base_5min_2015_2020.parquet` | Five-minute demand/system spine with calendar context and the original annual Queensland population series |

The processed base retains the original annual `total_qld_population` scalar. It changes on 1 April and is standardized using training-fold statistics only.

These outputs are not yet scaled or converted to model tensors. Regional
weather remains hourly and long-form; the notebook verifies that a backward
as-of join never selects a future observation. Distributed-energy allocation,
causal lags, training-only scaling, and final feature geometry are deliberately
reserved for Notebook 03.
