# Third-party data notice

This repository contains original code, documentation, schemas, and compact
project-authored evidence under the repository's MIT licence. That licence does
not grant rights to third-party datasets or override provider terms.

Complete provider datasets are not committed. Users obtain them from their
original providers and are responsible for checking the applicable access,
licensing, attribution, and redistribution terms.

## Provider sources

| Provider or authority | Project use | Public-repository treatment |
|---|---|---|
| Australian Energy Market Operator (AEMO) | Queensland operational demand and P5MIN forecast products | Source identities and acquisition logic are retained; complete source files are excluded. |
| Australian Bureau of Statistics (ABS) | ASGS 2016 geography and population context | Pinned source identity and project-derived regional mapping are retained; complete source products are excluded. |
| Clean Energy Regulator (CER) | Small-scale solar installation/capacity context | Pinned source identity and narrow project-authored postcode overrides are retained; complete source products are excluded. |
| Open-Meteo | Historical weather observations for the five project nodes | Request/acquisition logic and attribution are retained; complete responses are excluded. |
| Australian Bureau of Meteorology (BOM) | Southern Oscillation Index context | Acquisition and transformation logic are retained; complete provider datasets are excluded. |
| State of Queensland | Public-holiday and school-calendar context | Narrow project configuration records and registration requirements are retained; complete provider records are not redistributed. |
| Queensland Government Statistician's Office | Historical annual Queensland population context | Bounded provenance and transformation logic are retained; complete provider products are excluded. |

Tracked configuration files such as
`config/sa2_2016_project_region_mapping.csv`,
`config/cer_postcode_overrides.csv`, and
`config/qld_public_holiday_periods_2015_2020.csv` are project inputs or derived
mapping/context records. Their inclusion does not imply ownership of the
underlying provider data.

Exact expected source identities, temporal limits, checksums where applicable,
and acquisition modes are declared in
`data/manifests/source_registry_2020.yml`. See `data/README.md` for the guarded
local acquisition and verification workflow.
