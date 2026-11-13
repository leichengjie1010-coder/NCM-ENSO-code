# Data

| File | Contents |
|---|---|
| `network_inputs_366.npz` | Historical features, targets, observations and Markov forecasts |
| `forecast2026_inputs.npz` | Prepared July and August 2026 origins |

## Sources and preparation

The physical inputs are ECMWF ORAS5 monthly fields: `sosstsst` (SST, degC),
`sozotaux` and `sometauy` (surface stress, N m-2), `sossheig` (SSH, m), `sohtc300`
(upper-300-m heat content, J m-2), and `so20chgt` (20-degC isotherm depth, m).
ORAS5 is distributed through the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5).
The analysis grid is 2.5 degrees, 20S-20N and 120E-280E. Historical assembly uses
the frozen 1958-2012 field support, consolidated fields for 2013-2014, and
operational fields for 2015-2024. The forecast inputs use operational fields from
July 2025 through August 2026.

Preparation removes the 1958-2012 calendar-month climatology, fits separate
unweighted covariance EOFs retaining 99% variance, and fits seasonal Markov
models using three joint MEOF modes. MMarkov-4 uses SST, stress, SSH and Z20;
MMarkov-3 omits Z20 and is fitted separately. Niño3.4 is the cosine-latitude
weighted SST mean over 5S-5N, 170W-120W. The fitted EOF, Markov and climatology
files are in `Model/weights/linear/`.

## Historical archive

`dates` contains 804 monthly dates from January 1958 through December 2024.
All origin arrays are zero-based indices into `dates`; lead 1 targets the next
month. The named initialization month is the month after the last complete input
month. For example, an origin in January 2015 is a February 2015 initialization.

| Member | Shape | Meaning |
|---|---|---|
| `train_x`, `train_origins` | `(647, 366)`, `(647,)` | January 1959-November 2012 historical origins |
| `train_y`, `train_y3` | `(647, 20)` | Monthly and centered-three-month targets |
| `mask`, `mask3` | `(647, 20)` | Valid training targets; these masks are authoritative |
| `val_x`, `val_origins` | `(143, 366)`, `(143,)` | January 2013-November 2024 development origins |
| `all_x`, `all_origins` | `(792, 366)`, `(792,)` | All available origins from January 1959-December 2024 |
| `obs`, `dates` | `(804,)` | Monthly native-reference Niño3.4 anomalies and dates |
| `train_markov`, `val_markov` | `(647, 20)`, `(143, 20)` | MMarkov-4 forecasts for those subsets |
| `all_markov`, `all_markov3` | `(792, 20)` | MMarkov-4 and MMarkov-3 forecasts for all origins |
| `active_input_mask` | `(366,)` | 324 active columns; columns 306:348 are inactive |
| `all_deleted42_valid` | `(792, 42)` | Original validity of the inactive columns |
| `feature_names` | `(366,)` | Exact feature order |
| `nino_climatology_1958_2012` | `(12,)` | Native calendar-month climatology |
| `metadata_json` | scalar string | Archived fitting and input metadata |

The `origins` member aliases `train_origins`.

## Feature order

The zero-based, half-open column ranges are:

| Columns | Contents |
|---|---|
| `0:20` | MMarkov-4 forecasts at leads 1-20 |
| `20:260` | First ten PCs of four blocks at lags 0, 1, 2, 3, 6 and 12 |
| `260:280` | Three MEOF states and Niño3.4 at lags 0, 1, 3, 6 and 12 |
| `280:286` | Seasonal sine/cosine and four current Niño3.4 transformations |
| `286:294` | Niño3.4 changes and lagged values at lags 1, 3, 6 and 12 |
| `294:306` | Twelve calendar-month indicators |
| `306:348` | Inactive historical proxy columns, masked in every network route |
| `348:366` | Eighteen thermal-state and interaction features |

The active inputs require 13 complete months through the origin.
Exact feature names and transformations are retained in the archive and
`Model/model/linear.py`.

## 2026 forecast inputs

`forecast2026_inputs.npz` has 14 monthly dates from July 2025-August 2026, two
366-column feature rows, their origin indices and 20-lead Markov forecasts.
The August origin provides the September 2026 initialization. It contains no
observations for the forecast target months.
