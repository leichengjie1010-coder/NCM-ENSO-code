# NCM code and figures

Model code, parameters, input data and plotting scripts for the NCM study.

| Directory or file | Contents |
|---|---|
| `Data/` | Historical predictors, 2026 forecast inputs and data descriptions |
| `Model/` | NCM and Markov code, fitted parameters and evaluation tables |
| `Figures/` | Separate scripts for Figures 2-6 and their input CSV files |
| `requirements.txt` | Python dependencies and versions |

## Model

Paths below are relative to `Model/`.

| File or directory | Purpose |
|---|---|
| `train.py` | Entry point for prediction, final-stage fitting and parameter reconstruction |
| `config.py` | Input and output paths and forecast settings |
| `model/linear.py` | EOF analysis, Markov models and predictor construction |
| `model/neural.py` | NCM architecture, parameter loading, prediction and fitting |
| `model/evaluation.py` | ACC, MAE, RMSE, bias and event rankings |
| `model/final_recipe.py` | Final-stage fitting, parameter reconstruction and verification |
| `weights/` | Final NCM parameters and fitted EOF and Markov states |
| `training_recipe/` | Fitting configuration, starting parameters and saved updates |
| `reference_tables/` | Comparison forecasts, benchmark metrics and reference climatology |

## Figures

Paths below are relative to `Figures/`.

| Script | Figure |
|---|---|
| `Fig2_ACC_DJF.py` | ACC curves and DJF forecast-observation scatter |
| `Fig3_Seasonal_ACC.py` | Seasonal ACC maps and selected initialization-month curves |
| `Fig4_February_events.py` | February-July forecasts and MAE for the 2015 and 2023 events |
| `Fig5_September_events.py` | September-February forecasts and MAE for the 2015/16 and 2023/24 events |
| `Fig6_2026_2027.py` | NCM and Markov forecasts for September 2026-February 2027 |

`NCM_plot_utils.py` provides shared plotting styles, CSV reading and image export.
Each figure script reads the adjacent CSV files and exports PNG, PDF and SVG.
