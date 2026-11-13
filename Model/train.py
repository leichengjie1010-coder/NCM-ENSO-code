from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import config


def arguments():
    parser = argparse.ArgumentParser(description="Fit or evaluate the final NCM.")
    parser.add_argument("--data", type=Path, default=config.DATA_FILE)
    parser.add_argument("--forecast-data", type=Path, default=config.FORECAST_FILE)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--reuse-linear", type=Path)
    parser.add_argument("--forecast-origin", default=config.FORECAST_ORIGIN)
    parser.add_argument(
        "--mode",
        choices=("predict", "fit-final", "replay-final", "verify-recipe"),
        default="predict",
    )
    args = parser.parse_args()
    if args.mode in ("replay-final", "verify-recipe"):
        return args
    bank = args.reuse_linear / "network_inputs_366.npz" if args.reuse_linear else args.data
    if not bank.is_file():
        parser.error("Prepared inputs were not found; pass --data or --reuse-linear.")
    if bank.suffix.lower() not in (".npz", ".nc"):
        parser.error("--data must be a prepared feature NPZ or monthly-field NetCDF.")
    if args.mode == "fit-final" and bank.suffix.lower() != ".npz":
        parser.error("fit-final requires the original matching feature NPZ.")
    if args.forecast_origin != "2026-09":
        parser.error("The supplied forecast comparison uses September 2026 initialization.")
    if args.forecast_data is not None and not args.forecast_data.is_file():
        parser.error("--forecast-data must name an existing prepared-input file.")
    return args


def forecast_values(path, net, linear_dir, origin, output):
    import numpy as np
    from model import neural
    from model.evaluation import dates_of, output_offsets, read_bank

    if path.suffix.lower() == ".npz":
        event = read_bank(path)
    elif path.suffix.lower() == ".nc":
        from model import linear

        destination = output / "forecast_inputs"
        linear.project_inputs(path, linear_dir, destination)
        event = read_bank(destination / "network_inputs_366.npz")
    else:
        raise ValueError("Forecast inputs must be prepared NPZ or NetCDF.")
    input_month = str(np.datetime64(origin, "M") - np.timedelta64(1, "M"))
    dates = dates_of(event)
    selected = np.flatnonzero(dates[event["all_origins"]] == input_month)
    if len(selected) != 1:
        raise ValueError(f"Forecast inputs do not contain {input_month}.")
    index = int(selected[0])
    prediction = neural.predict(net, event["all_x"][[index]])[0]
    offsets = output_offsets(event)
    result = {}
    for lead in range(1, 7):
        target = str(np.datetime64(input_month, "M") + np.timedelta64(lead, "M"))
        month = int(target[5:7])
        for name, values in (
            ("NCM", prediction),
            ("MMarkov-4", event["all_markov"][index]),
            ("MMarkov-3", event["all_markov3"][index]),
        ):
            result[name, target] = float(values[lead - 1] + offsets[month - 1])
    return result


def main():
    args = arguments()
    if args.mode in ("replay-final", "verify-recipe"):
        from model import final_recipe

        if args.mode == "verify-recipe":
            result = final_recipe.verify_recipe(config.RECIPE_DIR)
        else:
            result = final_recipe.replay(args.output, config.RECIPE_DIR)
        print(json.dumps(result, indent=2))
        return

    import numpy as np
    import torch
    from model import neural
    from model.evaluation import export_results, read_bank

    torch.set_num_threads(1)
    linear_dir = args.reuse_linear or config.LINEAR_DIR
    bank_path = args.reuse_linear / "network_inputs_366.npz" if args.reuse_linear else args.data
    if bank_path.suffix.lower() == ".nc":
        from model import linear

        linear_dir = args.output / "linear"
        linear.fit_pipeline(bank_path, linear_dir)
        bank_path = linear_dir / "network_inputs_366.npz"
    bank = read_bank(bank_path)
    training_run = None
    if args.mode == "fit-final":
        from model import final_recipe

        net, training_run = final_recipe.fit(
            bank, bank_path, args.output, config.RECIPE_DIR
        )
    else:
        net = neural.load_model()
        args.output.mkdir(parents=True, exist_ok=True)
        neural.save_checkpoint(args.output / "model.npz", net)

    future = None
    if args.forecast_data is not None:
        future = forecast_values(
            args.forecast_data, net, linear_dir, args.forecast_origin, args.output
        )
    metrics = export_results(bank, net, args.output, future=future)
    run = {
        "mode": args.mode,
        "selected_final_model_sha256": config.FINAL_MODEL_SHA256,
        "output_checkpoint_sha256": hashlib.sha256(
            (args.output / "model.npz").read_bytes()
        ).hexdigest(),
        "feature_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "training_recipe": training_run,
        "evaluation": "2013-2024 development period",
        "independent_test_claim": False,
        "forecast": (
            "computed from prepared forecast inputs" if future else "archived reference values"
        ),
        "named_initialization_month": args.forecast_origin,
        "last_complete_input_month": str(
            np.datetime64(args.forecast_origin, "M") - np.timedelta64(1, "M")
        ),
        "metrics": metrics,
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
