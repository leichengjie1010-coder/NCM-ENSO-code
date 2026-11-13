"""Conditional neural correction for 20-month ENSO trajectories."""
from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_DIM = 366
HORIZON = 20
DESCRIPTOR_DIM = 29
HIDDEN_UNITS = 4
TRAINABLE_PARAMETERS = 17843
MODEL_SCHEMA = {
    "model": "NCM",
    "input_dimensions": INPUT_DIM,
    "active_input_dimensions": 324,
    "summary_dimensions": DESCRIPTOR_DIM,
    "hidden_units": HIDDEN_UNITS,
    "forecast_months": HORIZON,
    "trainable_parameters": TRAINABLE_PARAMETERS,
}
DEFAULT_CONFIG = {
    **MODEL_SCHEMA,
    "epochs": 1,
    "lr": 1e-7,
    "seed": 20261013,
    "torch_threads": 1,
    "late_weight": 0.1,
    "acc_weight": 0.5,
    "monthly_mae_weight": 0.2,
    "three_month_mae_weight": 0.8,
    "gradient_clip_norm": 5.0,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "log_every": 50,
    "training_cutoff_exclusive": 660,
    "validation_start": 660,
    "validation_end_exclusive": 804,
}


class NCMNet(nn.Module):
    """Shared four-unit correction network with an internal inactive-input mask."""

    input_dim = INPUT_DIM
    horizon = HORIZON
    hidden = HIDDEN_UNITS

    def __init__(self):
        super().__init__()
        for name, shape, ones in [
            ("mean366", (366,), False),
            ("scale366", (366,), True),
            ("amp_mean29", (29,), False),
            ("amp_scale29", (29,), True),
        ]:
            self.register_buffer(
                name,
                (
                    torch.ones(shape, dtype=torch.float64)
                    if ones
                    else torch.zeros(shape, dtype=torch.float64)
                ),
            )
        for name, shape in {
            "L0": (366, 20),
            "L1": (366, 20),
            "B0": (18, 20),
            "B1": (18, 20),
            "c0": (20,),
            "c1": (20,),
            "v": (366,),
            "gate_w": (366,),
            "gate_b": (),
        }.items():
            self.register_parameter(
                name, nn.Parameter(torch.zeros(shape, dtype=torch.float64))
            )
        self.phi_hidden = nn.Linear(29, 4, dtype=torch.float64)
        self.raw_hidden = nn.Linear(366, 4, bias=False, dtype=torch.float64)
        self.amplitude = nn.Linear(4, 1, dtype=torch.float64)
        self.trajectory = nn.Linear(4, 20, dtype=torch.float64)
        omega = torch.zeros(20, dtype=torch.float64)
        omega[2:6] = torch.tensor([0.1, 0.5, 0.5, 1.25], dtype=torch.float64)
        self.gain_omega = nn.Parameter(omega)
        self.slope_coef = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.register_buffer(
            "amplitude_bounds", torch.tensor([0.7, 1.4], dtype=torch.float64)
        )
        self.register_buffer(
            "omega_bounds", torch.tensor([0.0, 1.25], dtype=torch.float64)
        )
        mask = torch.ones(366, dtype=torch.float64)
        mask[306:348] = 0.0
        self.register_buffer("feature_mask366", mask)

    def forward_with_diagnostics(self, raw):
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=self.L0.dtype, device=self.L0.device)
        else:
            raw = raw.to(dtype=self.L0.dtype, device=self.L0.device)
        if raw.ndim != 2 or raw.shape[1] != 366 or not torch.isfinite(raw).all():
            raise ValueError("NCM requires finite N x366 predictors")
        effective = torch.where(
            self.feature_mask366[None, :] > 0, raw, self.mean366[None, :]
        )
        z = (effective - self.mean366) / self.scale366
        q = z[:, 348:366]
        r = z @ self.v
        qr = q * r[:, None]
        gate = torch.clamp(z @ self.gate_w + self.gate_b, 0.0, 1.0)
        u = self.c0 + z @ self.L0 + qr @ self.B0
        u = u + gate[:, None] * (self.c1 + z @ self.L1 + qr @ self.B1)
        sd6 = torch.std(u[:, :6], dim=1, correction=0)
        sd20 = torch.std(u, dim=1, correction=0)
        ratio = sd6 / torch.where(sd20 > 1e-8, sd20, torch.ones_like(sd20))
        n0 = effective[:, 263]
        dn1 = n0 - effective[:, 267]
        phi = torch.cat(
            [
                sd6[:, None],
                sd20[:, None],
                ratio[:, None],
                n0[:, None],
                dn1[:, None],
                effective[:, 348:366],
                u[:, :6],
            ],
            dim=1,
        )
        hidden = torch.tanh(
            self.phi_hidden((phi - self.amp_mean29) / self.amp_scale29)
            + self.raw_hidden(z)
        )
        raw_amp = self.amplitude(hidden).squeeze(1)
        amplitude = torch.clamp(
            raw_amp, self.amplitude_bounds[0], self.amplitude_bounds[1]
        )
        delta = 0.2 * torch.tanh(self.trajectory(hidden))
        omega = torch.clamp(self.gain_omega, self.omega_bounds[0], self.omega_bounds[1])
        prediction = u * (1.0 + omega[None, :] * (amplitude[:, None] - 1.0))
        prediction = prediction + delta
        tail = prediction[:, 5] + self.slope_coef * (amplitude - 1.0) * (
            u[:, 4] - u[:, 5]
        )
        prediction = torch.cat(
            [prediction[:, :5], tail[:, None], prediction[:, 6:]], dim=1
        )
        reverse = (u[:, 5] - u[:, 4]) * (prediction[:, 5] - prediction[:, 4]) < 0.0
        pair = prediction[:, 4:6]
        pair = torch.where(
            reverse[:, None], pair.mean(1, keepdim=True).expand(-1, 2), pair
        )
        prediction = torch.cat([prediction[:, :4], pair, prediction[:, 6:]], dim=1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite NCM H20 prediction")
        return prediction, {
            "standardized_raw": z,
            "decoder": u,
            "product_projection": r,
            "gate": gate,
            "amplitude_features29": phi,
            "hidden": hidden,
            "raw_amplitude": raw_amp,
            "amplitude": amplitude,
            "trajectory_delta20": delta,
            "pair_reverse": reverse,
        }

    def forward(self, raw):
        return self.forward_with_diagnostics(raw)[0]


def validate_network(net):
    if sum(p.numel() for p in net.parameters()) != TRAINABLE_PARAMETERS:
        raise AssertionError("The model must contain 17843 trainable parameters")
    for name, value in net.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite state{name}")
    if torch.any(net.scale366 <= 0) or torch.any(net.amp_scale29 <= 0):
        raise ValueError("Normalization scales must be positive")
    mask = torch.ones(366, dtype=torch.float64)
    mask[306:348] = 0
    if not torch.equal(net.feature_mask366.cpu(), mask):
        raise ValueError("The inactive-input mask at coordinates 306:348 changed")
    if not torch.equal(
        net.amplitude_bounds.cpu(), torch.tensor([0.7, 1.4], dtype=torch.float64)
    ):
        raise ValueError("Amplitude safety bounds changed")
    if not torch.equal(
        net.omega_bounds.cpu(), torch.tensor([0.0, 1.25], dtype=torch.float64)
    ):
        raise ValueError("Omega safety bounds changed")


def _load_archive(source, *, require_schema=False):
    with np.load(source, allow_pickle=False) as archive:
        cfg = json.loads(str(archive["configuration"])) if "configuration" in archive else {}
        state = {key: np.asarray(archive[key]) for key in archive.files if key != "configuration"}
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint configuration must be an object")
    if require_schema and set(MODEL_SCHEMA) - set(cfg):
        raise ValueError("Checkpoint is missing the NCM model schema")
    for key, expected in MODEL_SCHEMA.items():
        if key in cfg and cfg[key] != expected:
            raise ValueError(f"Checkpoint {key} must be {expected}")
    net = NCMNet()
    net.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()}, strict=True)
    validate_network(net)
    net._checkpoint_configuration = cfg
    net.eval()
    return net


def load_model():
    path = Path(__file__).resolve().parents[1] / "weights" / "final_model.npz"
    expected = "05d21a64041969aa95632bfa8746a4d30603ee35786c76bd726ab83c598824d3"
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Final checkpoint SHA-256 mismatch")
    return _load_archive(path, require_schema=True)


def load_checkpoint(path):
    """Load a numeric NCM checkpoint, including the bundled R13 warm start."""
    return _load_archive(Path(path))


def save_checkpoint(path, net, configuration=None):
    """Save numeric state and retain operation/provenance metadata."""
    validate_network(net)
    cfg = {**getattr(net, "_checkpoint_configuration", {}), **(configuration or {})}
    for key, expected in MODEL_SCHEMA.items():
        if cfg.get(key, expected) != expected:
            raise ValueError(f"Checkpoint {key} must be {expected}")
    cfg.update(MODEL_SCHEMA)
    np.savez_compressed(
        Path(path), configuration=json.dumps(cfg, ensure_ascii=False),
        **{key: value.detach().cpu().numpy() for key, value in net.state_dict().items()},
    )
    net._checkpoint_configuration = cfg


def checked_inputs(data, cfg):
    """Use only pre-cutoff labels; no validation truth enters optimizer tensors."""
    required = (
        "train_x",
        "train_origins",
        "train_y",
        "train_y3",
        "mask",
        "mask3",
        "val_x",
        "val_origins",
        "obs",
        "dates",
    )
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing network input fields: {sorted(missing)}")
    d = dict(data)
    for name in ("train_origins", "val_origins"):
        raw = np.asarray(d[name])
        if (
            raw.ndim != 1
            or raw.dtype.kind not in "iu"
            or len(np.unique(raw)) != len(raw)
        ):
            raise ValueError(f"{name} must be unique integer indices")
        d[name] = raw.astype(np.int64)
    n = len(d["train_origins"])
    nv = len(d["val_origins"])
    for name, shape in [
        ("train_x", (n, 366)),
        ("train_y", (n, 20)),
        ("train_y3", (n, 20)),
        ("mask", (n, 20)),
        ("mask3", (n, 20)),
        ("val_x", (nv, 366)),
    ]:
        a = np.asarray(d[name], dtype=np.float64)
        if a.shape != shape:
            raise ValueError(f"{name}: expected{shape}, got{a.shape}")
        d[name] = a
    for name in ("train_x", "val_x"):
        if not np.isfinite(d[name]).all():
            raise ValueError(
                f"{name} is nonfinite; "
                "inactive coordinates must contain finite placeholders"
            )
    for name in ("mask", "mask3"):
        if not np.isin(d[name], [0.0, 1.0]).all():
            raise ValueError(f"{name} is not a binary validity mask")
    cutoff = int(cfg["training_cutoff_exclusive"])
    targets = d["train_origins"][:, None] + np.arange(1, 21)[None, :]
    # Centered labels require both neighboring months before the cutoff.
    if np.any((d["mask"] > 0) & ((targets >= cutoff) | (targets < 0))):
        raise ValueError("Monthly training label would cross2012Dec cutoff")
    if np.any((d["mask3"] > 0) & ((targets + 1 >= cutoff) | (targets - 1 < 0))):
        raise ValueError("Centered training label would use a post-cutoff neighbor")
    if np.any(d["train_origins"] >= cutoff):
        raise ValueError("Validation origins cannot be truth-training donors")
    for yname, mname in [("train_y", "mask"), ("train_y3", "mask3")]:
        if not np.isfinite(d[yname][d[mname] > 0]).all():
            raise ValueError(f"Valid labels in{yname} are nonfinite")
        d[yname] = np.where(d[mname] > 0, d[yname], 0.0)
    keep = d["mask"].any(axis=1)
    for name in ("train_x", "train_y", "train_y3", "mask", "mask3", "train_origins"):
        d[name] = d[name][keep]
    if (
        len(d["train_x"]) == 0
        or np.any(d["mask"].sum(0) <= 0)
        or np.any(d["mask3"].sum(0) < 4)
    ):
        raise ValueError("Insufficient historical labels for all H20 loss terms")
    d["obs"] = np.asarray(d["obs"], dtype=np.float64)
    # Use calendar-month keys for event and lead alignment.
    d["dates"] = np.asarray(d["dates"]).astype(str).astype("U7")
    if d["obs"].ndim != 1 or d["dates"].shape != d["obs"].shape:
        raise ValueError("obs/dates must be equal-length one-dimensional arrays")
    month_no = np.array(
        [int(v[:4]) * 12 + int(v[5:7]) - 1 for v in d["dates"]], dtype=int
    )
    if not np.all(np.diff(month_no) == 1) or d["dates"][0][:7] != "1958-01":
        raise ValueError("The NCM input contract uses continuous dates,index0=1958-01")
    if d["val_origins"].min() < int(cfg["validation_start"]):
        raise ValueError("Validation origins overlap training-period origins")
    # Check the masked historical labels against the verification series.
    train_targets = d["train_origins"][:, None] + np.arange(1, 21)[None, :]
    for label_name, mask_name, centered in [
        ("train_y", "mask", False),
        ("train_y3", "mask3", True),
    ]:
        valid = d[mask_name] > 0
        indices = train_targets[valid]
        expected = (
            (d["obs"][indices - 1] + d["obs"][indices] + d["obs"][indices + 1]) / 3.0
            if centered
            else d["obs"][indices]
        )
        if not np.isfinite(expected).all() or not np.allclose(
            d[label_name][valid], expected, rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                f"{label_name} does not match aligned pre-cutoff observations"
            )
    return d


def _stats(a, b):
    error = a - b
    acc = None
    if len(a) >= 4 and np.std(a) >= 1e-12 and np.std(b) >= 1e-12:
        aa = a - a.mean()
        bb = b - b.mean()
        acc = float(
            np.clip(
                np.dot(aa, bb) / np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)), -1.0, 1.0
            )
        )
    return {
        "ACC": acc,
        "MAE": float(np.abs(error).mean()) if len(error) else None,
        "RMSE": float(np.sqrt(np.square(error).mean())) if len(error) else None,
        "bias": float(error.mean()) if len(error) else None,
        "n_pairs": len(error),
    }


def score_predictions(prediction, origins, obs, cfg):
    """Score monthly values and three-month means with fixed endpoint handling."""
    start = int(cfg["validation_start"])
    end = int(cfg["validation_end_exclusive"])
    if not (0 <= start < end <= len(obs)):
        raise ValueError("Validation target period is outside observations")
    targets = origins[:, None] + np.arange(1, 21)[None, :]
    period = (targets >= start) & (targets < end)
    if prediction.shape != (len(origins), 20) or not np.isfinite(prediction).all():
        raise ValueError("Expected finite validation N x20 predictions")
    rows = []
    for metric in ("monthly", "three_month"):
        p = prediction.copy()
        truth = np.full_like(p, np.nan)
        valid = period.copy()
        if metric == "monthly":
            truth[valid] = obs[targets[valid]]
        else:
            p[:, 1:-1] = (
                prediction[:, :-2] + prediction[:, 1:-1] + prediction[:, 2:]
            ) / 3.0
            t = obs[start:end].copy()
            t[1:-1] = (
                obs[start : end - 2] + obs[start + 1 : end - 1] + obs[start + 2 : end]
            ) / 3.0
            truth[valid] = t[targets[valid] - start]
        valid &= np.isfinite(p) & np.isfinite(truth)
        for h in range(20):
            rows.append(
                {
                    "metric": metric,
                    "lead": h + 1,
                    **_stats(p[valid[:, h], h], truth[valid[:, h], h]),
                }
            )
    return rows


def predict(model, predictors):
    """Return monthly forecasts; standardization is internal to the network."""
    if not isinstance(model, NCMNet):
        raise TypeError("model must be an NCMNet instance")
    model.eval()
    with torch.no_grad():
        return model(predictors).cpu().numpy()


def continue_fit(data, model=None, config=None):
    """Continue supervised training from the supplied or final model.

    Normalization and the input basis remain those of the supplied model. The
    objective combines monthly MAE, three-month MAE, and correlation loss.
    Continued training updates the weights; it does not regenerate the
    released checkpoint from a random initialization.
    """
    supplied = config or {}
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **supplied}
    data = checked_inputs(data, cfg)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(cfg["torch_threads"]))
    torch.manual_seed(int(cfg["seed"]))
    if model is not None and not isinstance(model, NCMNet):
        raise TypeError("model must be an NCMNet instance or None")
    net = load_model() if model is None else copy.deepcopy(model)
    net.requires_grad_(True)
    x = torch.as_tensor(data["train_x"], dtype=torch.float64)
    y = torch.as_tensor(data["train_y"], dtype=torch.float64)
    y3 = torch.as_tensor(data["train_y3"], dtype=torch.float64)
    mask = torch.as_tensor(data["mask"], dtype=torch.float64)
    mask3 = torch.as_tensor(data["mask3"], dtype=torch.float64)
    lead_weights = torch.ones(HORIZON)
    lead_weights[8:] = cfg["late_weight"]
    wm = mask / mask.sum(0) * lead_weights[None, :] / lead_weights.sum()
    wc = mask3 / mask3.sum(0) * lead_weights[None, :] / lead_weights.sum()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg["lr"], betas=tuple(cfg["adam_betas"]),
        eps=cfg["adam_eps"],
    )
    trace = []
    net.train()
    for step in range(int(cfg["epochs"])):
        optimizer.zero_grad()
        prediction = net(x)
        smooth = torch.cat(
            [prediction[:, :1],
             (prediction[:, :-2] + prediction[:, 1:-1] + prediction[:, 2:]) / 3,
             prediction[:, -1:]], dim=1,
        )
        monthly_mae = ((prediction - y).abs() * wm).sum()
        three_month_mae = ((smooth - y3).abs() * wc).sum()
        correlations = []
        for lead in range(6):
            use = mask3[:, lead] > 0
            a, b = smooth[use, lead], y3[use, lead]
            a, b = a - a.mean(), b - b.mean()
            correlations.append(
                (a * b).sum() / torch.sqrt(
                    a.square().sum() * b.square().sum() + 1e-20)
            )
        correlations = torch.stack(correlations)
        correlation_loss = 1 - 0.5 * correlations.mean() - 0.5 * correlations[-1]
        loss = (
            cfg["monthly_mae_weight"] * monthly_mae
            + cfg["three_month_mae_weight"] * three_month_mae
            + cfg["acc_weight"] * correlation_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite joint loss at step {step+1}")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in net.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at step {step+1}")
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip_norm"])
        optimizer.step()
        if step % int(cfg["log_every"]) == 0 or step+1 == int(cfg["epochs"]):
            trace.append({
                "step": step+1, "loss": float(loss.detach()),
                "monthly_mae": float(monthly_mae.detach()),
                "three_month_mae": float(three_month_mae.detach()),
                "correlation_loss": float(correlation_loss.detach()),
            })
    net.eval()
    validate_network(net)
    net._checkpoint_configuration = {
        **getattr(net, "_checkpoint_configuration", {}),
        "training_mode": "additional_historical_continuation",
        "current_role": "New continued-fit candidate",
        "formal_primary_model": False,
        "explicit_user_adoption": False,
        "exact_final_reconstruction": False,
        "independent_test_claim": False,
        "continuation_configuration": cfg,
    }
    return net, trace


def fit(data, model=None, config=None):
    """Compatibility alias for additional continuation; use final_recipe for the final lineage."""
    import warnings
    warnings.warn("neural.fit is additional continuation, not the selected final recipe; "
                  "use final_recipe.fit or final_recipe.replay", FutureWarning, stacklevel=2)
    return continue_fit(data, model=model, config=config)
