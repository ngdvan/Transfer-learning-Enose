from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA = PACKAGE_ROOT / "datasets" / "NH3_H2_dataset" / "TONG_NH3H2_CHUAN.xlsx"
OUT = PACKAGE_ROOT / "results" / "nh3_h2"
SENSORS = [f"Response{i}" for i in range(1, 21)]
TARGETS = ["NH3", "H2"]
STEPS = 64
N_SENSORS = 20
TOKEN = 32
TARGET_SCALE = np.asarray([500.0, 500.0], np.float32)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_exposures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    df = pd.read_excel(path)
    required = {*SENSORS, *TARGETS, "Combination"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    values, labels, groups, sessions, label_corrections = [], [], [], [], 0
    for group_id, group in df.groupby("Combination", sort=True):
        counts = Counter(map(tuple, group[TARGETS].to_numpy()))
        label = max(counts, key=counts.get)
        label_corrections += len(group) - counts[label]
        values.append(group[SENSORS].to_numpy(np.float32))
        labels.append(label)
        groups.append(int(group_id))
        sessions.append((int(group_id) - 1) // 63 + 1)
    audit = {
        "rows": int(len(df)),
        "exposures": len(values),
        "sessions": int(max(sessions)),
        "exposures_per_session": dict(
            (str(int(k)), int(v))
            for k, v in zip(*np.unique(sessions, return_counts=True))
        ),
        "unique_pairs_per_session": {
            str(s): len({tuple(labels[i]) for i in range(len(labels))
                         if sessions[i] == s})
            for s in range(1, 7)
        },
        "label_transition_rows_corrected": label_corrections,
    }
    return {
        "raw": np.asarray(values, dtype=object),
        "targets": np.asarray(labels, np.float32),
        "groups": np.asarray(groups, np.int32),
        "sessions": np.asarray(sessions, np.int32),
    }, audit


def make_windows(data: dict, exposure_idx: np.ndarray,
                 offsets=range(5)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, g = [], [], []
    for i in exposure_idx:
        raw = data["raw"][i]
        for offset in offsets:
            if offset + STEPS <= len(raw):
                x.append(raw[offset:offset + STEPS])
                y.append(data["targets"][i])
                g.append(data["groups"][i])
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(g, np.int32)


class SensorScaler:
    def fit(self, x: np.ndarray):
        flat = x.reshape(-1, N_SENSORS)
        self.center = np.median(flat, axis=0)
        q25, q75 = np.percentile(flat, [25, 75], axis=0)
        self.scale = np.maximum(q75 - q25, 1e-4)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.center[None, None]) / self.scale[None, None]
        return np.clip(z, -12, 12).astype(np.float32)


class SensorAwareEncoder(nn.Module):
    def __init__(self, use_sensor_ids: bool = True, use_attention: bool = True):
        super().__init__()
        self.use_sensor_ids = use_sensor_ids
        self.use_attention = use_attention
        self.temporal = nn.Sequential(
            nn.Conv1d(2, 24, 5, padding=2), nn.GELU(),
            nn.Conv1d(24, 32, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.project = nn.Sequential(nn.Linear(64, TOKEN), nn.GELU())
        self.sensor_embedding = nn.Parameter(torch.randn(N_SENSORS, TOKEN) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=TOKEN, nhead=4, dim_feedforward=96, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(TOKEN)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None):
        
        if observed is None:
            observed = torch.ones_like(x)
        batch = x.shape[0]
        signal = x.permute(0, 2, 1).reshape(-1, STEPS)
        mask = observed.permute(0, 2, 1).reshape(-1, STEPS)
        h = self.temporal(torch.stack([signal * mask, mask], dim=1))
        token = self.project(torch.cat([h.mean(2), h.amax(2)], dim=1))
        token = token.reshape(batch, N_SENSORS, TOKEN)
        if self.use_sensor_ids:
            token = token + self.sensor_embedding[None]
        if self.use_attention:
            token = self.context(token)
        return self.norm(token)


class MaskedAutoencoder(nn.Module):
    def __init__(self, use_sensor_ids: bool = True, use_attention: bool = True):
        super().__init__()
        self.encoder = SensorAwareEncoder(use_sensor_ids, use_attention)
        self.decoder = nn.Sequential(
            nn.Linear(TOKEN, 96), nn.GELU(), nn.Linear(96, STEPS)
        )

    def forward(self, x: torch.Tensor, observed: torch.Tensor):
        token = self.encoder(x, observed)
        return self.decoder(token).transpose(1, 2)


class Regressor(nn.Module):
    def __init__(self, encoder: SensorAwareEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(N_SENSORS * TOKEN, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.head(self.encoder(x).flatten(1))


def random_mask(x: torch.Tensor, temporal_mask: bool = True,
                sensor_masking: bool = True) -> torch.Tensor:
    
    
    
    
    observed = torch.ones(x.shape, dtype=x.dtype, device="cpu")
    b, _, s = x.shape
    
    for i in range(b):
        if temporal_mask:
            for _ in range(3):
                width = int(torch.randint(6, 17, (1,)))
                start = int(torch.randint(0, STEPS - width + 1, (1,)))
                span_sensors = torch.rand(s) < 0.45
                observed[i, start:start + width, span_sensors] = 0
        
        if sensor_masking:
            count = int(torch.randint(1, 4, (1,)))
            hidden = torch.randperm(s)[:count]
            observed[i, :, hidden] = 0
    return observed.to(x.device, non_blocking=True)


def pretrain(x: np.ndarray, epochs: int, seed: int, *,
             use_sensor_ids: bool = True, use_attention: bool = True,
             temporal_mask: bool = True, sensor_masking: bool = True):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoder(use_sensor_ids, use_attention).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        total, count = 0.0, 0
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            observed = random_mask(xb, temporal_mask, sensor_masking)
            pred = model(xb, observed)
            hidden = 1 - observed
            masked_loss = ((pred - xb).square() * hidden).sum() / hidden.sum().clamp_min(1)
            visible_loss = ((pred - xb).square() * observed).sum() / observed.sum()
            loss = masked_loss + 0.05 * visible_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        history.append(total / count)
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"stage": "pretrain", "epoch": epoch + 1,
                              "loss": history[-1]}), flush=True)
    return deepcopy(model.encoder.state_dict()), history


def farthest_subset(targets: np.ndarray, budget: int, seed: int) -> np.ndarray:
    
    unique, first = np.unique(targets, axis=0, return_index=True)
    budget = min(budget, len(unique))
    rng = np.random.default_rng(seed)
    z = unique / 500.0
    selected = [int(rng.integers(len(unique)))]
    distance = np.linalg.norm(z - z[selected[0]], axis=1)
    while len(selected) < budget:
        candidate = int(np.argmax(distance + rng.uniform(0, 1e-8, len(distance))))
        selected.append(candidate)
        distance = np.minimum(distance, np.linalg.norm(z - z[candidate], axis=1))
    return first[np.asarray(selected)]


def fit_neural_with_history(x: np.ndarray, y: np.ndarray, initial: dict | None,
                            frozen: bool, epochs: int, seed: int,
                            *, val_x: np.ndarray | None = None,
                            val_y: np.ndarray | None = None,
                            use_sensor_ids: bool = True,
                            use_attention: bool = True) -> tuple[Regressor, dict]:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SensorAwareEncoder(use_sensor_ids, use_attention)
    if initial is not None:
        encoder.load_state_dict(initial)
    model = Regressor(encoder).to(device)
    if frozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = model.head.parameters()
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=2e-4)
    elif initial is not None:
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": 8e-5},
            {"params": model.head.parameters(), "lr": 8e-4},
        ], weight_decay=2e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y / TARGET_SCALE)),
        batch_size=min(64, len(x)), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = {"train_loss": [], "validation_loss": [], "best_epoch": None,
               "best_validation_loss": None}
    best_state, best_val = None, float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_total, epoch_count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = nn.functional.huber_loss(pred, yb, delta=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            epoch_total += float(loss.detach()) * len(xb)
            epoch_count += len(xb)
        history["train_loss"].append(epoch_total / max(epoch_count, 1))
        if val_x is not None and val_y is not None and len(val_x):
            model.eval()
            with torch.no_grad():
                vx = torch.from_numpy(val_x).to(device)
                vy = torch.from_numpy(val_y / TARGET_SCALE).to(device)
                value = float(nn.functional.huber_loss(model(vx), vy, delta=0.1).cpu())
            history["validation_loss"].append(value)
            if value < best_val:
                best_val = value
                history["best_epoch"] = epoch + 1
                history["best_validation_loss"] = value
                best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def fit_neural(x: np.ndarray, y: np.ndarray, initial: dict | None,
               frozen: bool, epochs: int, seed: int) -> Regressor:
    model, _ = fit_neural_with_history(
        x, y, initial, frozen, epochs, seed
    )
    return model


def predict(model: Regressor, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            out.append(model(torch.from_numpy(x[start:start + 128]).to(device)).cpu().numpy())
    return np.clip(np.concatenate(out) * TARGET_SCALE, 0, TARGET_SCALE)


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict:
    result = {}
    for j, gas in enumerate(TARGETS):
        mae = mean_absolute_error(y[:, j], pred[:, j])
        result[gas] = {
            "mae_ppm": float(mae),
            "rmse_ppm": float(np.sqrt(mean_squared_error(y[:, j], pred[:, j]))),
            "r2": float(r2_score(y[:, j], pred[:, j])),
            "mae_percent_test_mean": float(100 * mae / y[:, j].mean()),
            "mae_percent_full_scale": float(100 * mae / 500),
        }
    return result


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for budget in sorted({r["budget"] for r in rows}):
        output[str(budget)] = {}
        for model in sorted({r["model"] for r in rows if r["budget"] == budget}):
            selected = [r for r in rows if r["budget"] == budget and r["model"] == model]
            output[str(budget)][model] = {}
            for gas in TARGETS:
                output[str(budget)][model][gas] = {}
                for metric in selected[0]["metrics"][gas]:
                    vals = np.asarray([r["metrics"][gas][metric] for r in selected])
                    output[str(budget)][model][gas][metric] = {
                        "mean": float(vals.mean()),
                        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data, audit = load_exposures(args.data)
    pre_idx = np.flatnonzero(data["sessions"] <= 4)
    adapt_idx = np.flatnonzero(data["sessions"] == 5)
    test_idx = np.flatnonzero(data["sessions"] == 6)
    pre_x_raw, _, _ = make_windows(data, pre_idx)
    scaler = SensorScaler().fit(pre_x_raw)
    pre_x = scaler.transform(pre_x_raw)
    test_x_raw, test_y, test_groups = make_windows(data, test_idx, offsets=[0])
    test_x = scaler.transform(test_x_raw)

    pretrained, pretrain_history = pretrain(
        pre_x, args.pretrain_epochs, args.seed
    )
    torch.save(pretrained, args.output / "pretrained_encoder.pt")
    np.savez(args.output / "scaler.npz", center=scaler.center, scale=scaler.scale)

    rows, prediction_rows = [], []
    budgets = [5, 10, 20, 40, 57]
    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat * 100
        local = farthest_subset(data["targets"][adapt_idx], 57, repeat_seed)
        ordered_adapt = adapt_idx[local]
        for budget in budgets:
            chosen = ordered_adapt[:budget]
            train_raw, train_y, _ = make_windows(data, chosen)
            train_x = scaler.transform(train_raw)
            models = {
                "scratch": fit_neural(
                    train_x, train_y, None, False, args.finetune_epochs,
                    repeat_seed + budget,
                ),
                "frozen_transfer": fit_neural(
                    train_x, train_y, pretrained, True, args.finetune_epochs,
                    repeat_seed + budget + 1,
                ),
                "finetuned_transfer": fit_neural(
                    train_x, train_y, pretrained, False, args.finetune_epochs,
                    repeat_seed + budget + 2,
                ),
            }
            predictions = {name: predict(model, test_x)
                           for name, model in models.items()}
            
            for name, pred in predictions.items():
                row = {
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "budget": budget,
                    "model": name,
                    "metrics": metric_dict(test_y, pred),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                prediction_rows.extend({
                    "repeat": repeat + 1, "budget": budget, "model": name,
                    "combination": int(group),
                    "NH3_true": float(y[0]), "H2_true": float(y[1]),
                    "NH3_pred": float(p[0]), "H2_pred": float(p[1]),
                } for group, y, p in zip(test_groups, test_y, pred))

    result = {
        "experiment": "same-device sensor-aware self-supervised session transfer",
        "protocol": {
            "unlabeled_pretraining_sessions": [1, 2, 3, 4],
            "labeled_adaptation_session": 5,
            "untouched_test_session": 6,
            "label_budgets_unique_concentration_pairs": budgets,
            "subset_selection": "seeded farthest-point sampling in normalized NH3/H2 space",
            "repeats": args.repeats,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "window": "first 64 samples; offsets 0..4 train, offset 0 test",
            "encoder": "shared temporal CNN + learned sensor IDs + 2-layer cross-sensor Transformer",
            "ssl_objective": "masked time spans and complete sensors reconstruction",
            "test_concentration_mean_ppm": dict(zip(TARGETS, map(float, test_y.mean(0)))),
            "test_concentration_range_ppm": [0, 500],
        },
        "data_audit": audit,
        "pretraining_loss": pretrain_history,
        "runs": rows,
        "summary": aggregate(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "predictions.csv", index=False
    )
    print(json.dumps({"summary": result["summary"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
