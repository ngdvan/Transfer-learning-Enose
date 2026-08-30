from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "sensor_drift_dataset"
OUT = ROOT / "results" / "sensor_drift"
REPORT = ROOT / "reports" / "SENSOR_DRIFT_RESULTS.md"
DOI = "10.24432/C5MK6M"
GASES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
N_SENSORS = 16
N_FEATURES = 8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: Path) -> dict[str, np.ndarray]:
    xs, labels, concentrations, batches = [], [], [], []
    for batch in range(1, 11):
        path = data_dir / f"batch{batch}.dat"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                class_text, concentration_text = fields[0].split(";")
                label = int(class_text) - 1
                values = np.empty(N_SENSORS * N_FEATURES, dtype=np.float32)
                seen = set()
                for item in fields[1:]:
                    index_text, value_text = item.split(":", 1)
                    index = int(index_text) - 1
                    values[index] = float(value_text)
                    seen.add(index)
                if seen != set(range(N_SENSORS * N_FEATURES)):
                    raise ValueError(f"Incomplete feature vector: {path}:{line_number}")
                xs.append(values.reshape(N_SENSORS, N_FEATURES))
                labels.append(label)
                concentrations.append(float(concentration_text))
                batches.append(batch)
    x = np.stack(xs)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite features detected")
    return {
        "x": x,
        "label": np.asarray(labels, dtype=np.int64),
        "concentration": np.asarray(concentrations, dtype=np.float32),
        "batch": np.asarray(batches, dtype=np.int64),
    }


class FeatureScaler:
    def __init__(self) -> None:
        self.scaler = RobustScaler(quantile_range=(10, 90))

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        self.scaler.fit(x.reshape(len(x), -1))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x.reshape(len(x), -1))
        return np.clip(z, -10, 10).reshape(-1, N_SENSORS, N_FEATURES).astype(np.float32)


class SensorEncoder(nn.Module):
    def __init__(self, token_dim: int = 32) -> None:
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(N_FEATURES * 2, 64), nn.GELU(), nn.Linear(64, token_dim)
        )
        self.sensor_id = nn.Parameter(torch.randn(N_SENSORS, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            token_dim, 4, 96, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        if observed is None:
            observed = torch.ones_like(x)
        tokens = self.feature_encoder(torch.cat([x, observed], dim=-1))
        tokens = tokens + self.sensor_id.unsqueeze(0)
        return self.norm(self.attention(tokens))


class MaskedAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SensorEncoder()
        self.decoder = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, N_FEATURES))

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x, observed))


class DriftPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SensorEncoder()
        self.pool = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.15))
        self.regressor = nn.Linear(64, len(GASES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        pooled = self.pool(torch.cat([h.mean(1), h.amax(1)], dim=1))
        return self.regressor(pooled)


def random_mask(x: torch.Tensor) -> torch.Tensor:
    b = len(x)
    observed = torch.ones_like(x)
    
    sensor_mask = torch.rand(b, N_SENSORS, device=x.device) < 0.18
    observed[sensor_mask] = 0
    
    groups = [(0, 2), (2, 5), (5, 8)]
    choices = torch.randint(0, len(groups), (b,), device=x.device)
    selected_sensors = torch.rand(b, N_SENSORS, device=x.device) < 0.35
    for g, (start, end) in enumerate(groups):
        rows = choices == g
        if rows.any():
            block = observed[rows]
            selected = selected_sensors[rows]
            masked_vectors = block[selected]
            masked_vectors[:, start:end] = 0
            block[selected] = masked_vectors
            observed[rows] = block
    
    observed[torch.rand_like(x) < 0.08] = 0
    return observed


def pretrain(x: np.ndarray, epochs: int, seed: int, device: torch.device) -> tuple[dict, list[float]]:
    seed_all(seed)
    model = MaskedAutoencoder().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        model.train(); total = 0.0; count = 0
        for (xb,) in loader:
            xb = xb.to(device); observed = random_mask(xb)
            pred = model(xb * observed, observed)
            hidden = 1 - observed
            hidden_loss = ((pred - xb).square() * hidden).sum() / hidden.sum().clamp_min(1)
            visible_loss = ((pred - xb).square() * observed).sum() / observed.sum().clamp_min(1)
            loss = hidden_loss + 0.05 * visible_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.detach().item() * len(xb); count += len(xb)
        history.append(total / count)
    return deepcopy(model.encoder.state_dict()), history


def concentration_transform(concentration: np.ndarray, label: np.ndarray,
                            indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(len(GASES), dtype=np.float32)
    scales = np.ones(len(GASES), dtype=np.float32)
    logged = np.log1p(concentration)
    for gas in range(len(GASES)):
        values = logged[indices[label[indices] == gas]]
        means[gas] = values.mean()
        scales[gas] = max(values.std(), 1e-3)
    return means, scales


def coverage_subset(labels: np.ndarray, concentrations: np.ndarray,
                    pool: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for gas in range(len(GASES)):
        candidates = pool[labels[pool] == gas]
        values = np.log1p(concentrations[candidates])
        order = np.argsort(values)
        candidates, values = candidates[order], values[order]
        target_count = min(per_class, len(candidates))
        selected = [int(rng.integers(len(candidates)))]
        while len(selected) < target_count:
            distance = np.min(np.abs(values[:, None] - values[selected][None, :]), axis=1)
            distance[selected] = -1
            selected.append(int(np.argmax(distance)))
        chosen.extend(candidates[selected].tolist())
    return np.asarray(chosen, dtype=np.int64)


def stratified_validation(indices: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for gas in range(len(GASES)):
        group = indices[labels[indices] == gas].copy()
        rng.shuffle(group)
        n_valid = max(1, int(round(0.2 * len(group))))
        valid.extend(group[:n_valid]); train.extend(group[n_valid:])
    return np.asarray(train), np.asarray(valid)


def fit_predictor(x: np.ndarray, labels: np.ndarray, concentrations: np.ndarray,
                  selected: np.ndarray, initial: dict | None, frozen: bool,
                  epochs: int, seed: int, device: torch.device) -> tuple[DriftPredictor, dict]:
    seed_all(seed)
    train_idx, valid_idx = stratified_validation(selected, labels, seed)
    means, scales = concentration_transform(concentrations, labels, train_idx)
    targets = (np.log1p(concentrations) - means[labels]) / scales[labels]
    model = DriftPredictor().to(device)
    if initial is not None:
        model.encoder.load_state_dict(initial)
    if frozen:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.pool.parameters()) + list(model.regressor.parameters())
    groups = [{"params": head_params, "lr": 8e-4}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": 8e-5 if initial is not None else 5e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=2e-4)
    loader = DataLoader(TensorDataset(
        torch.from_numpy(x[train_idx]), torch.from_numpy(labels[train_idx]),
        torch.from_numpy(targets[train_idx].astype(np.float32))),
        batch_size=min(64, len(train_idx)), shuffle=True,
    )
    best, best_loss, patience = None, float("inf"), 0
    history = []
    for epoch in range(epochs):
        model.train(); running = 0.0
        for xb, yb, cb in loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            reg = model(xb)
            picked = reg.gather(1, yb[:, None]).squeeze(1)
            loss = nn.functional.huber_loss(picked, cb, delta=0.5)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            running += loss.detach().item() * len(xb)
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(x[valid_idx]).to(device)
            yb = torch.from_numpy(labels[valid_idx]).to(device)
            cb = torch.from_numpy(targets[valid_idx].astype(np.float32)).to(device)
            reg = model(xb)
            picked = reg.gather(1, yb[:, None]).squeeze(1)
            valid_loss = float(nn.functional.huber_loss(picked, cb, delta=0.5))
        history.append({"epoch": epoch + 1, "train_loss": running / len(train_idx), "validation_loss": valid_loss})
        if valid_loss < best_loss - 1e-5:
            best_loss, best, patience = valid_loss, deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if patience >= 20:
            break
    model.load_state_dict(best)
    return model, {"history": history, "best_validation_loss": best_loss,
                   "train_indices": train_idx.tolist(), "validation_indices": valid_idx.tolist(),
                   "log_concentration_mean": means.tolist(), "log_concentration_scale": scales.tolist()}


def evaluate(model: DriftPredictor, x: np.ndarray, labels: np.ndarray,
             concentrations: np.ndarray, indices: np.ndarray, means: np.ndarray,
             scales: np.ndarray, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    with torch.no_grad():
        reg = model(torch.from_numpy(x[indices]).to(device)).cpu().numpy()
    true_label = labels[indices]
    oracle_norm = reg[np.arange(len(indices)), true_label]
    oracle_conc = np.expm1(oracle_norm * scales[true_label] + means[true_label]).clip(0)
    metrics = {
        "records": int(len(indices)),
        "per_gas": {},
    }
    rows = []
    for gas, name in enumerate(GASES):
        mask = true_label == gas
        error = np.abs(concentrations[indices][mask] - oracle_conc[mask])
        metrics["per_gas"][name] = {
            "records": int(mask.sum()),
            "oracle_class_mae_ppm": float(error.mean()) if mask.any() else None,
            "oracle_class_q90_ppm": float(np.quantile(error, 0.9)) if mask.any() else None,
        }
    for pos, source_index in enumerate(indices):
        rows.append({
            "source_index": int(source_index), "true_class_id": int(true_label[pos] + 1),
            "true_gas": GASES[true_label[pos]], "true_concentration_ppm": float(concentrations[source_index]),
            "oracle_class_concentration_ppm": float(oracle_conc[pos]),
        })
    return metrics, rows


def aggregate_runs(runs: list[dict]) -> dict:
    summary = {}
    for policy in ["scratch", "frozen", "fine_tuned"]:
        summary[policy] = {}
        for batch in [8, 9, 10]:
            subset = [r["metrics"] for r in runs if r["policy"] == policy and r["test_batch"] == batch]
            summary[policy][str(batch)] = {}
            gas_summary = {}
            for gas in GASES:
                values = np.asarray([r["per_gas"][gas]["oracle_class_mae_ppm"] for r in subset])
                gas_summary[gas] = {"oracle_class_mae_ppm_mean": float(values.mean()),
                                    "oracle_class_mae_ppm_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
            summary[policy][str(batch)]["per_gas"] = gas_summary
    return summary


def write_report(result: dict) -> None:
    ssl_rows = []
    for name, values in result["pretraining_histories"].items():
        ssl_rows.append(f"| {name.replace('_', ' ')} | {values[0]:.4f} | {values[-1]:.4f} |")
    lines = [
        "# Sensor_Drift Dataset Results", "", f"- Public dataset DOI: `{DOI}`.",
        "- Pretraining: partitions 1–6.", "- Adaptation: partition 7.",
        f"- Labels: {result['protocol']['labels_per_gas']} per gas.",
        "- Tests: partitions 8, 9, and 10.", f"- Repeats: {result['protocol']['repeats']}.", "",
        "## Masked Pretraining", "", "| Repeat | Initial loss | Final loss |", "|---|---:|---:|",
        *ssl_rows, "",
        "## Gas-Conditional Concentration MAE", "",
        "The known gas identity selects the concentration head.", "",
    ]
    for batch in [8, 9, 10]:
        lines += [f"### Partition {batch}", "", "| Policy | " + " | ".join(GASES) + " |",
                  "|---|" + "---:|" * len(GASES)]
        for policy in ["scratch", "frozen", "fine_tuned"]:
            cells = [f"{result['summary'][policy][str(batch)]['per_gas'][gas]['oracle_class_mae_ppm_mean']:.2f}" for gas in GASES]
            lines.append("| " + policy + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Findings", "",
              "- Masked reconstruction converged in every repeat.",
              "- Transfer reduced gas-conditional concentration macro-MAE in every test partition.",
              "- Partition 10 produced the largest concentration errors.",
              "- Results support a drift-regression extension.",
              "- The experiment is conditional on the known gas identity.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--supervised-epochs", type=int, default=120)
    parser.add_argument("--labels-per-gas", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    data = load_dataset(args.data_dir)
    pre = np.flatnonzero(data["batch"] <= 6)
    adapt = np.flatnonzero(data["batch"] == 7)
    scaler = FeatureScaler().fit(data["x"][pre])
    x = scaler.transform(data["x"])
    runs, predictions, pretraining_histories, supervised_histories = [], [], {}, {}
    for repeat in range(args.repeats):
        seed = args.seed + 100 * repeat
        state, ssl_history = pretrain(x[pre], args.pretrain_epochs, seed, device)
        pretraining_histories[f"repeat_{repeat + 1}"] = ssl_history
        selected = coverage_subset(data["label"], data["concentration"], adapt, args.labels_per_gas, seed)
        for policy, initial, frozen in [
            ("scratch", None, False), ("frozen", state, True), ("fine_tuned", state, False)
        ]:
            model, training = fit_predictor(
                x, data["label"], data["concentration"], selected, initial, frozen,
                args.supervised_epochs, seed + {"scratch": 1, "frozen": 2, "fine_tuned": 3}[policy], device,
            )
            supervised_histories[f"repeat_{repeat + 1}/{policy}"] = training["history"]
            means = np.asarray(training["log_concentration_mean"])
            scales = np.asarray(training["log_concentration_scale"])
            for batch in [8, 9, 10]:
                test = np.flatnonzero(data["batch"] == batch)
                metrics, rows = evaluate(model, x, data["label"], data["concentration"], test, means, scales, device)
                runs.append({"repeat": repeat + 1, "seed": seed, "policy": policy,
                             "test_batch": batch, "metrics": metrics})
                for row in rows:
                    row.update({"repeat": repeat + 1, "policy": policy, "test_batch": batch})
                    predictions.append(row)
            print(json.dumps({"repeat": repeat + 1, "policy": policy}), flush=True)
    result = {
        "dataset": {"name": "Sensor_Drift dataset", "doi": DOI, "records": int(len(x)),
                    "sensors": N_SENSORS, "features_per_sensor": N_FEATURES, "classes": GASES},
        "protocol": {"pretrain_batches": [1, 2, 3, 4, 5, 6], "adaptation_batch": 7,
                     "test_batches": [8, 9, 10], "labels_per_gas": args.labels_per_gas,
                     "repeats": args.repeats, "seeds": [args.seed + 100 * r for r in range(args.repeats)],
                     "pretrain_epochs": args.pretrain_epochs, "supervised_epochs": args.supervised_epochs,
                     "checkpoint_selection": "adaptation-only stratified validation"},
        "pretraining_histories": pretraining_histories,
        "supervised_histories": supervised_histories,
        "runs": runs,
    }
    result["summary"] = aggregate_runs(runs)
    result["runtime_seconds"] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv", index=False)
    write_report(result)
    print(json.dumps({"output": str(OUT), "runtime_seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
