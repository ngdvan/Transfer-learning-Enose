from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import same_device_session_transfer as core


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CACHE = PACKAGE_ROOT / "datasets" / "dynamic_gas_mixture" / "raw_sequences.npz"
OUT = PACKAGE_ROOT / "results" / "co_ethylene"
GASES = ["CO", "Ethylene"]
TARGET_SCALE = np.asarray([533.33, 20.0], np.float32)


core.N_SENSORS = 16
core.TARGETS = GASES
core.TARGET_SCALE = TARGET_SCALE


def load_data(cache: Path) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(cache)
    groups = z["groups"].astype(np.int32)
    unique_groups = np.unique(groups)
    if len(unique_groups) != 245:
        raise ValueError(f"Expected 245 exposure runs, found {len(unique_groups)}")
    block_map = {
        int(group): block
        for block, chunk in enumerate(np.array_split(unique_groups, 5), 1)
        for group in chunk
    }
    blocks = np.asarray([block_map[int(g)] for g in groups], np.int32)
    audit = {
        "sequence_windows": int(len(groups)),
        "exposure_runs": int(len(unique_groups)),
        "sensor_channels": int(z["sequences"].shape[2]),
        "chronological_blocks": 5,
        "exposures_per_block": {
            str(b): int(np.unique(groups[blocks == b]).size) for b in range(1, 6)
        },
        "unique_pairs_per_block": {
            str(b): int(np.unique(z["targets"][blocks == b], axis=0).shape[0])
            for b in range(1, 6)
        },
        "offsets": sorted(map(int, np.unique(z["offsets"]))),
    }
    return {
        "x": z["sequences"].astype(np.float32),
        "y": z["targets"].astype(np.float32),
        "groups": groups,
        "offsets": z["offsets"].astype(np.int32),
        "blocks": blocks,
    }, audit


def farthest_order(targets: np.ndarray, seed: int) -> np.ndarray:
    unique, first = np.unique(targets, axis=0, return_index=True)
    z = unique / TARGET_SCALE
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(unique)))]
    distance = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < len(unique):
        candidate = int(np.argmax(distance + rng.uniform(0, 1e-8, len(distance))))
        chosen.append(candidate)
        distance = np.minimum(
            distance, np.linalg.norm(z - z[candidate], axis=1)
        )
    return first[np.asarray(chosen)]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for j, gas in enumerate(GASES):
        mae = mean_absolute_error(y[:, j], pred[:, j])
        out[gas] = {
            "mae_ppm": float(mae),
            "rmse_ppm": float(np.sqrt(mean_squared_error(y[:, j], pred[:, j]))),
            "r2": float(r2_score(y[:, j], pred[:, j])),
            "mae_percent_test_mean": float(100 * mae / y[:, j].mean()),
            "mae_percent_full_scale": float(100 * mae / TARGET_SCALE[j]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data, audit = load_data(args.cache)
    pre = data["blocks"] <= 3
    adapt_zero = (data["blocks"] == 4) & (data["offsets"] == 0)
    test = (data["blocks"] == 5) & (data["offsets"] == 0)
    scaler = core.SensorScaler().fit(data["x"][pre])
    pre_x = scaler.transform(data["x"][pre])
    test_x = scaler.transform(data["x"][test])
    test_y, test_groups = data["y"][test], data["groups"][test]

    pretrained, history = core.pretrain(pre_x, args.pretrain_epochs, args.seed)
    import torch
    torch.save(pretrained, args.output / "pretrained_encoder.pt")
    np.savez(args.output / "scaler.npz", center=scaler.center, scale=scaler.scale)

    adapt_x0 = data["x"][adapt_zero]
    adapt_y0 = data["y"][adapt_zero]
    adapt_g0 = data["groups"][adapt_zero]
    max_budget = int(np.unique(adapt_y0, axis=0).shape[0])
    budgets = sorted(set([5, 10, 20, max_budget]))
    rows, prediction_rows = [], []

    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat * 100
        order = farthest_order(adapt_y0, repeat_seed)
        ordered_groups = adapt_g0[order]
        for budget in budgets:
            selected_groups = ordered_groups[:budget]
            train = (data["blocks"] == 4) & np.isin(data["groups"], selected_groups)
            train_x = scaler.transform(data["x"][train])
            train_y = data["y"][train]
            model_predictions = {}
            for name, initial, frozen, seed_add in (
                ("scratch", None, False, 0),
                ("frozen_transfer", pretrained, True, 1),
                ("finetuned_transfer", pretrained, False, 2),
            ):
                model = core.fit_neural(
                    train_x, train_y, initial, frozen, args.finetune_epochs,
                    repeat_seed + budget + seed_add,
                )
                model_predictions[name] = core.predict(model, test_x)

            for name, pred in model_predictions.items():
                row = {
                    "repeat": repeat + 1, "seed": repeat_seed,
                    "budget": budget, "model": name,
                    "metrics": metrics(test_y, pred),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                prediction_rows.extend({
                    "repeat": repeat + 1, "budget": budget, "model": name,
                    "group": int(group), "CO_true": float(y[0]),
                    "Ethylene_true": float(y[1]), "CO_pred": float(p[0]),
                    "Ethylene_pred": float(p[1]),
                } for group, y, p in zip(test_groups, test_y, pred))

    result = {
        "experiment": "same-device sensor-aware chronological transfer for CO/ethylene",
        "protocol": {
            "session_warning": "No explicit sessions exist; contiguous chronological acquisition blocks are used.",
            "unlabeled_pretraining_blocks": [1, 2, 3],
            "labeled_adaptation_block": 4,
            "untouched_future_test_block": 5,
            "label_budgets_unique_pairs": budgets,
            "subset_selection": "seeded farthest-point coverage in gas concentration space",
            "repeats": args.repeats,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "encoder": "shared temporal CNN + sensor identity embeddings + cross-sensor Transformer",
            "ssl_objective": "masked temporal spans and complete-sensor reconstruction",
            "test_concentration_mean_ppm": dict(zip(GASES, map(float, test_y.mean(0)))),
            "full_scale_ppm": dict(zip(GASES, map(float, TARGET_SCALE))),
        },
        "data_audit": audit,
        "pretraining_loss": history,
        "runs": rows,
        "summary": core.aggregate(rows),
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
