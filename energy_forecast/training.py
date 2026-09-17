"""模块用途：训练作者 SparseTSF，保存最佳权重与原单位验证结果。

模块边界：仅允许 train/val 窗口，不定义神经网络、不评价测试集、不自动重试。
"""

import copy
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .baselines import metrics
from .data import PROJECT_ROOT, load_prepared, positive_int, provenance, read_json, sha256, write_json
from .sparsetsf import build_model, target_indices, verify_source


CONFIG_PATH = PROJECT_ROOT / "configs/sparsetsf_erco.json"
DEFAULT_TRAINING_CONFIG = read_json(CONFIG_PATH)


def validate_training_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_TRAINING_CONFIG):
        raise ValueError("Training config must be an object with known fields only")
    result = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    result.update(config)
    for key in ("period_len", "d_model", "batch_size", "epochs", "patience", "timeout_seconds"):
        positive_int(result[key], key)
    seed = result["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    lr = result["learning_rate"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if result["model_type"] not in ("linear", "mlp"):
        raise ValueError("Only the author linear / mlp branches are allowed")
    return result


def learning_rate(base, completed_epoch):
    return base if completed_epoch < 3 else base * .8 ** (completed_epoch - 3)


def finite_tensor(value, label):
    if not torch.isfinite(value).all().item():
        raise ValueError(f"Non-finite {label}")


class EarlyStopping:
    """按完整验证损失严格下降选权重；相等也算一次未改善。"""

    def __init__(self, patience):
        positive_int(patience, "patience")
        self.patience, self.bad_epochs = patience, 0
        self.best, self.best_epoch = math.inf, None

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience

    def update(self, value, epoch):
        if not math.isfinite(value):
            raise ValueError("Non-finite validation loss")
        if value < self.best:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False


class WindowDataset(Dataset):
    """按已登记起点惰性取窗；本模块故意不开放 test split。"""

    def __init__(self, data, split):
        if split not in ("train", "val"):
            raise ValueError("Only train and val datasets are allowed; test evaluation is separate")
        self.data, self.origins = data, data.origins[split]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        x, y = self.data.sample(int(self.origins[index]))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _device_and_seed(device, seed):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu (explicit tests) or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device(device)


def _setup(prepared, code_dir, config, device):
    config = validate_training_config(config)
    target_device = _device_and_seed(device, config["seed"])
    data, manifest = load_prepared(prepared)
    model, source = build_model(code_dir, data, config["model_type"], config["period_len"], config["d_model"])
    return data, manifest, model.to(target_device), source, config, target_device


def preflight(prepared, code_dir, config, device="cuda"):
    data, _, model, source, config, target_device = _setup(prepared, code_dir, config, device)
    x, _ = WindowDataset(data, "train")[0]
    with torch.no_grad():
        output = model(x[None].to(target_device))[..., target_indices(data)]
    finite_tensor(output, "preflight output")
    expected = (1, data.config["horizon"], len(data.config["target_columns"]))
    if tuple(output.shape) != expected:
        raise ValueError("Unexpected author model output shape")
    return {"status": "PREFLIGHT_ONLY_NO_TRAINING", "source": source, "config": config,
            "parameters": sum(p.numel() for p in model.parameters()), "output_shape": list(output.shape),
            "device": str(target_device), "torch": str(torch.__version__), "data_splits": data.splits}


def _epoch(model, loader, indices, device, optimizer=None, collect=False):
    model.train(optimizer is not None)
    squared_sum, count, predictions = 0.0, 0, []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            finite_tensor(x, "input")
            finite_tensor(y, "target")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)[..., indices]
            if prediction.shape != y.shape:
                raise ValueError("Prediction/target shape mismatch")
            finite_tensor(prediction, "prediction")
            loss = torch.nn.functional.mse_loss(prediction, y)
            finite_tensor(loss, "loss")
            squared_sum += (prediction.detach().double() - y.double()).square().sum().item()
            count += y.numel()
            if optimizer is not None:
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise ValueError(f"Missing gradient: {name}")
                    finite_tensor(parameter.grad, f"gradient {name}")
                optimizer.step()
                for name, parameter in model.named_parameters():
                    finite_tensor(parameter, f"weight {name}")
            if collect:
                predictions.append(prediction.detach().cpu().numpy())
    result = squared_sum / count
    if not math.isfinite(result):
        raise ValueError("Non-finite epoch loss")
    return result, np.concatenate(predictions) if collect else None


def _save_validation(output, data, prediction, report):
    origins = data.origins["val"]
    actual = np.stack([data.sample(int(j), raw=True)[1] for j in origins])
    targets = data.config["target_columns"]
    scores = metrics(actual, prediction, targets)
    name = "SparseTSF_" + report["training_config"]["model_type"]
    rows = [{"model": name, "split": "val", "target": k, **v} for k, v in scores["per_target"].items()]
    horizons = [{"model": name, "split": "val", "horizon": int(h), "target": k, **v}
                for h, values in scores["per_horizon"].items() for k, v in values.items()]
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False, mode="x")
    pd.DataFrame(horizons).to_csv(output / "per_horizon_metrics.csv", index=False, mode="x")
    write_json(output / "metrics.json", {"split": "val", "model": name, "metrics": scores,
               "best_epoch": report["best_epoch"], "source_sha256": report["source_sha256"],
               "prepared_manifest_sha256": report["prepared_manifest_sha256"],
               "validation_range": data.splits["val"], "target_columns": targets})
    with (output / "validation_predictions.npz").open("xb") as handle:
        np.savez_compressed(handle, prediction=prediction, actual=actual, targets=np.asarray(targets),
                            origin_utc=np.asarray([data.times[j].isoformat() for j in origins]),
                            target_utc=np.asarray([[data.times[j + h].isoformat() for h in range(data.config["horizon"])]
                                                   for j in origins]))
    return scores


def train(prepared, code_dir, output, config, device="cuda"):
    """独立新运行；CPU 只用于明确调用本函数的合成测试，正式 CLI 只开放 CUDA。"""
    config = validate_training_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        manifest_path = Path(prepared).resolve() / "manifest.json"
        manifest_hash = sha256(manifest_path)
        data, manifest, model, source, config, target_device = _setup(prepared, code_dir, config, device)
        indices = target_indices(data)
        run = {"training_config": config, "data_config": data.config, "scalers": data.scalers,
               "source_sha256": manifest["source_sha256"], "prepared_manifest_path": str(manifest_path),
               "prepared_manifest_sha256": manifest_hash, "author_source": source, "provenance": provenance(),
               "device": str(target_device), "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
               "device_name": torch.cuda.get_device_name(target_device) if device == "cuda" else "CPU synthetic-test execution",
               "parameters": sum(p.numel() for p in model.parameters()), "splits": data.splits,
               "deterministic_algorithms": True, "tf32": False, "amp": False,
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "loss": "equal-target standardized MSE; element-weighted epochs",
               "selection": "minimum validation standardized MSE; strictly decreasing",
               "test_evaluated": False,
               "notes": ["Unmodified author model; project-specific training/evaluation protocol, not original-paper ERCO reproduction.",
                         "Demand has no effect on scored channels in this channel-independent model.",
                         "Retrospective observations; real publication latency unvalidated.",
                         "No clipping of negative predictions; no mixed-unit overall physical score."]}
        write_json(output / "run_config.json", run)
        generator = torch.Generator().manual_seed(config["seed"])
        train_loader = DataLoader(WindowDataset(data, "train"), batch_size=config["batch_size"], shuffle=True,
                                  drop_last=False, num_workers=0, generator=generator)
        val_loader = DataLoader(WindowDataset(data, "val"), batch_size=config["batch_size"], shuffle=False,
                                drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(config["seed"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
        stopper, history = EarlyStopping(config["patience"]), []
        with (output / "train.log").open("x", encoding="utf-8") as log:
            def log_line(text):
                print(text, flush=True)
                log.write(text + "\n")
                log.flush()

            log_line(f"Author SparseTSF {config['model_type']}; parameters={run['parameters']}; device={target_device}")
            log_line("Training and validation only; no test-set evaluation.")
            log_line(f"Run directory: {output}")
            for epoch in range(1, config["epochs"] + 1):
                epoch_start = time.monotonic()
                used_lr = optimizer.param_groups[0]["lr"]
                train_loss, _ = _epoch(model, train_loader, indices, target_device, optimizer)
                val_loss, _ = _epoch(model, val_loader, indices, target_device)
                improved = stopper.update(val_loss, epoch)
                if improved:
                    pending = output / "checkpoint.pt.tmp"
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, pending)
                    pending.replace(output / "checkpoint.pt")
                row = {"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "learning_rate": used_lr,
                       "seconds": time.monotonic() - epoch_start, "best_epoch": stopper.best_epoch,
                       "improved": improved, "train_samples": len(train_loader.dataset), "val_samples": len(val_loader.dataset)}
                history.append(row)
                pd.DataFrame([row]).to_csv(output / "history.csv", index=False, mode="x" if epoch == 1 else "a", header=epoch == 1)
                log_line(f"Epoch {epoch}/{config['epochs']} | Train MSE={train_loss:.8f} | Val MSE={val_loss:.8f} | "
                         f"lr={used_lr:.8g} | best={stopper.best_epoch} | {row['seconds']:.2f}s")
                if stopper.stopped:
                    log_line(f"Early stopping: {stopper.bad_epochs} epochs without strict validation improvement.")
                    break
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(config["learning_rate"], epoch)
            model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=target_device, weights_only=True), strict=True)
            restored_loss, normalized = _epoch(model, val_loader, indices, target_device, collect=True)
            if not math.isclose(restored_loss, stopper.best, rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError("Reloaded best checkpoint does not reproduce its validation loss")
            run.update(best_epoch=stopper.best_epoch, best_val_mse=stopper.best, epochs_completed=len(history),
                       early_stopped=stopper.stopped, reloaded_val_mse=restored_loss)
            prediction = normalized.astype(np.float64) * np.asarray(data.scalers["target"]["scale"]) + np.asarray(data.scalers["target"]["mean"])
            scores = _save_validation(output, data, prediction, run)
            if (sha256(manifest_path) != manifest_hash or sha256(manifest["source_path"]) != manifest["source_sha256"]
                    or verify_source(code_dir) != source):
                raise ValueError("Source or preparation changed during training")
            carbon = scores["per_target"]["factor_generated_kg_per_mwh"]
            log_line(f"Validation carbon-factor MAE={carbon['mae']:.6f}, RMSE={carbon['rmse']:.6f} kg CO2/MWh")
            log_line("PASS: best author-model checkpoint and validation predictions saved; no test-set evaluation.")
        run.update(status="COMPLETED_VALIDATION_ONLY", elapsed_seconds=time.monotonic() - started,
                   artifact_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
        write_json(output / "run.json", run)
        return run
    except BaseException as error:
        if not (output / "run.json").exists():
            write_json(output / "run.json", {"status": "FAILED", "error": f"{type(error).__name__}: {error}",
                                             "test_evaluated": False, "elapsed_seconds": time.monotonic() - started})
        raise
