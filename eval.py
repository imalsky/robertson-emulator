#!/usr/bin/env python3
"""
eval.py

Minimal evaluation/visualization for Robertson flow-map emulators (Residual MLP + DeepONet)
trained by train_robertson.py.

This script does ONLY:
- Use the TEST split only.
- Pick ONE random test trajectory.
- Autoregressive rollout over the full trajectory using the true per-step Δt sequence.
- Save TWO figures:
    1) True vs MLP (predicted points shown as square markers at each jump)
    2) True vs DeepONet (predicted points shown as square markers at each jump)
  Both figures are log-log (time vs concentration), and use *denormalized physical concentrations*.

It also prints summary stats for the selected trajectory's Δt sequence (min/median/max and a short prefix).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

plt.style.use('science.mplstyle')

# -----------------------------
# CONFIG (edit here)
# -----------------------------
@dataclass(frozen=True)
class EvalConfig:
    log_dir: str = "runs"
    run_name: str = "baseline"
    ckpt: str = "best"  # "best" or "last"
    eval_device: str = "auto"  # "auto" | "cpu" | "gpu"
    seed: int = 0
    out_dirname: str = "plots"

    # Base name; script will append _mlp.png and _deeponet.png
    out_png_stem: str = "test_traj_loglog"


EVAL = EvalConfig()


# -----------------------------
# Device selection
# -----------------------------
def _prefer_gpu_device() -> jax.Device:
    try:
        gpus = jax.devices("gpu")
        if gpus:
            return gpus[0]
    except RuntimeError:
        pass
    return jax.devices("cpu")[0]


def _select_device(kind: str) -> jax.Device:
    kind_l = kind.lower().strip()
    if kind_l == "auto":
        return _prefer_gpu_device()
    if kind_l == "cpu":
        return jax.devices("cpu")[0]
    if kind_l == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("eval_device='gpu' requested but no GPU devices found.")
        return gpus[0]
    raise ValueError(f"Unknown device kind: {kind!r}")


# -----------------------------
# Shared model code (must match train_robertson.py)
# -----------------------------
def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    name_l = name.lower()
    if name_l == "relu":
        return jax.nn.relu
    if name_l == "swish":
        return jax.nn.silu
    if name_l == "gelu":
        return jax.nn.gelu
    if name_l == "tanh":
        return jnp.tanh
    if name_l == "elu":
        return jax.nn.elu
    raise ValueError(f"Unknown activation: {name}")


def mlp_apply(
    params: Any,  # List[Dict[str, jax.Array]]
    x: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = act(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def deeponet_apply(
    params: Any,  # Dict[str, Any]
    x_branch: jnp.ndarray,
    x_trunk: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    b = mlp_apply(params["branch"], x_branch, act)
    t = mlp_apply(params["trunk"], x_trunk, act)
    h = b * t
    head = params["head"]
    return h @ head["W"] + head["b"]


# -----------------------------
# IO helpers
# -----------------------------
def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_params(path: Path, device: jax.Device) -> Any:
    with open(path, "rb") as f:
        params_np = pickle.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params_np)
    return jax.device_put(params, device=device)


def load_norm_stats(path: Path, device: jax.Device) -> Dict[str, jnp.ndarray]:
    raw = load_npz(path)
    norm: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        if k in ("eps", "min_std"):
            norm[k] = jax.device_put(jnp.asarray(float(v), dtype=jnp.float32), device=device)
        else:
            norm[k] = jax.device_put(jnp.asarray(v, dtype=jnp.float32), device=device)
    return norm


def _resolve_dataset_path(run_dir: Path) -> Path:
    p_txt = run_dir / "dataset_path.txt"
    stored = Path(p_txt.read_text().strip())

    if stored.exists():
        return stored

    cand = (run_dir / stored).resolve()
    if cand.exists():
        return cand

    name = stored.name
    candidates = [
        run_dir.parent.parent / "robertson_data_cache" / name,
        run_dir.parent / "robertson_data_cache" / name,
        Path("robertson_data_cache") / name,
        run_dir / name,
    ]
    for c in candidates:
        if c.exists():
            return c

    return stored


# -----------------------------
# Time grid inference (robust)
# -----------------------------
def infer_time_and_dts(dts_or_t: np.ndarray, y_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_abs: shape [y_len] strictly increasing, with t_abs[0] possibly 0
      dts:   shape [y_len-1] positive step sizes (Δt)

    Supports two dataset conventions:
      A) stored per-step Δt (length y_len-1)
      B) stored absolute time points (length y_len)  (rare, but support defensively)
    """
    v = np.asarray(dts_or_t, dtype=np.float64).reshape(-1)
    if v.size == y_len - 1:
        dts = v
        if np.any(dts <= 0.0):
            raise ValueError("Found non-positive Δt in test trajectory.")
        t_abs = np.concatenate([[0.0], np.cumsum(dts)])
        return t_abs, dts

    if v.size == y_len:
        t_abs = v
        if np.any(np.diff(t_abs) <= 0.0):
            raise ValueError("Stored time points are not strictly increasing.")
        dts = np.diff(t_abs)
        if np.any(dts <= 0.0):
            raise ValueError("Derived Δt contains non-positive values.")
        return t_abs, dts

    raise ValueError(f"Cannot infer time grid: got {v.size} values, but y_len={y_len}.")


def make_log_time(t_abs: np.ndarray) -> np.ndarray:
    """
    For log-x plotting: replace any non-positive entries with a small positive value
    based on the smallest positive time in the trajectory (not a fixed 1e-300).
    """
    t = np.asarray(t_abs, dtype=np.float64).copy()
    pos = t[t > 0.0]
    if pos.size == 0:
        raise ValueError("All times are non-positive; cannot log-plot.")
    t0 = float(pos.min() * 0.5)
    t[t <= 0.0] = t0
    return t


# -----------------------------
# Numerics helpers
# -----------------------------
def _zscore(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, min_std: jnp.ndarray) -> jnp.ndarray:
    return (x - mean) / jnp.clip(std, min_std, jnp.inf)


def _log10_safe(y: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    return jnp.log10(y + eps)


def _logy_to_y_simplex(logy: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    y = jnp.power(10.0, logy) - eps
    y = jnp.clip(y, 0.0, jnp.inf)
    y = y / jnp.clip(jnp.sum(y, axis=-1, keepdims=True), 1e-30, jnp.inf)
    return y


# -----------------------------
# JIT-compiled rollouts (act_name is static)
# -----------------------------
@partial(jax.jit, static_argnames=("act_name",))
def rollout_mlp_full(
    params: Any,
    act_name: str,
    norm: Mapping[str, jnp.ndarray],
    y0: jnp.ndarray,      # [3] physical
    dts: jnp.ndarray,     # [T] physical
) -> jnp.ndarray:
    act = get_activation(act_name)
    eps = norm["eps"]
    min_std = norm["min_std"]

    logy0 = _log10_safe(y0, eps)
    logdt = jnp.log10(dts)

    def step_fn(logy_curr: jnp.ndarray, logdt_step: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_state = _zscore(logy_curr, norm["state_mean"], norm["state_std"], min_std)
        x_dt = _zscore(logdt_step[None], norm["dt_mean"], norm["dt_std"], min_std)
        x = jnp.concatenate([x_state, x_dt], axis=-1)[None, :]

        pred_norm = mlp_apply(params, x, act)[0]  # normalized Δlog
        dlog = pred_norm * jnp.clip(norm["out_std_mlp"], min_std, jnp.inf) + norm["out_mean_mlp"]
        logy_next = logy_curr + dlog
        y_next = _logy_to_y_simplex(logy_next[None, :], eps)[0]
        return logy_next, y_next

    _, ys_next = lax.scan(step_fn, logy0, logdt)
    return jnp.concatenate([y0[None, :], ys_next], axis=0)


@partial(jax.jit, static_argnames=("act_name",))
def rollout_deeponet_full(
    params: Any,
    act_name: str,
    norm: Mapping[str, jnp.ndarray],
    y0: jnp.ndarray,      # [3] physical
    dts: jnp.ndarray,     # [T] physical
) -> jnp.ndarray:
    act = get_activation(act_name)
    eps = norm["eps"]
    min_std = norm["min_std"]

    logy0 = _log10_safe(y0, eps)
    logdt = jnp.log10(dts)

    def step_fn(logy_curr: jnp.ndarray, logdt_step: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_state = _zscore(logy_curr, norm["state_mean"], norm["state_std"], min_std)
        x_dt = _zscore(logdt_step[None], norm["dt_mean"], norm["dt_std"], min_std)

        pred_norm = deeponet_apply(params, x_state[None, :], x_dt[None, :], act)[0]
        logy_next = pred_norm * jnp.clip(norm["out_std_deeponet"], min_std, jnp.inf) + norm["out_mean_deeponet"]

        y_next = _logy_to_y_simplex(logy_next[None, :], eps)[0]
        return logy_next, y_next

    _, ys_next = lax.scan(step_fn, logy0, logdt)
    return jnp.concatenate([y0[None, :], ys_next], axis=0)


# -----------------------------
# Plotting
# -----------------------------
def _plot_one_model(
    t_log: np.ndarray,          # [T+1], positive
    y_true: np.ndarray,         # [T+1,3], physical
    y_pred: np.ndarray,         # [T+1,3], physical
    out_png: Path,
    title: str,
    eps_plot: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)

    labels = ["y1", "y2", "y3"]
    for i, ax in enumerate(axes):
        # True: continuous line
        ax.loglog(t_log, y_true[:, i] + eps_plot, "-", label="True")

        # Pred: show exact jump values with square markers (and a faint connecting line for readability)
        ax.loglog(
            t_log,
            y_pred[:, i] + eps_plot,
            "-",
            marker="s",
            markersize=3.0,
            linewidth=0.8,
            label="Pred",
        )

        ax.set_ylabel(labels[i])
        ax.grid(True, which="both", alpha=0.3)
        if i == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("time t")
    fig.suptitle(title, y=0.98)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _print_dt_summary(dts: np.ndarray, max_show: int = 12) -> None:
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    print("Δt summary for selected TEST trajectory:")
    print(f"  n_steps: {dts.size}")
    print(f"  min:     {dts.min():.6e}")
    print(f"  median:  {np.median(dts):.6e}")
    print(f"  max:     {dts.max():.6e}")
    show = min(max_show, dts.size)
    head = " ".join(f"{x:.3e}" for x in dts[:show])
    print(f"  first {show} Δt: {head}")
    if dts.size > show:
        tail = " ".join(f"{x:.3e}" for x in dts[-min(5, dts.size):])
        print(f"  last {min(5, dts.size)} Δt:  {tail}")


# -----------------------------
# Main
# -----------------------------
def main(cfg: EvalConfig) -> None:
    run_dir = Path(cfg.log_dir) / cfg.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg_used_path = run_dir / "config_used.json"
    if not cfg_used_path.exists():
        raise FileNotFoundError(f"Missing config_used.json: {cfg_used_path}")
    cfg_used = json.loads(cfg_used_path.read_text())
    act_name = str(cfg_used.get("activation", "swish"))

    dataset_path = _resolve_dataset_path(run_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset cache not found: {dataset_path}")

    splits_path = run_dir / "splits.npz"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits.npz: {splits_path}")

    splits = load_npz(splits_path)
    test_idx = splits.get("test_idx", None)
    if test_idx is None or np.asarray(test_idx).size == 0:
        raise RuntimeError("No test trajectories found (test_idx missing or empty).")
    test_idx = np.asarray(test_idx, dtype=np.int64)

    data = load_npz(dataset_path)
    ys_all = np.asarray(data["ys"][test_idx], dtype=np.float64)       # [Ntest, T+1, 3]
    dts_all = np.asarray(data["dts"][test_idx], dtype=np.float64)     # [Ntest, T] (or [Ntest, T+1] if stored times)
    if ys_all.ndim != 3 or ys_all.shape[-1] != 3:
        raise ValueError(f"Unexpected ys shape: {ys_all.shape}")
    if dts_all.ndim != 2:
        raise ValueError(f"Unexpected dts shape: {dts_all.shape}")

    rng = np.random.default_rng(int(cfg.seed))
    pick = int(rng.integers(0, ys_all.shape[0]))
    y_true = ys_all[pick]      # [T+1, 3]
    dts_or_t = dts_all[pick]   # [T] or [T+1]

    t_abs, dts = infer_time_and_dts(dts_or_t, y_len=y_true.shape[0])
    t_log = make_log_time(t_abs)

    _print_dt_summary(dts)

    dev = _select_device(cfg.eval_device)

    norm_path = run_dir / "norm_stats.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm_stats.npz: {norm_path}")
    norm = load_norm_stats(norm_path, device=dev)

    ckpt = cfg.ckpt.lower().strip()
    if ckpt not in ("best", "last"):
        raise ValueError("ckpt must be 'best' or 'last'")

    mlp_path = run_dir / "models" / f"mlp_{ckpt}.pkl"
    deeponet_path = run_dir / "models" / f"deeponet_{ckpt}.pkl"
    if not mlp_path.exists():
        raise FileNotFoundError(f"Missing MLP params: {mlp_path}")
    if not deeponet_path.exists():
        raise FileNotFoundError(f"Missing DeepONet params: {deeponet_path}")

    mlp_params = load_params(mlp_path, device=dev)
    deeponet_params = load_params(deeponet_path, device=dev)

    y0 = jax.device_put(jnp.asarray(y_true[0], dtype=jnp.float32), device=dev)
    dts_j = jax.device_put(jnp.asarray(dts, dtype=jnp.float32), device=dev)

    y_mlp = np.asarray(rollout_mlp_full(mlp_params, act_name=act_name, norm=norm, y0=y0, dts=dts_j))
    y_deep = np.asarray(rollout_deeponet_full(deeponet_params, act_name=act_name, norm=norm, y0=y0, dts=dts_j))

    out_dir = run_dir / cfg.out_dirname
    out_png_mlp = out_dir / f"{cfg.out_png_stem}_mlp.png"
    out_png_deep = out_dir / f"{cfg.out_png_stem}_deeponet.png"

    eps_plot = float(np.asarray(norm["eps"]))

    _plot_one_model(
        t_log=t_log,
        y_true=y_true,
        y_pred=y_mlp,
        out_png=out_png_mlp,
        title="Random TEST trajectory: True vs MLP (square markers = jump predictions)",
        eps_plot=eps_plot,
    )
    _plot_one_model(
        t_log=t_log,
        y_true=y_true,
        y_pred=y_deep,
        out_png=out_png_deep,
        title="Random TEST trajectory: True vs DeepONet (square markers = jump predictions)",
        eps_plot=eps_plot,
    )

    print(f"Saved: {out_png_mlp}")
    print(f"Saved: {out_png_deep}")
    print(f"(Picked trajectory index within test split: {pick})")


if __name__ == "__main__":
    main(EVAL)
