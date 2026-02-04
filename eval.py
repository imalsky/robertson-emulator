#!/usr/bin/env python3
"""
Evaluation for Robertson emulators.

What this script does (per your request):
- For EACH model (MLP, DeepONet):
    1) Pick 1 random test trajectory ("profile").
    2) Pick a random starting index inside that trajectory.
    3) Run 100-step autoregressive rollout starting from that point using the TRUE dt sequence.
    4) Plot the ENTIRE ground-truth trajectory, and superimpose ONLY the model predictions.

Output:
  runs_robertson/<run_name>/plots/
    overlay_fulltraj_ar_<ckpt>.png

Notes:
- MLP predicts normalized Δlog10(y) (residual in log-space).
- DeepONet predicts normalized log10(y_next).
- Conditioning always uses y0 (3) and k (3) from the trajectory metadata.
- Rollout is JIT-compiled; random selection + slicing happens in Python to keep indices static.
- No argparse: edit EvalConfig below.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import matplotlib.pyplot as plt


# -----------------------------
# CONFIG (edit here)
# -----------------------------
@dataclass(frozen=True)
class EvalConfig:
    log_dir: str = "runs_robertson"
    run_name: str = "baseline_v2"
    ckpt: str = "best"  # "best" or "last"

    # Autoregressive overlay
    ar_steps: int = 100
    same_profile_for_models: bool = True  # if True, both models use the same random profile + start index

    # Randomness
    seed: int = 0

    # Plot settings
    eps_plot: float = 1e-30  # for semilogy
    out_png_name: str = "overlay_fulltraj_ar"


EVAL = EvalConfig()

# x64 helps on some numerics; model compute stays float32.
jax.config.update("jax_enable_x64", True)


# -----------------------------
# Plot defaults
# -----------------------------
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
        }
    )


# -----------------------------
# Shared model code (must match training)
# -----------------------------
def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    name = name.lower()
    if name == "relu":
        return jax.nn.relu
    if name == "swish":
        return jax.nn.silu
    if name == "gelu":
        return jax.nn.gelu
    if name == "tanh":
        return jnp.tanh
    if name == "elu":
        return jax.nn.elu
    raise ValueError(f"Unknown activation: {name}")


def mlp_apply(
    params: List[Dict[str, jax.Array]],
    x: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = act(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def deeponet_apply(
    params: Dict[str, object],
    x_branch: jnp.ndarray,
    x_trunk: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    b = mlp_apply(params["branch"], x_branch, act)
    t = mlp_apply(params["trunk"], x_trunk, act)
    h = b * t
    head = params["head"]
    return h @ head["W"] + head["b"]


def forward_apply(model_type: str, params, batch: Dict[str, jnp.ndarray], act) -> jnp.ndarray:
    if model_type == "mlp":
        return mlp_apply(params, batch["x"], act)
    if model_type == "deeponet":
        return deeponet_apply(params, batch["x_branch"], batch["x_trunk"], act)
    raise ValueError(f"Unknown model_type: {model_type}")


def load_params(path: Path):
    with open(path, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params_np)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_norm_stats(path: Path) -> Dict[str, jnp.ndarray]:
    raw = load_npz(path)
    norm = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}
    norm["eps"] = jnp.asarray(float(raw["eps"]), dtype=jnp.float32)
    norm["min_std"] = jnp.asarray(float(raw["min_std"]), dtype=jnp.float32)
    return norm


def log10_safe(x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    return jnp.log10(x + eps)


def encode_inputs(
    y_t_phys: jnp.ndarray,   # [3] or [B,3]
    dt_phys: jnp.ndarray,    # [1] or [B,1]
    y0_phys: jnp.ndarray,    # [3] or [B,3]
    k_phys: jnp.ndarray,     # [3] or [B,3]
    norm: Dict[str, jnp.ndarray],
    model_type: str,
) -> Dict[str, jnp.ndarray]:
    eps = norm["eps"]
    min_std = norm["min_std"]

    logy_t = log10_safe(y_t_phys, eps)
    logdt = jnp.log10(dt_phys)
    logy0 = log10_safe(y0_phys, eps)
    logk = jnp.log10(k_phys)

    x_state = (logy_t - norm["state_mean"]) / jnp.clip(norm["state_std"], min_std, jnp.inf)
    x_dt = (logdt - norm["dt_mean"]) / jnp.clip(norm["dt_std"], min_std, jnp.inf)
    x_y0 = (logy0 - norm["y0_mean"]) / jnp.clip(norm["y0_std"], min_std, jnp.inf)
    x_k = (logk - norm["k_mean"]) / jnp.clip(norm["k_std"], min_std, jnp.inf)

    if model_type == "mlp":
        x = jnp.concatenate([x_state, x_dt, x_y0, x_k], axis=-1)  # [...,10]
        return {"x": x}

    if model_type == "deeponet":
        x_branch = jnp.concatenate([x_y0, x_k], axis=-1)          # [...,6]
        x_trunk = jnp.concatenate([x_state, x_dt], axis=-1)       # [...,4]
        return {"x_branch": x_branch, "x_trunk": x_trunk}

    raise ValueError(f"Unknown model_type: {model_type}")


def decode_to_next_y_phys(
    model_type: str,
    pred_out_norm: jnp.ndarray,   # [3] or [B,3]
    y_curr_phys: jnp.ndarray,     # [3] or [B,3]
    norm: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    eps = norm["eps"]
    min_std = norm["min_std"]

    if model_type == "deeponet":
        logy_next = pred_out_norm * jnp.clip(norm["out_std_deeponet"], min_std, jnp.inf) + norm["out_mean_deeponet"]
    elif model_type == "mlp":
        logy_curr = log10_safe(y_curr_phys, eps)
        dlog = pred_out_norm * jnp.clip(norm["out_std_mlp"], min_std, jnp.inf) + norm["out_mean_mlp"]
        logy_next = logy_curr + dlog
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    y_next = jnp.power(10.0, logy_next) - eps
    y_next = jnp.clip(y_next, 0.0, jnp.inf)
    y_next = y_next / jnp.clip(jnp.sum(y_next, axis=-1, keepdims=True), 1e-30, jnp.inf)
    return y_next


# -----------------------------
# JIT-compiled single-trajectory rollout
# -----------------------------
@partial(jax.jit, static_argnames=("model_type", "act_name"))
def rollout_from_point_jit(
    model_type: str,
    params,
    act_name: str,
    norm: Dict[str, jnp.ndarray],
    y_start: jnp.ndarray,   # [3]
    dt_seq: jnp.ndarray,    # [K,1]
    y0_const: jnp.ndarray,  # [3]
    k_const: jnp.ndarray,   # [3]
) -> jnp.ndarray:
    """
    Returns ys_pred including the start state: [K+1, 3]
    """
    act = get_activation(act_name)

    def step_fn(y_curr, dt_step):
        batch = encode_inputs(
            y_t_phys=y_curr[None, :],
            dt_phys=dt_step[None, :],
            y0_phys=y0_const[None, :],
            k_phys=k_const[None, :],
            norm=norm,
            model_type=model_type,
        )
        pred_out = forward_apply(model_type, params, batch, act)[0]  # [3]
        y_next = decode_to_next_y_phys(model_type, pred_out, y_curr, norm)
        return y_next, y_next

    _, ys_next = lax.scan(step_fn, y_start, dt_seq)
    return jnp.concatenate([y_start[None, :], ys_next], axis=0)


# -----------------------------
# Plot: full GT + overlay AR
# -----------------------------
def plot_fulltraj_overlay(
    out_png: Path,
    gt_time: np.ndarray,         # [T] (includes 0)
    gt_y: np.ndarray,            # [T,3]
    start_t: float,              # scalar time where rollout begins
    pred_time: np.ndarray,       # [K] times corresponding to predicted points (t_{start+1..start+K})
    pred_y: np.ndarray,          # [K,3] predicted y at those times
    model_label: str,
    eps_plot: float,
    ax: plt.Axes,
) -> None:
    names = ["y1", "y2", "y3"]
    # Avoid t=0 on log-x.
    x_gt = gt_time[1:]
    y_gt = gt_y[1:, :]

    ax.set_xscale("log")
    ax.set_yscale("log")

    # GT lines
    for s in range(3):
        ax.plot(x_gt, y_gt[:, s] + eps_plot, label=f"GT {names[s]}")

    # Overlay predictions as markers (unlabeled)
    for s in range(3):
        ax.scatter(
            pred_time,
            pred_y[:, s] + eps_plot,
            s=26,
            alpha=0.95,
            marker="o",
            linewidths=0.6,
            edgecolors="black",
            zorder=5,
        )

    # Mark start time
    ax.axvline(start_t, linestyle="--", linewidth=1.2)
    ax.set_title(model_label)
    ax.set_xlabel("time (cumulative dt)")
    ax.set_ylabel("mixing ratio")


# -----------------------------
# Main
# -----------------------------
def main(cfg: EvalConfig) -> None:
    set_plot_style()

    rdir = Path(cfg.log_dir) / cfg.run_name
    if not rdir.exists():
        raise FileNotFoundError(f"Run directory not found: {rdir}")

    cfg_used = json.loads((rdir / "config_used.json").read_text())
    act_name = cfg_used["activation"]

    dataset_path = Path((rdir / "dataset_path.txt").read_text().strip())
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset cache not found: {dataset_path}")

    data = load_npz(dataset_path)
    splits = load_npz(rdir / "splits.npz")
    norm = load_norm_stats(rdir / "norm_stats.npz")

    test_idx = splits["test_idx"]
    if test_idx.size == 0:
        raise RuntimeError("No test trajectories found (test split is empty).")

    ys_all = data["ys"][test_idx]         # [Ntest, T, 3] (float64)
    dts_all = data["dts"][test_idx]       # [Ntest, T-1] (float64)
    y0s_all = data["y0s"][test_idx]       # [Ntest, 3] (float64)
    ks_all = data["ks"][test_idx]         # [Ntest, 3] (float64)

    Ntest, T, _ = ys_all.shape
    K = int(cfg.ar_steps)
    if (T - 1) <= K:
        raise RuntimeError(f"Trajectory too short for ar_steps={K}: T={T}")

    rng = np.random.default_rng(cfg.seed)

    # Choose profile + start index selection policy
    def pick_profile_and_start() -> Tuple[int, int]:
        prof = int(rng.integers(0, Ntest))
        start = int(rng.integers(0, (T - 1) - K))  # ensures K steps available
        return prof, start

    if cfg.same_profile_for_models:
        shared_prof, shared_start = pick_profile_and_start()
        picks = {"mlp": (shared_prof, shared_start), "deeponet": (shared_prof, shared_start)}
    else:
        picks = {"mlp": pick_profile_and_start(), "deeponet": pick_profile_and_start()}

    plots_dir = rdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), sharey=True)

    for ax, model_type in zip(axes, ["mlp", "deeponet"]):
        prof_i, start_i = picks[model_type]

        ys = ys_all[prof_i].astype(np.float64)         # [T,3]
        dts = dts_all[prof_i].astype(np.float64)       # [T-1]
        y0c = y0s_all[prof_i].astype(np.float64)       # [3]
        kc = ks_all[prof_i].astype(np.float64)         # [3]

        # Full time axis for GT
        t_full = np.concatenate([[0.0], np.cumsum(dts)], axis=0)  # [T]
        start_t = float(t_full[start_i])

        # AR dt segment and starting state
        dt_seg = dts[start_i : start_i + K]                        # [K]
        t_pred = t_full[start_i + 1 : start_i + 1 + K]             # [K]
        y_start = ys[start_i]                                      # [3]
        y_true_seg = ys[start_i + 1 : start_i + 1 + K]             # [K,3] (unused for plot but helpful debugging)

        # Load params
        params_path = rdir / "models" / f"{model_type}_{cfg.ckpt}.pkl"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {params_path}")
        params = load_params(params_path)

        # JIT rollout (all float32)
        y_pred_all = rollout_from_point_jit(
            model_type=model_type,
            params=params,
            act_name=act_name,
            norm=norm,
            y_start=jnp.asarray(y_start, dtype=jnp.float32),
            dt_seq=jnp.asarray(dt_seg[:, None], dtype=jnp.float32),
            y0_const=jnp.asarray(y0c, dtype=jnp.float32),
            k_const=jnp.asarray(kc, dtype=jnp.float32),
        )
        y_pred = np.asarray(y_pred_all[1:, :])  # [K,3] predicted at t_pred

        # Plot entire GT + overlay preds
        plot_fulltraj_overlay(
            out_png=Path(""),  # unused here (we save once for the combined figure)
            gt_time=t_full,
            gt_y=ys,
            start_t=start_t,
            pred_time=t_pred,
            pred_y=y_pred,
            model_label=f"{model_type.upper()} ({cfg.ckpt}) | profile={prof_i} start_idx={start_i}",
            eps_plot=cfg.eps_plot,
            ax=ax,
        )

        ax.legend(loc="upper right", ncol=1)

    fig.suptitle(f"Full ground-truth trajectories with 100-step AR overlays (seed={cfg.seed})")
    fig.tight_layout()

    out_png = plots_dir / f"{cfg.out_png_name}_{cfg.ckpt}.png"
    fig.savefig(out_png)
    plt.close(fig)

    print("Saved:", str(out_png.resolve()))
    print("Selection:", json.dumps({k: {"profile": v[0], "start_idx": v[1]} for k, v in picks.items()}, indent=2))


if __name__ == "__main__":
    main(EVAL)
