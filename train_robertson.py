#!/usr/bin/env python3
"""
train_robertson.py

Train Robertson flow-map emulators (Residual MLP + DeepONet) in JAX.

Core problem:
- Robertson stiff ODE with parameterized reaction rates k=(k1,k2,k3).
- Each trajectory has:
    y0  (3)  ~ log-uniform components, normalized to the simplex (unless y0_fixed=True)
    k   (3)  ~ log-uniform (unless rates_fixed=True)
    dt  (T)  ~ log-uniform per step in [dt_min, dt_max]
- We solve trajectories with a stiff Diffrax solver (Kvaerno5 + PID controller).

Modeling:
- Both models are trained on one-jump targets (flow maps):
    y_{t+1} = F(y_t, dt_t, y0, k)
- Inputs and targets are in log10 space with z-score normalization (train statistics only).
- MLP predicts residual Δlog10(y) (normalized).
- DeepONet predicts log10(y_{t+1}) (normalized).

Training:
- One-jump pairs are sampled at random allowable step indices within each trajectory
  each epoch (for diversity), while preserving trajectory-level splits.
- Training batches and shuffles are device-side (no NumPy index gathers).
- Drop-last batches are enforced to keep shapes stable (avoid recompiles).

Optuna:
- Optional hyperparameter tuning controlled by Config.tuning.enabled.
- Tuning space is defined in a separate section (TUNING_SPACE).
- Each Optuna trial writes to a unique run directory (clobber-proof).

Outputs:
  <log_dir>/<run_name>/
    config.json
    config_used.json
    dataset_path.txt
    splits.npz
    norm_stats.npz
    models/{mlp,deeponet}_{best,last}.pkl
    logs/{mlp,deeponet}_metrics.csv
    (when tuning enabled) optuna_study_summary.json

Note:
- N-jump rollout fractional-error metrics are intentionally NOT computed during training.
  This file provides helper functions for evaluation to compute those metrics, but does
  not implement the evaluation script itself.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, List

import numpy as np
from tqdm import tqdm

import jax

# x64 is helpful for stiff ODE solves; training uses float32.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
import diffrax


# -----------------------------
# TUNING SPACE (edit here)
# -----------------------------
@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = False
    n_trials: int = 30
    study_name: str = "robertson"
    direction: str = "minimize"
    sampler_seed: int = 0
    objective: str = "val_mlp_mse"  # currently supported objective for Optuna


@dataclass(frozen=True)
class TuningSpace:
    # EXACT parameters varied when Optuna is enabled.
    activation_choices: Tuple[str, ...] = ("relu", "swish", "gelu", "tanh", "elu")
    mlp_width_min: int = 64
    mlp_width_max: int = 1024
    mlp_depth_min: int = 2
    mlp_depth_max: int = 6

    lr_min: float = 1e-4
    lr_max: float = 3e-3

    weight_decay_min: float = 1e-8
    weight_decay_max: float = 1e-4

    # Optional: tune batch size (must be power-of-two for best throughput).
    tune_batch_size: bool = True
    batch_size_choices: Tuple[int, ...] = (1024, 2048, 4096, 8192)


TUNING_SPACE = TuningSpace()


# -----------------------------
# CONFIG (edit here)
# -----------------------------
@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 0

    # Devices
    # - sim_device: where trajectories are generated (CPU recommended for stiff solve)
    # - train_device: where training runs ("auto" prefers GPU)
    sim_device: str = "cpu"   # "cpu" recommended for stiff solve
    train_device: str = "auto"  # "auto" | "cpu" | "gpu"

    # If True, host the full (split) dataset arrays on the training device (GPU on A100).
    dataset_on_device: bool = True

    # Run/artifacts
    log_dir: str = "runs"
    run_name: str = "baseline"

    # Dataset cache
    dataset_dir: str = "robertson_data_cache"
    use_cache: bool = True

    # Dataset size / horizons
    n_trajectories: int = 10_000
    n_steps: int = 1000  # trajectory length in dt steps

    # dt sampling (per-step log-uniform)
    dt_min: float = 1e-1
    dt_max: float = 1e1

    # Initial condition sampling:
    # If y0_fixed=True -> always [1,0,0]
    # Else sample positive components log-uniform then normalize to simplex.
    y0_fixed: bool = False  # default per instructions
    y0_log10_min: float = -12.0
    y0_log10_max: float = 0.0

    # Reaction rate sampling:
    # If rates_fixed=True -> canonical Robertson rates.
    # Else sample log-uniform per component in configured ranges.
    rates_fixed: bool = False
    k1_log10_min: float = -3.0
    k1_log10_max: float = 0.0
    k2_log10_min: float = 2.0
    k2_log10_max: float = 6.0
    k3_log10_min: float = 5.0
    k3_log10_max: float = 9.0

    # Canonical rates (used when rates_fixed=True)
    k1_canonical: float = 0.04
    k2_canonical: float = 1.0e4
    k3_canonical: float = 3.0e7

    # ODE solver defaults (stiff)
    rtol: float = 1e-7
    atol: float = 1e-10
    solver_dt0: float = 1e-12
    solver_max_steps: int = 5_000_000

    # Splits (trajectory-level)
    frac_train: float = 0.8
    frac_val: float = 0.1  # test remainder

    # Normalization
    eps: float = 1e-30       # for log10(y + eps)
    min_std: float = 1e-12   # avoid divide-by-zero

    # Training
    epochs: int = 100  # per instructions
    batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    warmup_frac: float = 0.05
    lr_min_frac: float = 0.05

    # For each epoch, sample this many one-jump transitions per trajectory.
    # (1 means one random step per trajectory per epoch.)
    samples_per_trajectory_per_epoch: int = 1

    # MLP (residual in log space):
    # Input dim = [logy_t(3), logdt(1), logy0(3), logk(3)] = 10 (after z-score).
    mlp_depth: int = 3
    mlp_width: int = 256

    # DeepONet:
    # branch input: [logy0(3), logk(3)] => 6
    # trunk input:  [logy_t(3), logdt(1)] => 4
    deeponet_depth: int = 3
    deeponet_branch_width: int = 192
    deeponet_trunk_width: int = 192
    deeponet_feature_dim: int = 256

    # Activation (also used for hyperparam sweeps)
    activation: str = "swish"  # "relu","swish","gelu","tanh","elu"

    # Rough parameter parity (DeepONet widths/feature_dim tuned around current values)
    match_params: bool = True

    # Optuna tuning config
    tuning: TuningConfig = TuningConfig()


CONFIG = Config()


# -----------------------------
# Small utilities
# -----------------------------
def _prefer_gpu_device() -> jax.Device:
    gpus = jax.devices("gpu")
    if gpus:
        return gpus[0]
    return jax.devices("cpu")[0]


def _select_device(kind: str) -> jax.Device:
    kind_l = kind.lower()
    if kind_l == "auto":
        return _prefer_gpu_device()
    if kind_l == "cpu":
        return jax.devices("cpu")[0]
    if kind_l == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("train_device='gpu' requested but no GPU devices found.")
        return gpus[0]
    raise ValueError(f"Unknown device kind: {kind}")


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, size: Tuple[int, ...]) -> np.ndarray:
    """Sample log-uniform in base10: 10**U where U~Uniform(lo,hi)."""
    u = rng.uniform(lo, hi, size=size)
    return np.power(10.0, u)


def _simplex_from_log_uniform(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    """Sample positive vector via log-uniform components then normalize to sum 1."""
    raw = _log_uniform(rng, lo, hi, size=(n, 3)).astype(np.float64)
    s = np.sum(raw, axis=1, keepdims=True)
    return (raw / np.clip(s, 1e-300, np.inf)).astype(np.float64)


def _sample_y0s(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.y0_fixed:
        y0 = np.zeros((cfg.n_trajectories, 3), dtype=np.float64)
        y0[:, 0] = 1.0
        return y0
    return _simplex_from_log_uniform(rng, cfg.n_trajectories, cfg.y0_log10_min, cfg.y0_log10_max)


def _sample_rates(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.rates_fixed:
        k = np.array([cfg.k1_canonical, cfg.k2_canonical, cfg.k3_canonical], dtype=np.float64)
        return np.broadcast_to(k[None, :], (cfg.n_trajectories, 3)).copy()
    k1 = _log_uniform(rng, cfg.k1_log10_min, cfg.k1_log10_max, size=(cfg.n_trajectories, 1))
    k2 = _log_uniform(rng, cfg.k2_log10_min, cfg.k2_log10_max, size=(cfg.n_trajectories, 1))
    k3 = _log_uniform(rng, cfg.k3_log10_min, cfg.k3_log10_max, size=(cfg.n_trajectories, 1))
    return np.concatenate([k1, k2, k3], axis=1).astype(np.float64)


def _sample_dt_sequences(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    lo = math.log10(cfg.dt_min)
    hi = math.log10(cfg.dt_max)
    return _log_uniform(rng, lo, hi, size=(cfg.n_trajectories, cfg.n_steps)).astype(np.float64)


def _run_dir(cfg: Config) -> Path:
    d = Path(cfg.log_dir) / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "models").mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    return d


def _save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(obj), indent=2, sort_keys=True))


def _save_params(path: Path, params: Any) -> None:
    params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(path, "wb") as f:
        pickle.dump(params_np, f, protocol=pickle.HIGHEST_PROTOCOL)


def _unique_run_name(base: str, root: Path) -> str:
    """
    Return a clobber-proof run name under root. If root/base exists, append _r<k>.
    """
    cand = base
    k = 0
    while (root / cand).exists():
        k += 1
        cand = f"{base}_r{k}"
    return cand


# -----------------------------
# Robertson ODE (parameterized)
# -----------------------------
def robertson_rhs(t: jnp.ndarray, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
    """
    args = (k1,k2,k3), all positive.
    Classic Robertson:
      dy1 = -k1 y1 + k2 y2 y3
      dy2 =  k1 y1 - k2 y2 y3 - k3 y2^2
      dy3 =  k3 y2^2
    """
    k1, k2, k3 = args[0], args[1], args[2]
    y1, y2, y3 = y[0], y[1], y[2]
    dy1 = -k1 * y1 + k2 * y2 * y3
    dy2 = k1 * y1 - k2 * y2 * y3 - k3 * (y2 * y2)
    dy3 = k3 * (y2 * y2)
    return jnp.array([dy1, dy2, dy3], dtype=y.dtype)


@partial(jax.jit, static_argnames=("max_steps",))
def _simulate_one_trajectory(
    y0: jnp.ndarray,
    k: jnp.ndarray,
    dts: jnp.ndarray,
    rtol: float,
    atol: float,
    dt0: float,
    max_steps: int,
) -> jnp.ndarray:
    ts = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), jnp.cumsum(dts)], axis=0)

    term = diffrax.ODETerm(robertson_rhs)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=y0,
        args=k,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=controller,
        max_steps=max_steps,  # MUST be static
    )

    ys = sol.ys
    ys = jnp.clip(ys, 0.0, jnp.inf)
    ys = ys / jnp.clip(jnp.sum(ys, axis=-1, keepdims=True), 1e-30, jnp.inf)
    return ys


# -----------------------------
# Dataset caching
# -----------------------------
def _dataset_cache_paths(cfg: Config) -> Dict[str, Path]:
    d = Path(cfg.dataset_dir)
    d.mkdir(parents=True, exist_ok=True)
    tag = (
        f"traj{cfg.n_trajectories}_steps{cfg.n_steps}_dt"
        f"{cfg.dt_min:g}-{cfg.dt_max:g}_seed{cfg.seed}_"
        f"y0{'fixed' if cfg.y0_fixed else 'rand'}_"
        f"k{'fixed' if cfg.rates_fixed else 'rand'}"
    )
    return {"npz": d / f"robertson_{tag}.npz", "meta": d / f"robertson_{tag}_meta.txt"}


def generate_or_load_dataset(cfg: Config) -> Dict[str, np.ndarray]:
    """
    Returns:
      y0s: [N,3]
      ks:  [N,3]
      dts: [N,n_steps]
      ys:  [N,n_steps+1,3]
    """
    paths = _dataset_cache_paths(cfg)
    if cfg.use_cache and paths["npz"].exists():
        z = np.load(paths["npz"], allow_pickle=False)
        return {k: z[k] for k in z.files}

    rng = np.random.default_rng(cfg.seed)
    y0s = _sample_y0s(cfg, rng)
    ks = _sample_rates(cfg, rng)
    dts = _sample_dt_sequences(cfg, rng)

    sim_dev = _select_device(cfg.sim_device)
    ys_all: List[np.ndarray] = []

    with jax.default_device(sim_dev):
        for i in tqdm(range(cfg.n_trajectories), desc=f"Simulating trajectories on {sim_dev.platform}"):
            y0 = jnp.asarray(y0s[i], dtype=jnp.float64)
            k = jnp.asarray(ks[i], dtype=jnp.float64)
            dt_seq = jnp.asarray(dts[i], dtype=jnp.float64)

            ys = _simulate_one_trajectory(
                y0=y0,
                k=k,
                dts=dt_seq,
                rtol=float(cfg.rtol),
                atol=float(cfg.atol),
                dt0=float(cfg.solver_dt0),
                max_steps=int(cfg.solver_max_steps),
            )
            ys_all.append(np.asarray(ys))

    ys_all_np = np.stack(ys_all, axis=0).astype(np.float64)

    out = {"y0s": y0s, "ks": ks, "dts": dts, "ys": ys_all_np}
    np.savez_compressed(paths["npz"], **out)

    paths["meta"].write_text(
        f"Generated: {time.ctime()}\n"
        f"n_trajectories={cfg.n_trajectories}, n_steps={cfg.n_steps}\n"
        f"dt_range=[{cfg.dt_min:g}, {cfg.dt_max:g}] (log-uniform per step)\n"
        f"y0_fixed={cfg.y0_fixed}, rates_fixed={cfg.rates_fixed}\n"
        f"rtol={cfg.rtol}, atol={cfg.atol}, dt0={cfg.solver_dt0}, max_steps={cfg.solver_max_steps}\n"
        f"seed={cfg.seed}\n"
    )
    return out


# -----------------------------
# Splits + normalization (train stats only)
# -----------------------------
def build_splits(cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(cfg.n_trajectories)
    rng.shuffle(idx)
    n_train = int(round(cfg.frac_train * cfg.n_trajectories))
    n_val = int(round(cfg.frac_val * cfg.n_trajectories))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _mean_std_np(a: np.ndarray, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
    m = a.mean(axis=0)
    s = a.std(axis=0)
    s = np.where(s < min_std, min_std, s)
    return m, s


def compute_norm_stats(cfg: Config, data: Dict[str, np.ndarray], train_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Train-only stats. All logs are log10.

    Inputs:
      - log10(y_t + eps)   (3)
      - log10(dt)          (1)
      - log10(y0 + eps)    (3)
      - log10(k)           (3)

    Outputs:
      - DeepONet target: log10(y_{t+1} + eps)
      - MLP target:      Δlog10(y) = log10(y_{t+1}+eps) - log10(y_t+eps)
    """
    eps = float(cfg.eps)

    ys = data["ys"][train_idx]   # [Ntr, T+1, 3]
    dts = data["dts"][train_idx] # [Ntr, T]
    y0s = data["y0s"][train_idx] # [Ntr, 3]
    ks = data["ks"][train_idx]   # [Ntr, 3]

    logy = np.log10(ys + eps).astype(np.float64)          # [Ntr, T+1, 3]
    logdt = np.log10(dts).astype(np.float64)[..., None]   # [Ntr, T, 1]
    logy0 = np.log10(y0s + eps).astype(np.float64)        # [Ntr, 3]
    logk = np.log10(ks).astype(np.float64)                # [Ntr, 3]

    logy_t = logy[:, :-1, :].reshape(-1, 3)
    logy_tp1 = logy[:, 1:, :].reshape(-1, 3)
    logdt_f = logdt.reshape(-1, 1)

    dlog = (logy_tp1 - logy_t)  # [Ntr*T, 3]

    state_mean, state_std = _mean_std_np(logy_t, cfg.min_std)
    dt_mean, dt_std = _mean_std_np(logdt_f, cfg.min_std)
    y0_mean, y0_std = _mean_std_np(logy0, cfg.min_std)
    k_mean, k_std = _mean_std_np(logk, cfg.min_std)

    out_mean_deep, out_std_deep = _mean_std_np(logy_tp1, cfg.min_std)
    out_mean_mlp, out_std_mlp = _mean_std_np(dlog, cfg.min_std)

    return {
        "eps": np.array(cfg.eps, dtype=np.float64),
        "min_std": np.array(cfg.min_std, dtype=np.float64),
        "state_mean": state_mean,
        "state_std": state_std,
        "dt_mean": dt_mean,
        "dt_std": dt_std,
        "y0_mean": y0_mean,
        "y0_std": y0_std,
        "k_mean": k_mean,
        "k_std": k_std,
        "out_mean_deeponet": out_mean_deep,
        "out_std_deeponet": out_std_deep,
        "out_mean_mlp": out_mean_mlp,
        "out_std_mlp": out_std_mlp,
    }


# -----------------------------
# Minimal JAX models (no flax/equinox)
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


def init_dense(key: jax.Array, fan_in: int, fan_out: int) -> Dict[str, jax.Array]:
    w_key, _ = jax.random.split(key)
    scale = 1.0 / math.sqrt(max(1, fan_in))
    w = scale * jax.random.normal(w_key, (fan_in, fan_out), dtype=jnp.float32)
    b = jnp.zeros((fan_out,), dtype=jnp.float32)
    return {"W": w, "b": b}


def mlp_init(key: jax.Array, in_dim: int, out_dim: int, depth: int, width: int) -> List[Dict[str, jax.Array]]:
    keys = jax.random.split(key, depth + 1)
    layers: List[Dict[str, jax.Array]] = []
    d = in_dim
    for i in range(depth):
        layers.append(init_dense(keys[i], d, width))
        d = width
    layers.append(init_dense(keys[-1], d, out_dim))
    return layers


def mlp_apply(params: List[Dict[str, jax.Array]], x: jnp.ndarray, act: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = act(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def deeponet_init(
    key: jax.Array,
    branch_in: int,
    trunk_in: int,
    feature_dim: int,
    depth: int,
    branch_width: int,
    trunk_width: int,
    out_dim: int,
) -> Dict[str, Any]:
    k1, k2, k3 = jax.random.split(key, 3)
    branch = mlp_init(k1, branch_in, feature_dim, depth, branch_width)
    trunk = mlp_init(k2, trunk_in, feature_dim, depth, trunk_width)
    head = init_dense(k3, feature_dim, out_dim)
    return {"branch": branch, "trunk": trunk, "head": head}


def deeponet_apply(
    params: Dict[str, Any],
    x_branch: jnp.ndarray,
    x_trunk: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    b = mlp_apply(params["branch"], x_branch, act)
    t = mlp_apply(params["trunk"], x_trunk, act)
    h = b * t
    head = params["head"]
    return h @ head["W"] + head["b"]


def init_model(cfg: Config, model_type: str, key: jax.Array) -> Any:
    if model_type == "mlp":
        # Residual flow-map: predicts Δlog10(y) in normalized space.
        return mlp_init(key, in_dim=10, out_dim=3, depth=cfg.mlp_depth, width=cfg.mlp_width)
    if model_type == "deeponet":
        return deeponet_init(
            key,
            branch_in=6,
            trunk_in=4,
            feature_dim=cfg.deeponet_feature_dim,
            depth=cfg.deeponet_depth,
            branch_width=cfg.deeponet_branch_width,
            trunk_width=cfg.deeponet_trunk_width,
            out_dim=3,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def tree_num_params(pytree: Any) -> int:
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    return int(sum(int(np.prod(x.shape)) for x in leaves))


def match_deeponet_to_mlp(cfg: Config) -> Config:
    """
    Rough parameter matching for comparability:
    Adjust DeepONet (branch_width, trunk_width, feature_dim) around current values
    to get within a small relative gap to MLP param count.
    """
    key = jax.random.PRNGKey(0)
    mlp_p = init_model(cfg, "mlp", key)
    target = tree_num_params(mlp_p)

    best_cfg = cfg
    best_rel = float("inf")

    # Small grid around current widths.
    for bw in [max(32, cfg.deeponet_branch_width // 2), cfg.deeponet_branch_width, cfg.deeponet_branch_width * 2]:
        for tw in [max(32, cfg.deeponet_trunk_width // 2), cfg.deeponet_trunk_width, cfg.deeponet_trunk_width * 2]:
            for fd in [max(32, cfg.deeponet_feature_dim // 2), cfg.deeponet_feature_dim, cfg.deeponet_feature_dim * 2]:
                cand = replace(cfg, deeponet_branch_width=int(bw), deeponet_trunk_width=int(tw), deeponet_feature_dim=int(fd))
                dp_p = init_model(cand, "deeponet", key)
                n = tree_num_params(dp_p)
                rel = abs(n - target) / max(1, target)
                if rel < best_rel:
                    best_rel = rel
                    best_cfg = cand
    return best_cfg


# -----------------------------
# Optimizer (cosine + warmup)
# -----------------------------
def make_schedule(cfg: Config, total_steps: int) -> optax.Schedule:
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    lr0 = float(cfg.lr)
    lr_min = float(cfg.lr) * float(cfg.lr_min_frac)

    warmup = optax.linear_schedule(init_value=0.0, end_value=lr0, transition_steps=warmup_steps)
    cosine = optax.cosine_decay_schedule(init_value=lr0, decay_steps=decay_steps, alpha=lr_min / lr0)
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def make_optimizer(cfg: Config, schedule: optax.Schedule) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )


# -----------------------------
# Device-side normalized batch builder (flow maps)
# -----------------------------
@dataclass(frozen=True)
class NormStatsJax:
    eps: float
    min_std: float
    state_mean: jnp.ndarray
    state_std: jnp.ndarray
    dt_mean: jnp.ndarray
    dt_std: jnp.ndarray
    y0_mean: jnp.ndarray
    y0_std: jnp.ndarray
    k_mean: jnp.ndarray
    k_std: jnp.ndarray
    out_mean_deeponet: jnp.ndarray
    out_std_deeponet: jnp.ndarray
    out_mean_mlp: jnp.ndarray
    out_std_mlp: jnp.ndarray


def _norm_to_jax(norm_np: Dict[str, np.ndarray], device: jax.Device) -> NormStatsJax:
    eps = float(norm_np["eps"])
    min_std = float(norm_np["min_std"])

    def _put(x: np.ndarray) -> jnp.ndarray:
        return jax.device_put(jnp.asarray(x, dtype=jnp.float32), device=device)

    return NormStatsJax(
        eps=eps,
        min_std=min_std,
        state_mean=_put(norm_np["state_mean"]),
        state_std=_put(norm_np["state_std"]),
        dt_mean=_put(norm_np["dt_mean"]),
        dt_std=_put(norm_np["dt_std"]),
        y0_mean=_put(norm_np["y0_mean"]),
        y0_std=_put(norm_np["y0_std"]),
        k_mean=_put(norm_np["k_mean"]),
        k_std=_put(norm_np["k_std"]),
        out_mean_deeponet=_put(norm_np["out_mean_deeponet"]),
        out_std_deeponet=_put(norm_np["out_std_deeponet"]),
        out_mean_mlp=_put(norm_np["out_mean_mlp"]),
        out_std_mlp=_put(norm_np["out_std_mlp"]),
    )


@dataclass
class SplitCache:
    # logy: [N, T+1, 3]
    logy: jnp.ndarray
    # logdt: [N, T]
    logdt: jnp.ndarray
    # logy0: [N, 3]
    logy0: jnp.ndarray
    # logk: [N, 3]
    logk: jnp.ndarray


def _prepare_split_cache(
    cfg: Config,
    data: Dict[str, np.ndarray],
    split_idx: np.ndarray,
    device: jax.Device,
    *,
    place_on_device: bool,
) -> SplitCache:
    eps = float(cfg.eps)
    ys = data["ys"][split_idx].astype(np.float32)    # [N, T+1, 3]
    dts = data["dts"][split_idx].astype(np.float32)  # [N, T]
    y0s = data["y0s"][split_idx].astype(np.float32)  # [N, 3]
    ks = data["ks"][split_idx].astype(np.float32)    # [N, 3]

    # Precompute logs once (training is float32).
    logy = np.log10(ys + eps).astype(np.float32)
    logdt = np.log10(dts).astype(np.float32)
    logy0 = np.log10(y0s + eps).astype(np.float32)
    logk = np.log10(ks).astype(np.float32)

    if place_on_device:
        logy_j = jax.device_put(jnp.asarray(logy), device=device)
        logdt_j = jax.device_put(jnp.asarray(logdt), device=device)
        logy0_j = jax.device_put(jnp.asarray(logy0), device=device)
        logk_j = jax.device_put(jnp.asarray(logk), device=device)
    else:
        # Keep on CPU device. (Transfers happen implicitly if you train on GPU.)
        cpu = jax.devices("cpu")[0]
        logy_j = jax.device_put(jnp.asarray(logy), device=cpu)
        logdt_j = jax.device_put(jnp.asarray(logdt), device=cpu)
        logy0_j = jax.device_put(jnp.asarray(logy0), device=cpu)
        logk_j = jax.device_put(jnp.asarray(logk), device=cpu)

    return SplitCache(logy=logy_j, logdt=logdt_j, logy0=logy0_j, logk=logk_j)


def _zscore(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, min_std: float) -> jnp.ndarray:
    return (x - mean) / jnp.clip(std, min_std, jnp.inf)


def _make_batch_mlp(cache: SplitCache, norm: NormStatsJax, traj_idx: jnp.ndarray, step_idx: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    # Gather: y_t, y_{t+1}, dt, y0, k in log space.
    logy_t = cache.logy[traj_idx, step_idx, :]            # [B,3]
    logy_tp1 = cache.logy[traj_idx, step_idx + 1, :]      # [B,3]
    logdt = cache.logdt[traj_idx, step_idx]               # [B]
    logy0 = cache.logy0[traj_idx, :]                      # [B,3]
    logk = cache.logk[traj_idx, :]                        # [B,3]

    x_state = _zscore(logy_t, norm.state_mean, norm.state_std, norm.min_std)
    x_dt = _zscore(logdt[:, None], norm.dt_mean, norm.dt_std, norm.min_std)
    x_y0 = _zscore(logy0, norm.y0_mean, norm.y0_std, norm.min_std)
    x_k = _zscore(logk, norm.k_mean, norm.k_std, norm.min_std)

    dlog = (logy_tp1 - logy_t)
    y_out = _zscore(dlog, norm.out_mean_mlp, norm.out_std_mlp, norm.min_std)
    x = jnp.concatenate([x_state, x_dt, x_y0, x_k], axis=-1)  # [B,10]
    return {"x": x, "y_out": y_out}


def _make_batch_deeponet(cache: SplitCache, norm: NormStatsJax, traj_idx: jnp.ndarray, step_idx: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    logy_t = cache.logy[traj_idx, step_idx, :]            # [B,3]
    logy_tp1 = cache.logy[traj_idx, step_idx + 1, :]      # [B,3]
    logdt = cache.logdt[traj_idx, step_idx]               # [B]
    logy0 = cache.logy0[traj_idx, :]                      # [B,3]
    logk = cache.logk[traj_idx, :]                        # [B,3]

    x_state = _zscore(logy_t, norm.state_mean, norm.state_std, norm.min_std)
    x_dt = _zscore(logdt[:, None], norm.dt_mean, norm.dt_std, norm.min_std)
    x_y0 = _zscore(logy0, norm.y0_mean, norm.y0_std, norm.min_std)
    x_k = _zscore(logk, norm.k_mean, norm.k_std, norm.min_std)

    y_out = _zscore(logy_tp1, norm.out_mean_deeponet, norm.out_std_deeponet, norm.min_std)
    x_branch = jnp.concatenate([x_y0, x_k], axis=-1)  # [B,6]
    x_trunk = jnp.concatenate([x_state, x_dt], axis=-1)  # [B,4]
    return {"x_branch": x_branch, "x_trunk": x_trunk, "y_out": y_out}


# -----------------------------
# Training (one-jump flow maps)
# -----------------------------
def train_one_model(
    cfg: Config,
    model_type: str,
    train_cache: SplitCache,
    val_cache: SplitCache,
    norm: NormStatsJax,
    *,
    csv_path: Path,
    key: jax.Array,
) -> Tuple[Any, Any]:
    """
    Returns (best_params, last_params) according to best validation MSE in normalized output space.
    """
    act = get_activation(cfg.activation)

    n_train_traj = int(train_cache.logy.shape[0])
    n_val_traj = int(val_cache.logy.shape[0])
    n_steps = int(train_cache.logdt.shape[1])

    if n_steps != int(val_cache.logdt.shape[1]):
        raise ValueError("Train/val n_steps mismatch (unexpected).")

    # Define the number of samples per epoch by sampling random step indices per trajectory.
    spp = int(cfg.samples_per_trajectory_per_epoch)
    if spp < 1:
        raise ValueError("samples_per_trajectory_per_epoch must be >= 1")

    n_train_samples = n_train_traj * spp
    steps_per_epoch = n_train_samples // int(cfg.batch_size)  # drop last
    if steps_per_epoch < 1:
        raise ValueError("Batch size too large: no full train batches. Reduce batch_size or increase n_trajectories.")

    total_steps = steps_per_epoch * int(cfg.epochs)

    schedule = make_schedule(cfg, total_steps)
    opt = make_optimizer(cfg, schedule)

    params = init_model(cfg, model_type, key)
    opt_state = opt.init(params)

    if model_type == "mlp":

        @jax.jit
        def _forward(p: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            return mlp_apply(p, batch["x"], act)

        make_batch = _make_batch_mlp

    elif model_type == "deeponet":

        @jax.jit
        def _forward(p: Any, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            return deeponet_apply(p, batch["x_branch"], batch["x_trunk"], act)

        make_batch = _make_batch_deeponet
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    @jax.jit
    def train_step(p: Any, s: Any, traj_idx: jnp.ndarray, step_idx: jnp.ndarray) -> Tuple[Any, Any, jnp.ndarray]:
        batch = make_batch(train_cache, norm, traj_idx, step_idx)

        def loss_fn(pp: Any) -> jnp.ndarray:
            pred = _forward(pp, batch)
            return jnp.mean((pred - batch["y_out"]) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s2 = opt.update(grads, s, p)
        p2 = optax.apply_updates(p, updates)
        return p2, s2, loss

    @jax.jit
    def val_mse(p: Any, traj_idx: jnp.ndarray, step_idx: jnp.ndarray) -> jnp.ndarray:
        batch = make_batch(val_cache, norm, traj_idx, step_idx)
        pred = _forward(p, batch)
        return jnp.mean((pred - batch["y_out"]) ** 2)

    def _build_epoch_indices(rng_key: jax.Array, n_traj: int, *, spp_local: int) -> Tuple[jax.Array, jax.Array]:
        """
        Create per-epoch (traj_idx, step_idx) arrays (on device):
        - traj_idx: permutation of trajectories, repeated spp_local times
        - step_idx: random step index for each sample, in [0, n_steps-1]
        """
        k1, k2 = jax.random.split(rng_key, 2)
        traj = jax.random.permutation(k1, n_traj)  # [n_traj]
        if spp_local == 1:
            traj_rep = traj
        else:
            traj_rep = jnp.repeat(traj, spp_local, axis=0)  # [n_traj*spp_local]
        step = jax.random.randint(k2, (traj_rep.shape[0],), 0, n_steps)  # [n_samples]
        return traj_rep, step

    # Precompute a fixed validation index set for stable val curves.
    # Use one random step per val trajectory by default; still drop-last in batching.
    val_spp = 1
    k_val = jax.random.fold_in(key, 999)
    val_traj_rep, val_step = _build_epoch_indices(k_val, n_val_traj, spp_local=val_spp)
    n_val_samples = int(val_traj_rep.shape[0])
    n_val_full = (n_val_samples // int(cfg.batch_size)) * int(cfg.batch_size)
    val_traj_rep = val_traj_rep[:n_val_full]
    val_step = val_step[:n_val_full]
    val_traj_batches = val_traj_rep.reshape((-1, int(cfg.batch_size)))
    val_step_batches = val_step.reshape((-1, int(cfg.batch_size)))

    @jax.jit
    def _val_epoch(p: Any) -> jnp.ndarray:
        def body(carry: jnp.ndarray, xs: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
            traj_b, step_b = xs
            mse_b = val_mse(p, traj_b, step_b)
            return carry + mse_b, None

        total, _ = jax.lax.scan(body, jnp.array(0.0, dtype=jnp.float32), (val_traj_batches, val_step_batches))
        return total / jnp.maximum(1.0, float(val_traj_batches.shape[0]))

    @jax.jit
    def _train_epoch(rng_key: jax.Array, p: Any, s: Any) -> Tuple[jax.Array, Any, Any, jnp.ndarray]:
        traj_rep, step = _build_epoch_indices(rng_key, n_train_traj, spp_local=spp)

        n_full = (traj_rep.shape[0] // int(cfg.batch_size)) * int(cfg.batch_size)
        traj_rep = traj_rep[:n_full]
        step = step[:n_full]

        traj_batches = traj_rep.reshape((-1, int(cfg.batch_size)))
        step_batches = step.reshape((-1, int(cfg.batch_size)))

        def body(carry: Tuple[Any, Any, jnp.ndarray], xs: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[Tuple[Any, Any, jnp.ndarray], None]:
            p0, s0, loss_acc = carry
            traj_b, step_b = xs
            p1, s1, loss_b = train_step(p0, s0, traj_b, step_b)
            return (p1, s1, loss_acc + loss_b), None

        (p_out, s_out, loss_sum), _ = jax.lax.scan(
            body,
            (p, s, jnp.array(0.0, dtype=jnp.float32)),
            (traj_batches, step_batches),
        )
        mean_loss = loss_sum / jnp.maximum(1.0, float(traj_batches.shape[0]))
        return rng_key, p_out, s_out, mean_loss

    # CSV log header
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "global_step", "lr", "train_mse", "val_mse"])

    best_val: Optional[float] = None
    best_params: Optional[Any] = None

    global_step = 0
    epoch_key = jax.random.fold_in(key, 12345)

    for epoch in tqdm(range(int(cfg.epochs)), desc=f"Training {model_type}"):
        epoch_key, subkey = jax.random.split(epoch_key, 2)

        # Train epoch (device-side)
        _, params, opt_state, train_mse_j = _train_epoch(subkey, params, opt_state)
        train_mse = float(train_mse_j)

        # Validation epoch (fixed indices)
        val_mse_j = _val_epoch(params)
        val_mse_f = float(val_mse_j)

        global_step += steps_per_epoch
        lr_now = float(schedule(max(0, global_step - 1)))

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, global_step, lr_now, train_mse, val_mse_f])

        if (best_val is None) or (val_mse_f < best_val):
            best_val = val_mse_f
            best_params = params

    assert best_params is not None
    return best_params, params


# -----------------------------
# Main runner
# -----------------------------
def run_train(cfg: Config, *, return_objective: Optional[str] = None) -> Optional[float]:
    """
    Train both models and save artifacts. If return_objective is provided,
    returns a scalar suitable for hyperparam search.

    Supported return_objective:
      - "val_mlp_mse": final logged val MSE for MLP (normalized output space)
    """
    rdir = _run_dir(cfg)
    _save_json(rdir / "config.json", asdict(cfg))

    # Dataset
    data = generate_or_load_dataset(cfg)
    cache_path = _dataset_cache_paths(cfg)["npz"].resolve()
    (rdir / "dataset_path.txt").write_text(str(cache_path))

    # Splits
    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx, test_idx = build_splits(cfg, rng)
    np.savez_compressed(rdir / "splits.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # Norm stats (train-only)
    norm_np = compute_norm_stats(cfg, data, train_idx)
    np.savez_compressed(rdir / "norm_stats.npz", **norm_np)

    # Optional parameter parity adjustment
    cfg_used = match_deeponet_to_mlp(cfg) if cfg.match_params else cfg
    _save_json(rdir / "config_used.json", asdict(cfg_used))

    # Training device and placement
    train_dev = _select_device(cfg_used.train_device)
    norm_j = _norm_to_jax(norm_np, device=train_dev)

    # Split caches (precompute logs once; optionally place on training device).
    place = bool(cfg_used.dataset_on_device)
    train_cache = _prepare_split_cache(cfg_used, data, train_idx, device=train_dev, place_on_device=place)
    val_cache = _prepare_split_cache(cfg_used, data, val_idx, device=train_dev, place_on_device=place)

    # Train both models
    key0 = jax.random.PRNGKey(cfg_used.seed)
    k_mlp, k_dp = jax.random.split(key0, 2)

    mlp_best, mlp_last = train_one_model(
        cfg_used,
        "mlp",
        train_cache,
        val_cache,
        norm_j,
        csv_path=rdir / "logs" / "mlp_metrics.csv",
        key=k_mlp,
    )
    _save_params(rdir / "models" / "mlp_best.pkl", mlp_best)
    _save_params(rdir / "models" / "mlp_last.pkl", mlp_last)

    dp_best, dp_last = train_one_model(
        cfg_used,
        "deeponet",
        train_cache,
        val_cache,
        norm_j,
        csv_path=rdir / "logs" / "deeponet_metrics.csv",
        key=k_dp,
    )
    _save_params(rdir / "models" / "deeponet_best.pkl", dp_best)
    _save_params(rdir / "models" / "deeponet_last.pkl", dp_last)

    if return_objective is not None:
        if return_objective == "val_mlp_mse":
            rows = (rdir / "logs" / "mlp_metrics.csv").read_text().strip().splitlines()
            if len(rows) < 2:
                return float("inf")
            last = rows[-1].split(",")
            return float(last[-1])
        raise ValueError(f"Unknown return_objective: {return_objective}")

    print("\nSaved run to:", str(rdir.resolve()))
    print("Dataset:", str(cache_path))
    print("Models:", str((rdir / "models").resolve()))
    print("Logs:", str((rdir / "logs").resolve()))
    print("Train device:", train_dev)
    print("Dataset on device:", bool(cfg_used.dataset_on_device))
    return None


# -----------------------------
# Optuna integration (enabled by config knob)
# -----------------------------
def _apply_trial_to_cfg(base_cfg: Config, trial: Any) -> Config:
    """
    Apply ONLY parameters defined in TUNING_SPACE.
    """
    act = trial.suggest_categorical("activation", list(TUNING_SPACE.activation_choices))
    mlp_w = trial.suggest_int("mlp_width", TUNING_SPACE.mlp_width_min, TUNING_SPACE.mlp_width_max, log=True)
    mlp_d = trial.suggest_int("mlp_depth", TUNING_SPACE.mlp_depth_min, TUNING_SPACE.mlp_depth_max)

    lr = trial.suggest_float("lr", TUNING_SPACE.lr_min, TUNING_SPACE.lr_max, log=True)
    wd = trial.suggest_float("weight_decay", TUNING_SPACE.weight_decay_min, TUNING_SPACE.weight_decay_max, log=True)

    cfg = replace(base_cfg, activation=str(act), mlp_width=int(mlp_w), mlp_depth=int(mlp_d), lr=float(lr), weight_decay=float(wd))

    if TUNING_SPACE.tune_batch_size:
        bs = trial.suggest_categorical("batch_size", list(TUNING_SPACE.batch_size_choices))
        cfg = replace(cfg, batch_size=int(bs))

    if cfg.match_params:
        cfg = match_deeponet_to_mlp(cfg)
    return cfg


def run_optuna_tuning(base_cfg: Config) -> Config:
    """
    Run Optuna study, writing each trial to a unique run directory.
    Returns the best Config (with best hyperparameters applied).
    """
    try:
        import optuna  # local import; still required when tuning is enabled
    except Exception as e:
        raise RuntimeError("Optuna is required when Config.tuning.enabled=True but could not be imported.") from e

    # Put all tuning artifacts under a stable directory keyed by study_name and timestamp.
    ts = time.strftime("%Y%m%d_%H%M%S")
    tuning_root = Path(base_cfg.log_dir) / f"optuna_{base_cfg.tuning.study_name}_{ts}"
    tuning_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: Any) -> float:
        cfg_trial = _apply_trial_to_cfg(base_cfg, trial)

        # Clobber-proof per-trial run name under tuning_root.
        run_name = f"{base_cfg.run_name}_trial{trial.number:04d}"
        run_name = _unique_run_name(run_name, tuning_root)
        cfg_trial = replace(cfg_trial, log_dir=str(tuning_root), run_name=run_name)

        val = run_train(cfg_trial, return_objective=base_cfg.tuning.objective)
        assert val is not None
        return float(val)

    sampler = optuna.samplers.TPESampler(seed=int(base_cfg.tuning.sampler_seed))
    study = optuna.create_study(direction=str(base_cfg.tuning.direction), study_name=str(base_cfg.tuning.study_name), sampler=sampler)
    study.optimize(objective, n_trials=int(base_cfg.tuning.n_trials))

    # Persist a small summary
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }
    _save_json(tuning_root / "optuna_study_summary.json", summary)

    # Build best config and run a final "best" training job.
    best_cfg = _apply_trial_to_cfg(base_cfg, study.best_trial)
    best_run_name = _unique_run_name(f"{base_cfg.run_name}_best", tuning_root)
    best_cfg = replace(best_cfg, log_dir=str(tuning_root), run_name=best_run_name)
    run_train(best_cfg)

    return best_cfg


# -----------------------------
# Evaluation helpers (N-jump fractional errors)
# -----------------------------
def _denorm_mlp_dlog(y_norm: jnp.ndarray, norm: NormStatsJax) -> jnp.ndarray:
    # y_norm: normalized Δlog10(y) [B,3] -> Δlog10(y)
    return y_norm * jnp.clip(norm.out_std_mlp, norm.min_std, jnp.inf) + norm.out_mean_mlp


def _denorm_deeponet_logy(y_norm: jnp.ndarray, norm: NormStatsJax) -> jnp.ndarray:
    # y_norm: normalized log10(y_next) [B,3] -> log10(y_next)
    return y_norm * jnp.clip(norm.out_std_deeponet, norm.min_std, jnp.inf) + norm.out_mean_deeponet


def _flowmap_step_mlp(
    params: Any,
    cache: SplitCache,
    norm: NormStatsJax,
    act: Callable[[jnp.ndarray], jnp.ndarray],
    traj_idx: jnp.ndarray,
    step_idx: jnp.ndarray,
    logy_curr: jnp.ndarray,
) -> jnp.ndarray:
    # Build inputs using provided logy_curr (instead of cache.logy[...]) for rollout.
    logdt = cache.logdt[traj_idx, step_idx]  # [B]
    logy0 = cache.logy0[traj_idx, :]         # [B,3]
    logk = cache.logk[traj_idx, :]           # [B,3]

    x_state = _zscore(logy_curr, norm.state_mean, norm.state_std, norm.min_std)
    x_dt = _zscore(logdt[:, None], norm.dt_mean, norm.dt_std, norm.min_std)
    x_y0 = _zscore(logy0, norm.y0_mean, norm.y0_std, norm.min_std)
    x_k = _zscore(logk, norm.k_mean, norm.k_std, norm.min_std)
    x = jnp.concatenate([x_state, x_dt, x_y0, x_k], axis=-1)

    y_norm = mlp_apply(params, x, act)
    dlog = _denorm_mlp_dlog(y_norm, norm)
    return logy_curr + dlog


def _flowmap_step_deeponet(
    params: Any,
    cache: SplitCache,
    norm: NormStatsJax,
    act: Callable[[jnp.ndarray], jnp.ndarray],
    traj_idx: jnp.ndarray,
    step_idx: jnp.ndarray,
    logy_curr: jnp.ndarray,
) -> jnp.ndarray:
    logdt = cache.logdt[traj_idx, step_idx]  # [B]
    logy0 = cache.logy0[traj_idx, :]         # [B,3]
    logk = cache.logk[traj_idx, :]           # [B,3]

    x_state = _zscore(logy_curr, norm.state_mean, norm.state_std, norm.min_std)
    x_dt = _zscore(logdt[:, None], norm.dt_mean, norm.dt_std, norm.min_std)
    x_y0 = _zscore(logy0, norm.y0_mean, norm.y0_std, norm.min_std)
    x_k = _zscore(logk, norm.k_mean, norm.k_std, norm.min_std)

    x_branch = jnp.concatenate([x_y0, x_k], axis=-1)
    x_trunk = jnp.concatenate([x_state, x_dt], axis=-1)

    y_norm = deeponet_apply(params, x_branch, x_trunk, act)
    return _denorm_deeponet_logy(y_norm, norm)


def _logy_to_y_simplex(logy: jnp.ndarray, eps: float) -> jnp.ndarray:
    y = jnp.power(10.0, logy) - eps
    y = jnp.clip(y, 0.0, jnp.inf)
    y = y / jnp.clip(jnp.sum(y, axis=-1, keepdims=True), 1e-30, jnp.inf)
    return y


def compute_n_jump_fractional_errors(
    cfg: Config,
    model_type: str,
    params: Any,
    cache: SplitCache,
    norm: NormStatsJax,
    *,
    horizons: Sequence[int],
    n_segments: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Compute N-jump fractional errors for evaluation (not used in training).

    For each sampled segment:
      - pick (traj, start_step) uniformly with start_step <= n_steps - 1 - max_horizon
      - rollout the model for each horizon h in horizons
      - compare y_pred(t+h) vs y_true(t+h) with fractional error:
            |y_pred - y_true| / max(y_true, tiny)

    Returns a dict with:
      - "horizons": int array [H]
      - "mean_frac_err": float array [H, 3] (per species)
      - "mean_frac_err_all": float array [H] (species-mean)
    """
    if len(horizons) < 1:
        raise ValueError("horizons must be non-empty.")
    max_h = int(max(int(h) for h in horizons))
    if max_h < 1:
        raise ValueError("All horizons must be >= 1.")

    n_traj = int(cache.logy.shape[0])
    n_steps = int(cache.logdt.shape[1])

    if max_h > n_steps - 1:
        raise ValueError(f"Max horizon {max_h} exceeds available steps {n_steps}.")

    act = get_activation(cfg.activation)
    tiny = 1e-30

    key = jax.random.PRNGKey(int(seed))
    k1, k2 = jax.random.split(key, 2)
    traj_idx = jax.random.randint(k1, (int(n_segments),), 0, n_traj)
    start_idx = jax.random.randint(k2, (int(n_segments),), 0, n_steps - max_h)

    def rollout_to_h(traj_i: jnp.ndarray, start_i: jnp.ndarray, h: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Initial condition in log space from cached truth
        logy0 = cache.logy[traj_i, start_i, :][None, :]  # [1,3]
        traj_b = traj_i[None]
        logy_curr = logy0

        def body(logy_c: jnp.ndarray, step_off: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
            step = start_i + step_off
            if model_type == "mlp":
                logy_next = _flowmap_step_mlp(params, cache, norm, act, traj_b, step[None], logy_c)
            elif model_type == "deeponet":
                logy_next = _flowmap_step_deeponet(params, cache, norm, act, traj_b, step[None], logy_c)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            return logy_next, None

        step_offsets = jnp.arange(h, dtype=jnp.int32)
        logy_pred, _ = jax.lax.scan(body, logy_curr, step_offsets)

        logy_true = cache.logy[traj_i, start_i + h, :][None, :]  # [1,3]
        y_pred = _logy_to_y_simplex(logy_pred, float(norm.eps))
        y_true = _logy_to_y_simplex(logy_true, float(norm.eps))
        frac = jnp.abs(y_pred - y_true) / jnp.clip(y_true, tiny, jnp.inf)
        return frac[0], jnp.mean(frac)

    # Vectorize over segments for each horizon.
    def per_horizon(h: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        frac_species, frac_all = jax.vmap(lambda ti, si: rollout_to_h(ti, si, h))(traj_idx, start_idx)
        return jnp.mean(frac_species, axis=0), jnp.mean(frac_all, axis=0)

    mean_species_list = []
    mean_all_list = []
    for h in horizons:
        ms, ma = per_horizon(int(h))
        mean_species_list.append(ms)
        mean_all_list.append(ma)

    mean_species = jnp.stack(mean_species_list, axis=0)
    mean_all = jnp.stack(mean_all_list, axis=0)

    return {
        "horizons": np.asarray(list(horizons), dtype=np.int32),
        "mean_frac_err": np.asarray(mean_species),
        "mean_frac_err_all": np.asarray(mean_all),
    }


# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    if CONFIG.tuning.enabled:
        run_optuna_tuning(CONFIG)
    else:
        run_train(CONFIG)


if __name__ == "__main__":
    main()
