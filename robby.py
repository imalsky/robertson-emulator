#!/usr/bin/env python3
"""
Robertson ML emulator (single-file science framework, JAX-first).

What this does
- Generates stiff Robertson ODE trajectories with a stiff-capable JAX integrator (diffrax).
- Builds TWO emulators:
    1) Flow-map MLP:      (y_t, dt) -> y_{t+1}
    2) DeepONet-style:    branch(y_t) ⊙ trunk(dt) -> y_{t+1}
- Trains both on ONE-JUMP supervision (adjacent transitions) with best-practice trajectory-level splits.
- Uses log10 + z-score normalization for all features/targets.
- Evaluates fractional error for 1-step, 2-step, 3-step, ... autoregressive rollout (from initial condition).
- Provides an Optuna objective for efficient hyperparameter search.

Dependencies (common in JAX stacks)
- jax, jaxlib
- optax
- diffrax
- tqdm
- optuna (optional; only needed if you set CONFIG.run_optuna=True)

Notes / defaults (chosen to be reasonable for Robertson)
- Time grid: log-spaced from t_min to t_final, plus t=0
- Stiff integrator: diffrax.Kvaerno5 with PIDController
- Initial conditions: mixture biased toward [1,0,0] but with variability (Dirichlet)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import math
import time
import csv
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import optax
from tqdm import tqdm

# -----------------------------
# CONFIG (no argparse; edit here)
# -----------------------------
@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 0

    # Dataset generation
    dataset_dir: str = "robertson_data_cache"
    use_cache: bool = True

    n_trajectories: int = 512            # total trajectories (train+val+test)
    n_save: int = 256                    # number of saved states per trajectory (including t=0)
    t_min: float = 1e-8                  # first positive save time
    t_final: float = 1e5                 # final time (classic Robertson spans wide time scales)
    rtol: float = 1e-7
    atol: float = 1e-10
    dt0: float = 1e-12                   # initial solver step guess

    # Split best practice: split by trajectory, not by transition
    frac_train: float = 0.8
    frac_val: float = 0.1                # test is remainder

    # Normalization (log10 then z-score)
    eps: float = 1e-30                   # for log10(y + eps)

    # Training
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0

    # Cosine schedule with warmup
    warmup_frac: float = 0.05            # fraction of total steps for linear warmup
    lr_min_frac: float = 0.05            # final lr = lr * lr_min_frac

    # Model selection
    model_type: str = "mlp"              # "mlp" or "deeponet"

    # Capacity controls (MLP)
    mlp_depth: int = 3
    mlp_width: int = 256

    # Capacity controls (DeepONet)
    deeponet_depth: int = 3
    deeponet_branch_width: int = 192
    deeponet_trunk_width: int = 192
    deeponet_feature_dim: int = 256      # shared latent feature dimension

    # Activation
    activation: str = "swish"            # "relu", "swish", "gelu", "tanh", "elu"

    # Evaluation: autoregressive steps reported from initial condition
    eval_max_steps: int = 32             # report 1..eval_max_steps (limited by n_save-1)

    # Logging
    log_dir: str = "runs_robertson"
    run_name: str = "baseline"

    # Optuna
    run_optuna: bool = False
    optuna_trials: int = 25
    optuna_epochs: int = 50              # for tuning speed; set to 100 if you want full
    optuna_dataset_frac: float = 0.5     # fraction of trajectories used during tuning
    optuna_metric: str = "val_onejump_frac"  # objective: "val_mse" or "val_onejump_frac"


CONFIG = Config()


# -----------------------------
# Robertson ODE (stiff system)
# -----------------------------
def robertson_rhs(t: jnp.ndarray, y: jnp.ndarray, args: None) -> jnp.ndarray:
    # y = [y1, y2, y3]
    y1, y2, y3 = y[0], y[1], y[2]
    dy1 = -0.04 * y1 + 1.0e4 * y2 * y3
    dy2 = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * (y2 * y2)
    dy3 = 3.0e7 * (y2 * y2)
    return jnp.array([dy1, dy2, dy3], dtype=y.dtype)


# -----------------------------
# Diffrax simulation (JAX-native stiff solver)
# -----------------------------
def build_time_grid(cfg: Config) -> np.ndarray:
    # Include t=0 plus log-spaced positive times
    ts_pos = np.logspace(np.log10(cfg.t_min), np.log10(cfg.t_final), cfg.n_save - 1)
    ts = np.concatenate([np.array([0.0], dtype=np.float64), ts_pos.astype(np.float64)])
    return ts


def sample_initial_conditions(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Mixture distribution:
    - Half: strongly biased toward y1~1, y2,y3 tiny (captures canonical Robertson stiffness regime).
    - Half: more generic Dirichlet (broader coverage).
    Always nonnegative and sums to 1.
    """
    n1 = n // 2
    n2 = n - n1

    # Biased (sparse-ish) Dirichlet: encourages y2,y3 small but not always zero.
    y_biased = rng.dirichlet(alpha=np.array([16.0, 0.35, 0.35], dtype=np.float64), size=n1)

    # Broad Dirichlet
    y_broad = rng.dirichlet(alpha=np.array([2.0, 2.0, 2.0], dtype=np.float64), size=n2)

    y0 = np.concatenate([y_biased, y_broad], axis=0)
    rng.shuffle(y0, axis=0)
    return y0.astype(np.float64)


def simulate_trajectory_diffrax(
    y0: jnp.ndarray,
    ts: jnp.ndarray,
    rtol: float,
    atol: float,
    dt0: float,
) -> jnp.ndarray:
    import diffrax  # local import so file still imports without diffrax if you don't run generation

    term = diffrax.ODETerm(robertson_rhs)
    solver = diffrax.Kvaerno5()

    # Stiff-friendly adaptive control
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
        max_steps=1_000_000,
    )

    ys = sol.ys  # [T, 3]

    # Keep physical constraints gentle and simple: nonneg + renormalize (helps minor solver drift)
    ys = jnp.clip(ys, 0.0, jnp.inf)
    ys = ys / jnp.clip(jnp.sum(ys, axis=-1, keepdims=True), 1e-30, jnp.inf)
    return ys


# -----------------------------
# Data pack: transitions + rollout arrays
# -----------------------------
def log10_safe(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return jnp.log10(x + eps)


def build_splits(cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(cfg.n_trajectories)
    rng.shuffle(idx)
    n_train = int(round(cfg.frac_train * cfg.n_trajectories))
    n_val = int(round(cfg.frac_val * cfg.n_trajectories))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def cache_paths(cfg: Config) -> Dict[str, Path]:
    d = Path(cfg.dataset_dir)
    d.mkdir(parents=True, exist_ok=True)
    tag = f"traj{cfg.n_trajectories}_save{cfg.n_save}_t{cfg.t_final:g}_seed{cfg.seed}"
    return {
        "npz": d / f"robertson_{tag}.npz",
        "meta": d / f"robertson_{tag}_meta.txt",
    }


def generate_or_load_dataset(cfg: Config) -> Dict[str, np.ndarray]:
    paths = cache_paths(cfg)
    if cfg.use_cache and paths["npz"].exists():
        data = np.load(paths["npz"], allow_pickle=False)
        return {k: data[k] for k in data.files}

    rng = np.random.default_rng(cfg.seed)
    ts = build_time_grid(cfg)  # numpy
    y0s = sample_initial_conditions(rng, cfg.n_trajectories)

    ts_j = jnp.asarray(ts, dtype=jnp.float64)

    # JIT the simulator once (diffrax is JIT-friendly; compilation happens at first call)
    sim_jit = jax.jit(lambda y0: simulate_trajectory_diffrax(y0, ts_j, cfg.rtol, cfg.atol, cfg.dt0))

    ys_all = []
    for i in tqdm(range(cfg.n_trajectories), desc="Simulating trajectories"):
        y0 = jnp.asarray(y0s[i], dtype=jnp.float64)
        ys = sim_jit(y0)  # [T,3]
        ys_all.append(np.asarray(ys))

    ys_all = np.stack(ys_all, axis=0)  # [Ntraj, T, 3], float64
    dts = np.diff(ts).astype(np.float64)  # [T-1]

    out = {
        "ts": ts.astype(np.float64),
        "dts": dts.astype(np.float64),
        "y0s": y0s.astype(np.float64),
        "ys": ys_all.astype(np.float64),
    }

    np.savez_compressed(paths["npz"], **out)
    paths["meta"].write_text(
        f"Generated: {time.ctime()}\n"
        f"n_trajectories={cfg.n_trajectories}, n_save={cfg.n_save}\n"
        f"t_min={cfg.t_min}, t_final={cfg.t_final}\n"
        f"rtol={cfg.rtol}, atol={cfg.atol}, dt0={cfg.dt0}\n"
        f"seed={cfg.seed}\n"
    )
    return out


def build_transitions(ys: np.ndarray, dts: np.ndarray) -> Dict[str, np.ndarray]:
    """
    ys: [Ntraj, T, 3]
    dts: [T-1]
    Returns transitions aligned per-trajectory:
      y_t:    [Ntraj, T-1, 3]
      y_tp1:  [Ntraj, T-1, 3]
      dt:     [Ntraj, T-1, 1]
    """
    y_t = ys[:, :-1, :]
    y_tp1 = ys[:, 1:, :]
    dt = np.broadcast_to(dts[None, :, None], (ys.shape[0], dts.shape[0], 1))
    return {"y_t": y_t, "y_tp1": y_tp1, "dt": dt}


def compute_norm_stats(cfg: Config, trans_train: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # Log10 for y and dt, then mean/std (z-score)
    y_t = jnp.asarray(trans_train["y_t"])
    y_tp1 = jnp.asarray(trans_train["y_tp1"])
    dt = jnp.asarray(trans_train["dt"])

    x_state = log10_safe(y_t, cfg.eps)       # [..., 3]
    x_dt = jnp.log10(dt)                     # [..., 1]
    y_out = log10_safe(y_tp1, cfg.eps)       # [..., 3]

    # Flatten across (traj, time)
    x_state_f = x_state.reshape(-1, 3)
    x_dt_f = x_dt.reshape(-1, 1)
    y_out_f = y_out.reshape(-1, 3)

    def mean_std(a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        m = jnp.mean(a, axis=0)
        s = jnp.std(a, axis=0)
        s = jnp.where(s < 1e-12, 1e-12, s)
        return m, s

    state_mean, state_std = mean_std(x_state_f)
    dt_mean, dt_std = mean_std(x_dt_f)
    out_mean, out_std = mean_std(y_out_f)

    return {
        "eps": np.array(cfg.eps, dtype=np.float64),
        "state_mean": np.asarray(state_mean),
        "state_std": np.asarray(state_std),
        "dt_mean": np.asarray(dt_mean),
        "dt_std": np.asarray(dt_std),
        "out_mean": np.asarray(out_mean),
        "out_std": np.asarray(out_std),
    }


def normalize_transitions(cfg: Config, trans: Dict[str, np.ndarray], norm: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    eps = float(norm["eps"])
    state_mean = jnp.asarray(norm["state_mean"])
    state_std = jnp.asarray(norm["state_std"])
    dt_mean = jnp.asarray(norm["dt_mean"])
    dt_std = jnp.asarray(norm["dt_std"])
    out_mean = jnp.asarray(norm["out_mean"])
    out_std = jnp.asarray(norm["out_std"])

    y_t = jnp.asarray(trans["y_t"])
    y_tp1 = jnp.asarray(trans["y_tp1"])
    dt = jnp.asarray(trans["dt"])

    x_state = (log10_safe(y_t, eps) - state_mean) / state_std
    x_dt = (jnp.log10(dt) - dt_mean) / dt_std
    y_out = (log10_safe(y_tp1, eps) - out_mean) / out_std

    return {
        "x_state": np.asarray(x_state),
        "x_dt": np.asarray(x_dt),
        "y_out": np.asarray(y_out),
        # Keep physical for fractional error computation if desired
        "y_t_phys": np.asarray(y_t),
        "y_tp1_phys": np.asarray(y_tp1),
        "dt_phys": np.asarray(dt),
    }


# -----------------------------
# Minimal MLP + DeepONet (no flax/equinox)
# -----------------------------
def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
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


def init_dense(key: jax.Array, fan_in: int, fan_out: int) -> Dict[str, jax.Array]:
    # Simple, decent default (LeCun/He-ish scaling)
    w_key, b_key = jax.random.split(key)
    std = 1.0 / math.sqrt(max(1, fan_in))
    W = std * jax.random.normal(w_key, (fan_in, fan_out), dtype=jnp.float32)
    b = jnp.zeros((fan_out,), dtype=jnp.float32)
    return {"W": W, "b": b}


def mlp_init(key: jax.Array, in_dim: int, out_dim: int, depth: int, width: int) -> List[Dict[str, jax.Array]]:
    keys = jax.random.split(key, depth + 1)
    layers = []
    d0 = in_dim
    for i in range(depth):
        layers.append(init_dense(keys[i], d0, width))
        d0 = width
    layers.append(init_dense(keys[-1], d0, out_dim))
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
) -> Dict[str, object]:
    k1, k2, k3 = jax.random.split(key, 3)
    branch = mlp_init(k1, branch_in, feature_dim, depth, branch_width)
    trunk = mlp_init(k2, trunk_in, feature_dim, depth, trunk_width)
    head = init_dense(k3, feature_dim, out_dim)
    return {"branch": branch, "trunk": trunk, "head": head}


def deeponet_apply(
    params: Dict[str, object],
    x_branch: jnp.ndarray,
    x_trunk: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    b = mlp_apply(params["branch"], x_branch, act)  # [feature_dim]
    t = mlp_apply(params["trunk"], x_trunk, act)    # [feature_dim]
    h = b * t
    head = params["head"]
    return h @ head["W"] + head["b"]                # [out_dim]


def tree_num_params(pytree) -> int:
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    return int(sum(int(np.prod(x.shape)) for x in leaves))


def match_deeponet_widths_to_mlp(cfg: Config) -> Config:
    """
    Cheap param matching:
    - Computes MLP param count.
    - Searches a small grid over branch/trunk widths and feature_dim to get close.
    """
    key = jax.random.PRNGKey(0)
    act = get_activation(cfg.activation)

    mlp_params = mlp_init(key, in_dim=4, out_dim=3, depth=cfg.mlp_depth, width=cfg.mlp_width)
    target = tree_num_params(mlp_params)

    # Search around current DeepONet settings
    best = None
    best_cfg = cfg
    for bw in [max(32, cfg.deeponet_branch_width // 2), cfg.deeponet_branch_width, cfg.deeponet_branch_width * 2]:
        for tw in [max(32, cfg.deeponet_trunk_width // 2), cfg.deeponet_trunk_width, cfg.deeponet_trunk_width * 2]:
            for fd in [max(32, cfg.deeponet_feature_dim // 2), cfg.deeponet_feature_dim, cfg.deeponet_feature_dim * 2]:
                dp = deeponet_init(
                    key,
                    branch_in=3,
                    trunk_in=1,
                    feature_dim=fd,
                    depth=cfg.deeponet_depth,
                    branch_width=bw,
                    trunk_width=tw,
                    out_dim=3,
                )
                n = tree_num_params(dp)
                diff = abs(n - target) / max(1, target)
                if (best is None) or (diff < best):
                    best = diff
                    best_cfg = replace(
                        cfg,
                        deeponet_branch_width=int(bw),
                        deeponet_trunk_width=int(tw),
                        deeponet_feature_dim=int(fd),
                    )
    return best_cfg


# -----------------------------
# Normalization decode for rollout
# -----------------------------
def decode_pred_to_phys(
    pred_out_norm: jnp.ndarray,
    norm: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    pred_out_norm: [..., 3] normalized log10(y_next + eps)
    returns y_next_phys: [..., 3] (nonneg, renormalized)
    """
    eps = norm["eps"]
    out_mean = norm["out_mean"]
    out_std = norm["out_std"]

    logy = pred_out_norm * out_std + out_mean
    y = jnp.power(10.0, logy) - eps
    y = jnp.clip(y, 0.0, jnp.inf)
    y = y / jnp.clip(jnp.sum(y, axis=-1, keepdims=True), 1e-30, jnp.inf)
    return y


def encode_state_dt(
    y_phys: jnp.ndarray,
    dt_phys: jnp.ndarray,
    norm: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    eps = norm["eps"]
    state_mean = norm["state_mean"]
    state_std = norm["state_std"]
    dt_mean = norm["dt_mean"]
    dt_std = norm["dt_std"]

    x_state = (jnp.log10(y_phys + eps) - state_mean) / state_std  # [...,3]
    x_dt = (jnp.log10(dt_phys) - dt_mean) / dt_std                # [...,1]
    return x_state, x_dt


# -----------------------------
# Training utilities
# -----------------------------
def make_schedule(cfg: Config, total_steps: int) -> optax.Schedule:
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    lr0 = cfg.lr
    lr_min = cfg.lr * cfg.lr_min_frac

    warmup = optax.linear_schedule(init_value=0.0, end_value=lr0, transition_steps=warmup_steps)
    cosine = optax.cosine_decay_schedule(init_value=lr0, decay_steps=decay_steps, alpha=lr_min / lr0)

    # Join schedules: warmup then cosine
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def make_optimizer(cfg: Config, schedule: optax.Schedule) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )


def make_batches(n: int, batch_size: int) -> List[slice]:
    return [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def flatten_transitions(normed: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    # Collapse (traj, time) -> (N, ...)
    x_state = jnp.asarray(normed["x_state"]).reshape(-1, 3)
    x_dt = jnp.asarray(normed["x_dt"]).reshape(-1, 1)
    y_out = jnp.asarray(normed["y_out"]).reshape(-1, 3)

    y_t_phys = jnp.asarray(normed["y_t_phys"]).reshape(-1, 3)
    y_tp1_phys = jnp.asarray(normed["y_tp1_phys"]).reshape(-1, 3)
    dt_phys = jnp.asarray(normed["dt_phys"]).reshape(-1, 1)

    return {
        "x_state": x_state,
        "x_dt": x_dt,
        "y_out": y_out,
        "y_t_phys": y_t_phys,
        "y_tp1_phys": y_tp1_phys,
        "dt_phys": dt_phys,
    }


def forward_apply(
    cfg: Config,
    params,
    x_state: jnp.ndarray,
    x_dt: jnp.ndarray,
    act: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    if cfg.model_type == "mlp":
        x = jnp.concatenate([x_state, x_dt], axis=-1)  # [...,4]
        return mlp_apply(params, x, act)               # [...,3]
    if cfg.model_type == "deeponet":
        return deeponet_apply(params, x_state, x_dt, act)  # [...,3]
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def init_model(cfg: Config, key: jax.Array):
    if cfg.model_type == "mlp":
        return mlp_init(key, in_dim=4, out_dim=3, depth=cfg.mlp_depth, width=cfg.mlp_width)
    if cfg.model_type == "deeponet":
        return deeponet_init(
            key,
            branch_in=3,
            trunk_in=1,
            feature_dim=cfg.deeponet_feature_dim,
            depth=cfg.deeponet_depth,
            branch_width=cfg.deeponet_branch_width,
            trunk_width=cfg.deeponet_trunk_width,
            out_dim=3,
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


# -----------------------------
# Metrics
# -----------------------------
def fractional_error(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-30) -> jnp.ndarray:
    # Mean over batch and components: |pred-true| / (|true| + eps)
    return jnp.mean(jnp.abs(y_pred - y_true) / (jnp.abs(y_true) + eps))


def onejump_fractional_error_from_norm(
    pred_out_norm: jnp.ndarray,
    y_tp1_phys: jnp.ndarray,
    norm: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    y_pred = decode_pred_to_phys(pred_out_norm, norm)
    return fractional_error(y_pred, y_tp1_phys, eps=float(norm["eps"]))


def rollout_autoregressive_errors(
    cfg: Config,
    params,
    act: Callable[[jnp.ndarray], jnp.ndarray],
    norm: Dict[str, jnp.ndarray],
    ys_test: jnp.ndarray,   # [Ntraj_test, T, 3] physical
    dts: jnp.ndarray,       # [T-1] physical
) -> jnp.ndarray:
    """
    Autoregressive rollout from each trajectory initial condition; report fractional error at step k.
    Returns errors[1..K] (shape [K]) where K = min(cfg.eval_max_steps, T-1).
    """
    T = ys_test.shape[1]
    K = int(min(cfg.eval_max_steps, T - 1))

    dts_use = dts[:K]  # [K]
    dts_use = dts_use[:, None]  # [K,1]

    def step_fn(y_curr, dt_step):
        x_state, x_dt = encode_state_dt(y_curr, dt_step, norm)
        pred_norm = forward_apply(cfg, params, x_state, x_dt, act)
        y_next = decode_pred_to_phys(pred_norm, norm)
        return y_next, y_next

    def rollout_one_traj(y0, y_true_traj):
        # y_true_traj: [T,3], physical
        _, ys_pred = lax.scan(step_fn, y0, dts_use)
        ys_pred_full = jnp.concatenate([y0[None, :], ys_pred], axis=0)  # [K+1,3]
        y_true_k = y_true_traj[: K + 1, :]
        # errors at steps 1..K (ignore step 0)
        per_step = jnp.mean(jnp.abs(ys_pred_full[1:] - y_true_k[1:]) / (jnp.abs(y_true_k[1:]) + float(norm["eps"])), axis=-1)
        return per_step  # [K]

    errs = jax.vmap(rollout_one_traj)(ys_test[:, 0, :], ys_test)  # [Ntraj_test, K]
    return jnp.mean(errs, axis=0)  # [K]


# -----------------------------
# Training loop (JIT-ed steps)
# -----------------------------
def train_and_evaluate(
    cfg: Config,
    data: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    log_tag: str,
) -> Dict[str, float]:
    # Build per-split transitions
    trans = build_transitions(data["ys"], data["dts"])
    trans_train = {k: v[train_idx] for k, v in trans.items()}
    trans_val = {k: v[val_idx] for k, v in trans.items()}
    trans_test = {k: v[test_idx] for k, v in trans.items()}

    norm_np = compute_norm_stats(cfg, trans_train)
    norm = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in norm_np.items()}
    norm["eps"] = jnp.asarray(float(norm_np["eps"]), dtype=jnp.float32)

    norm_train = normalize_transitions(cfg, trans_train, norm_np)
    norm_val = normalize_transitions(cfg, trans_val, norm_np)
    norm_test = normalize_transitions(cfg, trans_test, norm_np)

    flat_train = flatten_transitions(norm_train)
    flat_val = flatten_transitions(norm_val)
    flat_test = flatten_transitions(norm_test)

    n_train = flat_train["x_state"].shape[0]
    n_val = flat_val["x_state"].shape[0]
    n_test = flat_test["x_state"].shape[0]

    total_steps = (n_train // cfg.batch_size + int(n_train % cfg.batch_size != 0)) * cfg.epochs
    schedule = make_schedule(cfg, total_steps)
    opt = make_optimizer(cfg, schedule)

    # Init model
    key = jax.random.PRNGKey(cfg.seed)
    params = init_model(cfg, key)
    opt_state = opt.init(params)

    act = get_activation(cfg.activation)

    # JIT-ed train step
    @jax.jit
    def train_step(params, opt_state, x_state, x_dt, y_out):
        def loss_fn(p):
            pred = forward_apply(cfg, p, x_state, x_dt, act)
            loss = jnp.mean((pred - y_out) ** 2)
            return loss, pred

        (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    # JIT-ed eval step
    @jax.jit
    def eval_step(params, x_state, x_dt, y_out, y_tp1_phys):
        pred = forward_apply(cfg, params, x_state, x_dt, act)
        mse = jnp.mean((pred - y_out) ** 2)
        frac = onejump_fractional_error_from_norm(pred, y_tp1_phys, norm)
        return mse, frac

    # Logging
    run_dir = Path(cfg.log_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{log_tag}_metrics.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse", "val_onejump_frac"])

    # Training loop
    rng = np.random.default_rng(cfg.seed + 123)

    train_batches = make_batches(n_train, cfg.batch_size)
    val_batches = make_batches(n_val, cfg.batch_size)

    step_count = 0
    for epoch in tqdm(range(cfg.epochs), desc=f"Training ({cfg.model_type})"):
        # Shuffle indices each epoch
        perm = rng.permutation(n_train)

        train_losses = []
        for sl in train_batches:
            idx = perm[sl]
            xb_state = flat_train["x_state"][idx]
            xb_dt = flat_train["x_dt"][idx]
            yb = flat_train["y_out"][idx]

            params, opt_state, loss = train_step(params, opt_state, xb_state, xb_dt, yb)
            train_losses.append(float(loss))
            step_count += 1

        # Validation
        val_mses = []
        val_fracs = []
        for sl in val_batches:
            xb_state = flat_val["x_state"][sl]
            xb_dt = flat_val["x_dt"][sl]
            yb = flat_val["y_out"][sl]
            yb_phys = flat_val["y_tp1_phys"][sl]
            mse, frac = eval_step(params, xb_state, xb_dt, yb, yb_phys)
            val_mses.append(float(mse))
            val_fracs.append(float(frac))

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_mses))
        val_onejump_frac = float(np.mean(val_fracs))

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_mse, val_mse, val_onejump_frac])

    # Final test metrics (one-jump)
    test_batches = make_batches(n_test, cfg.batch_size)
    test_mses = []
    test_fracs = []
    for sl in test_batches:
        xb_state = flat_test["x_state"][sl]
        xb_dt = flat_test["x_dt"][sl]
        yb = flat_test["y_out"][sl]
        yb_phys = flat_test["y_tp1_phys"][sl]
        mse, frac = eval_step(params, xb_state, xb_dt, yb, yb_phys)
        test_mses.append(float(mse))
        test_fracs.append(float(frac))
    test_mse = float(np.mean(test_mses))
    test_onejump_frac = float(np.mean(test_fracs))

    # Autoregressive rollout errors (from initial condition)
    ys_test = jnp.asarray(data["ys"][test_idx], dtype=jnp.float32)     # [Ntest, T, 3]
    dts = jnp.asarray(data["dts"], dtype=jnp.float32)                 # [T-1]
    ar_errs = rollout_autoregressive_errors(cfg, params, act, norm, ys_test, dts)  # [K]
    ar_errs_np = np.asarray(ar_errs)

    # Print a compact summary
    print("\n=== Summary ===")
    print(f"model_type={cfg.model_type} activation={cfg.activation}")
    print(f"params={tree_num_params(params):,}")
    print(f"test_onejump_frac={test_onejump_frac:.3e}  test_mse={test_mse:.3e}")
    k_show = min(10, ar_errs_np.shape[0])
    print("autoregressive fractional error (steps 1..{}):".format(k_show))
    print("  " + "  ".join([f"{ar_errs_np[i]:.3e}" for i in range(k_show)]))

    # Return scalars for Optuna (and general use)
    out = {
        "test_mse": test_mse,
        "test_onejump_frac": test_onejump_frac,
        "val_last_mse": val_mse,
        "val_last_onejump_frac": val_onejump_frac,
    }
    # Also stash AR curve summary
    for i in range(ar_errs_np.shape[0]):
        out[f"test_ar_frac_step_{i+1}"] = float(ar_errs_np[i])
    return out


# -----------------------------
# Optuna integration
# -----------------------------
def run_optuna(cfg: Config, data: Dict[str, np.ndarray]) -> None:
    import optuna

    # Build base splits once; objective may subselect for speed
    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx, test_idx = build_splits(cfg, rng)

    # Optional: reduce dataset size during tuning
    if cfg.optuna_dataset_frac < 1.0:
        def subselect(idx: np.ndarray, frac: float) -> np.ndarray:
            n = max(1, int(round(frac * idx.shape[0])))
            return np.random.default_rng(cfg.seed + 999).choice(idx, size=n, replace=False)

        train_idx = subselect(train_idx, cfg.optuna_dataset_frac)
        val_idx = subselect(val_idx, cfg.optuna_dataset_frac)
        test_idx = subselect(test_idx, cfg.optuna_dataset_frac)

    base = cfg

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model_type", ["mlp", "deeponet"])
        activation = trial.suggest_categorical("activation", ["relu", "swish", "gelu", "tanh", "elu"])

        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
        bs = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192])

        # Capacity knobs with rough param parity
        if model_type == "mlp":
            depth = trial.suggest_int("mlp_depth", 2, 5)
            width = trial.suggest_categorical("mlp_width", [128, 192, 256, 320, 384])
            cfg_t = replace(
                base,
                model_type=model_type,
                activation=activation,
                lr=lr,
                weight_decay=wd,
                batch_size=bs,
                epochs=base.optuna_epochs,
                mlp_depth=depth,
                mlp_width=width,
            )
        else:
            depth = trial.suggest_int("deeponet_depth", 2, 5)
            feat = trial.suggest_categorical("deeponet_feature_dim", [128, 192, 256, 320, 384])
            bw = trial.suggest_categorical("deeponet_branch_width", [96, 128, 160, 192, 224])
            tw = trial.suggest_categorical("deeponet_trunk_width", [96, 128, 160, 192, 224])
            cfg_t = replace(
                base,
                model_type=model_type,
                activation=activation,
                lr=lr,
                weight_decay=wd,
                batch_size=bs,
                epochs=base.optuna_epochs,
                deeponet_depth=depth,
                deeponet_feature_dim=feat,
                deeponet_branch_width=bw,
                deeponet_trunk_width=tw,
            )

        # Try to match params across types (keeps comparison fair-ish)
        if cfg_t.model_type == "deeponet":
            cfg_t = match_deeponet_widths_to_mlp(cfg_t)

        metrics = train_and_evaluate(
            cfg_t,
            data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            log_tag=f"optuna_trial{trial.number}",
        )

        if base.optuna_metric == "val_mse":
            return metrics["val_last_mse"]
        return metrics["val_last_onejump_frac"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.optuna_trials)

    print("\n=== Optuna best ===")
    print("value:", study.best_value)
    print("params:", study.best_params)


# -----------------------------
# Main
# -----------------------------
def main(cfg: Config) -> None:
    # (Optional) For fairness, if running DeepONet baseline, match its params to MLP baseline.
    if cfg.model_type == "deeponet":
        cfg = match_deeponet_widths_to_mlp(cfg)

    data = generate_or_load_dataset(cfg)

    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx, test_idx = build_splits(cfg, rng)

    # Baseline run
    _ = train_and_evaluate(cfg, data, train_idx, val_idx, test_idx, log_tag="baseline")

    # Optional: hyperparameter search
    if cfg.run_optuna:
        run_optuna(cfg, data)


if __name__ == "__main__":
    main(CONFIG)
