#!/usr/bin/env python3
"""
train_robertson.py
==================

Robertson flow-map benchmark reproducing Malsky et al. 2026 (ApJ), Section 3.4
/ Figure 3 / Table 2.

Problem setup
-------------
Robertson ODE (paper Eq. 5):
    dx1/dt = -p1*x1 + p3*x2*x3
    dx2/dt =  p1*x1 - p3*x2*x3 - p2*x2^2
    dx3/dt =  p2*x2^2

Initial state (fixed, paper Table 2 caption):
    x(t0) = [1, 1e-30, 1e-30]

Per-trajectory reaction rates, sampled log-uniformly (paper L353):
    p1 in [2e-3, 6e-3]
    p2 in [1.5e7, 3.5e7]
    p3 in [5e3, 6e4]

Time grid (paper L353): 100 log-spaced points in [1e-5, 1e5] s, 10,000 trajectories.
Integrator: scipy.integrate.solve_ivp(method='Radau') (Radau IIA, 5th order,
stiffly stable; HairerWanner1996).

Each (p1, p2, p3, dt) is an input pair; each x(t0 + dt) is the target.

Models (paper Table 2)
----------------------
All four take (p1, p2, p3, dt) as input and predict x(t0 + dt) in R^3.

    MLP                 [4, 32, 32, 32, 32, 32, 32, 32, 32, 3]      7,651 params
    Flow-map            encoder [3, 32, 32, 16]
                        propagator [17, 32, 32, 32, 32, 16] (3 skips at width)
                        decoder [16, 32, 32, 3]
                        latent dim 16  (strictly < layer width 32)    7,683 params
    DeepONet            branch [3, 32, 32, 32, 32, 16]
                        trunk  [1, 32, 32, 32, 32, 16]
                        head Linear(16, 3)                            7,635 params

SiLU activation. Inputs log10-transformed; rates z-scored, time scaled to [0,1].
Outputs log10(x + eps) and z-scored (paper L353).

Training (matches main paper config where applicable):
    AdamW, lr 1e-4, weight_decay 1e-4, min_lr 5e-7
    Linear warmup 10 epochs, cosine annealing
    Batch size 1024, 250 epochs
    Hybrid loss: log10-MAE (physical units) + 0.5 * MSE (z-scored)

Run
---
    python -u train_robertson.py            # full pipeline (data + 4 models + plot)
    python -u train_robertson.py --skip-train   # plot from existing runs/ only

Artifacts
---------
    robertson_paper_data.npz   cached trajectories (regenerated if missing)
    runs/<model>/metrics.csv   per-epoch train/val mse_z and lr
    runs/<model>/model.pt      final weights
    runs/<model>/summary.json  n_params and test mse_z
    Fig3.pdf                   training-curve figure (matches paper Fig. 3)
    table2.txt                 tabular summary (matches paper Table 2)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# ====================================================================
# Constants (paper Section 3.4)
# ====================================================================
SEED = 0
N_TRAJECTORIES = 10_000
N_TIME_POINTS = 100
T_MIN, T_MAX = 1e-5, 1e5

P1_RANGE = (2e-3, 6e-3)
P2_RANGE = (1.5e7, 3.5e7)
P3_RANGE = (5e3, 6e4)

Y0 = np.array([1.0, 1e-30, 1e-30], dtype=np.float64)

RTOL = 1e-8
ATOL = 1e-14
EPS = 1e-30

BATCH_SIZE = 1024
EPOCHS = 250
LR = 1e-4
MIN_LR = 5e-7
WD = 1e-4
WARMUP_EPOCHS = 10
GRAD_CLIP_NORM = 1.0  # global-norm gradient clip
LAMBDA_LOG10_MAE = 1.0
LAMBDA_MSE_Z = 0.5

# Tiny floor on the warmup learning rate so the first step is not exactly
# zero. lr=0 has triggered intermittent NaN on Apple Silicon MPS.
LR_FLOOR = 1e-8

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "robertson_paper_data.npz"
RUNS_DIR = HERE / "runs"
LOCAL_FIG_PATH = HERE / "Fig3.pdf"
PAPER_FIG_PATH = HERE.parent.parent / "Chemulator_ApJ" / "Fig3.pdf"
TABLE2_PATH = HERE / "table2.txt"
STYLE_PATH = HERE / "science.mplstyle"


# ====================================================================
# Robertson ODE + data generation
# ====================================================================
def robertson_rhs(t: float, y: np.ndarray, p1: float, p2: float, p3: float) -> np.ndarray:
    y1, y2, y3 = y
    return np.array(
        [
            -p1 * y1 + p3 * y2 * y3,
            p1 * y1 - p3 * y2 * y3 - p2 * y2 * y2,
            p2 * y2 * y2,
        ]
    )


def robertson_jac(t: float, y: np.ndarray, p1: float, p2: float, p3: float) -> np.ndarray:
    y1, y2, y3 = y
    return np.array(
        [
            [-p1,             p3 * y3,                   p3 * y2],
            [ p1, -p3 * y3 - 2.0 * p2 * y2,             -p3 * y2],
            [0.0,              2.0 * p2 * y2,                0.0],
        ]
    )


def _integrate_one(args: Tuple[int, float, float, float, np.ndarray]) -> Tuple[int, bool, np.ndarray]:
    idx, p1, p2, p3, t_eval = args
    sol = solve_ivp(
        robertson_rhs,
        (0.0, float(t_eval[-1])),
        Y0,
        t_eval=t_eval,
        method="Radau",
        args=(p1, p2, p3),
        jac=robertson_jac,
        rtol=RTOL,
        atol=ATOL,
        dense_output=False,
    )
    if not sol.success:
        return idx, False, np.zeros((len(t_eval), 3), dtype=np.float64)
    ys = np.clip(sol.y.T, 0.0, np.inf)
    s = ys.sum(axis=-1, keepdims=True)
    ys = ys / np.clip(s, 1e-30, np.inf)
    return idx, True, ys


def generate_or_load_data() -> Dict[str, np.ndarray]:
    if DATA_FILE.exists():
        print(f"[data] loading cached dataset: {DATA_FILE}")
        z = np.load(DATA_FILE)
        return {k: z[k] for k in z.files}

    print(f"[data] generating {N_TRAJECTORIES} trajectories with Radau IIA")
    rng = np.random.default_rng(SEED)
    p1 = 10 ** rng.uniform(np.log10(P1_RANGE[0]), np.log10(P1_RANGE[1]), N_TRAJECTORIES)
    p2 = 10 ** rng.uniform(np.log10(P2_RANGE[0]), np.log10(P2_RANGE[1]), N_TRAJECTORIES)
    p3 = 10 ** rng.uniform(np.log10(P3_RANGE[0]), np.log10(P3_RANGE[1]), N_TRAJECTORIES)
    t_eval = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_TIME_POINTS)

    args_list = [(i, float(p1[i]), float(p2[i]), float(p3[i]), t_eval) for i in range(N_TRAJECTORIES)]
    ys_all = np.zeros((N_TRAJECTORIES, N_TIME_POINTS, 3), dtype=np.float64)
    valid = np.zeros(N_TRAJECTORIES, dtype=bool)

    n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"[data] using {n_workers} worker processes")
    with mp.Pool(n_workers) as pool:
        for i, ok, ys in tqdm(
            pool.imap_unordered(_integrate_one, args_list, chunksize=8),
            total=N_TRAJECTORIES, desc="[data] integrate", unit="traj",
        ):
            ys_all[i] = ys
            valid[i] = ok

    n_fail = int(np.sum(~valid))
    if n_fail:
        print(f"[data] WARNING: {n_fail} trajectories failed integration; dropping them")
        keep = valid
        p1, p2, p3, ys_all = p1[keep], p2[keep], p3[keep], ys_all[keep]

    np.savez_compressed(DATA_FILE, p1=p1, p2=p2, p3=p3, t_eval=t_eval, ys=ys_all)
    print(f"[data] saved {DATA_FILE} ({ys_all.shape[0]} trajectories)")
    return {"p1": p1, "p2": p2, "p3": p3, "t_eval": t_eval, "ys": ys_all}


# ====================================================================
# Dataset assembly
# ====================================================================
@dataclass
class NormStats:
    p_mean: np.ndarray
    p_std: np.ndarray
    dt_log_min: float
    dt_log_max: float
    y_log_mean: np.ndarray
    y_log_std: np.ndarray


def build_pairs(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t_eval: np.ndarray, ys: np.ndarray, traj_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    log_t = np.log10(t_eval).astype(np.float64)
    T = log_t.shape[0]
    log_p_sel = np.log10(np.stack([p1[traj_idx], p2[traj_idx], p3[traj_idx]], axis=-1))  # [N,3]
    log_y_sel = np.log10(ys[traj_idx] + EPS)                                              # [N,T,3]
    N = traj_idx.shape[0]

    X = np.empty((N * T, 4), dtype=np.float64)
    Y = np.empty((N * T, 3), dtype=np.float64)
    X[:, :3] = np.repeat(log_p_sel, T, axis=0)
    X[:, 3] = np.tile(log_t, N)
    Y[:, :] = log_y_sel.reshape(N * T, 3)
    return X, Y


def trajectory_split(n: int, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def compute_stats(X_train: np.ndarray, Y_train: np.ndarray) -> NormStats:
    return NormStats(
        p_mean=X_train[:, :3].mean(axis=0),
        p_std=np.maximum(X_train[:, :3].std(axis=0), 1e-12),
        dt_log_min=float(X_train[:, 3].min()),
        dt_log_max=float(X_train[:, 3].max()),
        y_log_mean=Y_train.mean(axis=0),
        y_log_std=np.maximum(Y_train.std(axis=0), 1e-12),
    )


def apply_norm(X: np.ndarray, Y: np.ndarray, s: NormStats) -> Tuple[np.ndarray, np.ndarray]:
    Xn = X.copy()
    Xn[:, :3] = (X[:, :3] - s.p_mean) / s.p_std
    Xn[:, 3] = (X[:, 3] - s.dt_log_min) / max(s.dt_log_max - s.dt_log_min, 1e-12)
    Yn = (Y - s.y_log_mean) / s.y_log_std
    return Xn, Yn


# ====================================================================
# Models (exact param counts from paper Table 2)
# ====================================================================
def _mlp(dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class MLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _mlp([4, 32, 32, 32, 32, 32, 32, 32, 32, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeRichPropagator(nn.Module):
    """Propagator that lifts (z, dt) from latent_dim+1 -> width hidden ->
    latent_dim out. Three skip-residual blocks at hidden width keep dt's
    information path deep (4 SiLU activations before the final reduction).
    With latent_dim=16, width=32: 576 + 3*1056 + 528 = 4,272 params.
    """
    def __init__(self, latent_dim: int = 16, width: int = 32) -> None:
        super().__init__()
        self.l1 = nn.Linear(latent_dim + 1, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, width)
        self.l4 = nn.Linear(width, width)
        self.l5 = nn.Linear(width, latent_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1)) + h1
        h3 = self.act(self.l3(h2)) + h2
        h4 = self.act(self.l4(h3)) + h3
        return self.l5(h4)

    def zero_init_output(self) -> None:
        nn.init.zeros_(self.l5.weight)
        nn.init.zeros_(self.l5.bias)


class FlowMapModel(nn.Module):
    """Autoencoder-style flow map with a true latent bottleneck.

    Latent dim (16) is strictly less than the hidden layer width (32), so
    the model is not architecturally collapsible to a plain MLP: the (z,dt)
    propagator and the decoder both operate on a compressed representation.
    Encoder/decoder are width-32 MLPs that project to/from R^16.
    Total: 1,712 (enc) + 4,272 (prop) + 1,699 (dec) = 7,683 params.
    """
    def __init__(self, residual: bool = False, latent_dim: int = 16, width: int = 32) -> None:
        super().__init__()
        self.encoder = _mlp([3, width, width, latent_dim])
        self.propagator = TimeRichPropagator(latent_dim=latent_dim, width=width)
        self.decoder = _mlp([latent_dim, width, width, 3])
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x[:, :3]
        dt = x[:, 3:4]
        z = self.encoder(p)
        z_next = self.propagator(torch.cat([z, dt], dim=-1))
        if self.residual:
            z_next = z + z_next
        return self.decoder(z_next)

    def post_init_reset(self) -> None:
        """Zero-init propagator output ONLY for the residual variant.

        Makes the residual variant start as `decoder(encoder(p))` -- the
        skip path is fully informative at t=0 and the propagator must learn
        the dt-correction. Standard residual-block stabilizer.
        """
        if self.residual:
            self.propagator.zero_init_output()


class DeepONetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = _mlp([3, 32, 32, 32, 32, 16])
        self.trunk = _mlp([1, 32, 32, 32, 32, 16])
        self.head = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x[:, :3]
        dt = x[:, 3:4]
        return self.head(self.branch(p) * self.trunk(dt))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def xavier_init_(model: nn.Module) -> None:
    """Apply Xavier-uniform init to all Linear weights; zero biases.

    PyTorch's default `nn.Linear` init is `kaiming_uniform_(a=sqrt(5))` with an
    effective gain of `sqrt(1/3)`, which produces weights too small for SiLU
    stacks of this depth and occasionally lands the optimizer in a bad basin
    (observed: seed-2 MLP stuck near 5e-2 z-MSE under default init while
    seeds 0/1 converged to ~2e-3). Xavier-uniform is wider and more stable.
    Reproducible: uses the current torch RNG, so seeding before construction
    is sufficient.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ====================================================================
# Training
# ====================================================================
def lr_at(step: int, steps_per_epoch: int) -> float:
    epoch = step / max(steps_per_epoch, 1)
    if epoch < WARMUP_EPOCHS:
        warmup_lr = LR * (epoch / WARMUP_EPOCHS) if WARMUP_EPOCHS > 0 else LR
        return max(warmup_lr, LR_FLOOR)
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    progress = min(max(progress, 0.0), 1.0)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def hybrid_loss(
    pred_z: torch.Tensor, target_z: torch.Tensor, y_log_mean: torch.Tensor, y_log_std: torch.Tensor,
) -> torch.Tensor:
    pred_log = pred_z * y_log_std + y_log_mean
    target_log = target_z * y_log_std + y_log_mean
    log_mae = (pred_log - target_log).abs().mean()
    mse_z = (pred_z - target_z).pow(2).mean()
    return LAMBDA_LOG10_MAE * log_mae + LAMBDA_MSE_Z * mse_z


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    s, n = 0.0, 0
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            s += (pred - Y).pow(2).sum().item()
            n += Y.numel()
    return s / max(n, 1)


def evaluate_with_hybrid(
    model: nn.Module, loader: DataLoader, stats: NormStats, device: torch.device,
) -> Tuple[float, float]:
    """Return (mse_z, hybrid) on a loader where hybrid = log10_mae_phys + 0.5 * mse_z."""
    model.eval()
    y_log_std_t = torch.from_numpy(stats.y_log_std).float().to(device)
    z_sq = 0.0
    phys_abs = 0.0
    n_total = 0
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            z_diff = pred - Y
            phys_diff = z_diff * y_log_std_t
            z_sq += z_diff.pow(2).sum().item()
            phys_abs += phys_diff.abs().sum().item()
            n_total += Y.numel()
    mse_z = z_sq / max(n_total, 1)
    log10_mae_phys = phys_abs / max(n_total, 1)
    return mse_z, log10_mae_phys + 0.5 * mse_z


def evaluate_all_metrics(
    model: nn.Module, loader: DataLoader, stats: NormStats, device: torch.device,
) -> Dict[str, float]:
    """Compute every candidate Test-MSE definition the paper might mean.

    Returns mse_z, mae_z, log10_mae_phys, mse_phys_log10, and the hybrid loss
    (log10_mae_phys + 0.5 * mse_z), plus per-species variants. Used by --reeval
    to identify which definition reproduces paper Table 2.
    """
    model.eval()
    y_log_std_t = torch.from_numpy(stats.y_log_std).float().to(device)

    z_sq = z_abs = phys_abs = phys_sq = 0.0
    z_sq_sp = [0.0, 0.0, 0.0]
    phys_abs_sp = [0.0, 0.0, 0.0]
    phys_sq_sp = [0.0, 0.0, 0.0]
    n_total = 0
    n_per_sp = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            z_diff = pred - Y                        # [B, 3]
            phys_diff = z_diff * y_log_std_t         # log10 units (mean cancels)

            z_sq += z_diff.pow(2).sum().item()
            z_abs += z_diff.abs().sum().item()
            phys_abs += phys_diff.abs().sum().item()
            phys_sq += phys_diff.pow(2).sum().item()

            n_total += Y.numel()
            n_per_sp += Y.shape[0]
            for i in range(3):
                z_sq_sp[i] += z_diff[:, i].pow(2).sum().item()
                phys_abs_sp[i] += phys_diff[:, i].abs().sum().item()
                phys_sq_sp[i] += phys_diff[:, i].pow(2).sum().item()

    nt = max(n_total, 1)
    ns = max(n_per_sp, 1)
    mse_z = z_sq / nt
    mae_z = z_abs / nt
    log10_mae_phys = phys_abs / nt
    mse_phys_log10 = phys_sq / nt
    out: Dict[str, float] = {
        "mse_z": mse_z,
        "mae_z": mae_z,
        "log10_mae_phys": log10_mae_phys,
        "mse_phys_log10": mse_phys_log10,
        "hybrid": log10_mae_phys + 0.5 * mse_z,
    }
    for i in range(3):
        out[f"mse_z_x{i+1}"] = z_sq_sp[i] / ns
        out[f"log10_mae_phys_x{i+1}"] = phys_abs_sp[i] / ns
        out[f"mse_phys_log10_x{i+1}"] = phys_sq_sp[i] / ns
    return out


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    stats: NormStats,
    device: torch.device,
    out_dir: Path,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    spe = len(train_loader)
    n_params = count_params(model)
    print(f"\n[{name}] params = {n_params:,}; device = {device}")

    y_log_mean_t = torch.from_numpy(stats.y_log_mean).float().to(device)
    y_log_std_t = torch.from_numpy(stats.y_log_std).float().to(device)

    history: Dict[str, List[float]] = {
        "epoch": [], "train_mse_z": [], "val_mse_z": [],
        "train_hybrid": [], "val_hybrid": [], "lr": [],
    }
    step = 0
    skipped = 0
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc=f"[{name}]", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        sq_sum, cnt = 0.0, 0
        loss_sum = 0.0
        last_lr = LR
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            cur_lr = lr_at(step, spe)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            last_lr = cur_lr
            pred = model(X)
            loss = hybrid_loss(pred, Y, y_log_mean_t, y_log_std_t)
            if not torch.isfinite(loss):
                # Skip non-finite batches (rare MPS hiccup); do not corrupt optimizer state.
                skipped += 1
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            sq_sum += (pred.detach() - Y).pow(2).sum().item()
            loss_sum += loss.detach().item() * Y.numel()
            cnt += Y.numel()
            step += 1
        train_mse_z = sq_sum / max(cnt, 1)
        train_hybrid = loss_sum / max(cnt, 1)
        val_mse_z, val_hybrid = evaluate_with_hybrid(model, val_loader, stats, device)
        history["epoch"].append(epoch)
        history["train_mse_z"].append(train_mse_z)
        history["val_mse_z"].append(val_mse_z)
        history["train_hybrid"].append(train_hybrid)
        history["val_hybrid"].append(val_hybrid)
        history["lr"].append(last_lr)
        epoch_bar.set_postfix(
            tr_hyb=f"{train_hybrid:.3e}",
            va_hyb=f"{val_hybrid:.3e}",
            lr=f"{last_lr:.1e}",
            skip=skipped,
        )

    test_mse_z = evaluate(model, test_loader, device)
    print(f"[{name}] TEST mse_z = {test_mse_z:.4e}  (skipped {skipped} bad batches)")

    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.csv", "w") as f:
        f.write("epoch,train_mse_z,val_mse_z,train_hybrid,val_hybrid,lr\n")
        rows = zip(
            history["epoch"], history["train_mse_z"], history["val_mse_z"],
            history["train_hybrid"], history["val_hybrid"], history["lr"],
        )
        for e, tr_m, va_m, tr_h, va_h, lr in rows:
            f.write(f"{e},{tr_m:.6e},{va_m:.6e},{tr_h:.6e},{va_h:.6e},{lr:.6e}\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"n_params": n_params, "test_mse_z": test_mse_z}, f, indent=2)

    return {"name": name, "n_params": n_params, "test_mse_z": test_mse_z, "history": history}


# ====================================================================
# Figure 3
# ====================================================================
ARCH_COLORS = {
    "MLP": "#DB5217",                # orange (paper Fig 3)
    "Flow-map": "#2753DB",           # blue
    "Residual Flow-map": "#7C4FA1",  # purple
    "DeepONet": "#117733",           # green
}


def plot_figure_3(results: List[Dict]) -> None:
    if STYLE_PATH.exists():
        try:
            plt.style.use(str(STYLE_PATH))
        except Exception:
            pass
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.5))
    ax.axvline(WARMUP_EPOCHS, color="gray", linestyle="--", linewidth=1.0, label="End of warmup")
    for res in results:
        name = res["name"]
        color = ARCH_COLORS.get(name, None)
        h = res["history"]
        lab = f"{name} (MSE={res['test_mse_z']:.2e})"
        ax.plot(h["epoch"], h["train_mse_z"], color=color, linestyle="-", linewidth=1.4, label=lab)
        ax.plot(h["epoch"], h["val_mse_z"], color=color, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")
    ax.set_xlim(0, EPOCHS)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(LOCAL_FIG_PATH)
    print(f"[fig] saved {LOCAL_FIG_PATH}")
    if PAPER_FIG_PATH.parent.exists():
        fig.savefig(PAPER_FIG_PATH)
        print(f"[fig] saved {PAPER_FIG_PATH}")
    plt.close(fig)


def write_table2(results: List[Dict]) -> None:
    with open(TABLE2_PATH, "w") as f:
        f.write("# Robertson architecture benchmark (matches paper Table 2)\n")
        f.write(f"# {'Architecture':<22} {'Params':>10} {'Test MSE (z-scored)':>22}\n")
        for r in results:
            f.write(f"{r['name']:<22} {r['n_params']:>10,} {r['test_mse_z']:>22.4e}\n")
    print(f"[table] saved {TABLE2_PATH}")


REEVAL_CSV_PATH = None  # set in reeval_main from RUNS_DIR
REEVAL_MD_PATH = None


PAPER_TABLE2 = {
    "MLP": 1.38e-3,
    "Flow-map": 1.25e-3,
    "DeepONet": 2.26e-3,
}

REEVAL_METRIC_KEYS = [
    "mse_z",
    "mae_z",
    "log10_mae_phys",
    "mse_phys_log10",
    "hybrid",
]


def write_reeval_outputs(all_metrics: Dict[str, Dict[str, float]]) -> None:
    csv_path = RUNS_DIR / "reeval_metrics.csv"
    md_path = RUNS_DIR / "reeval_metrics.md"

    keys = REEVAL_METRIC_KEYS + [
        f"{m}_x{i+1}" for m in ("mse_z", "log10_mae_phys", "mse_phys_log10") for i in range(3)
    ]
    with open(csv_path, "w") as f:
        f.write("arch," + ",".join(keys) + ",paper_table2\n")
        for arch, m in all_metrics.items():
            row = [arch] + [f"{m[k]:.6e}" for k in keys] + [f"{PAPER_TABLE2.get(arch, float('nan')):.6e}"]
            f.write(",".join(row) + "\n")

    with open(md_path, "w") as f:
        f.write("# Robertson re-evaluation on existing checkpoints\n\n")
        f.write("All values are computed on the test set (deterministic SEED=0 split). "
                "`paper` column shows paper Table 2 \"Test Set MSE Loss\" for direct comparison.\n\n")
        head = ["arch"] + REEVAL_METRIC_KEYS + ["paper"]
        f.write("| " + " | ".join(head) + " |\n")
        f.write("|" + "|".join(["---"] * len(head)) + "|\n")
        for arch, m in all_metrics.items():
            row = [arch] + [f"{m[k]:.3e}" for k in REEVAL_METRIC_KEYS] + [f"{PAPER_TABLE2.get(arch, float('nan')):.3e}"]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n## Per-species log10 MAE (physical units)\n\n")
        head2 = ["arch", "x1", "x2", "x3"]
        f.write("| " + " | ".join(head2) + " |\n|" + "|".join(["---"] * 4) + "|\n")
        for arch, m in all_metrics.items():
            f.write(f"| {arch} | {m['log10_mae_phys_x1']:.3e} | {m['log10_mae_phys_x2']:.3e} | {m['log10_mae_phys_x3']:.3e} |\n")
    print(f"[reeval] wrote {csv_path}")
    print(f"[reeval] wrote {md_path}")


def reeval_existing_checkpoints(device: torch.device) -> int:
    data = generate_or_load_data()
    n_total_traj = data["p1"].shape[0]
    train_idx, val_idx, test_idx = trajectory_split(n_total_traj)

    X_train, Y_train = build_pairs(
        data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], train_idx,
    )
    X_test, Y_test = build_pairs(
        data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], test_idx,
    )
    stats = compute_stats(X_train, Y_train)
    Xte, Yte = apply_norm(X_test, Y_test, stats)
    ds = TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(Yte).float())
    test_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    print(f"[reeval] test pairs: {Xte.shape[0]:,} | device={device}")

    builders = {
        "MLP": MLPModel,
        "Flow-map": lambda: FlowMapModel(residual=False),
        "DeepONet": DeepONetModel,
    }

    all_metrics: Dict[str, Dict[str, float]] = {}
    for name, build in builders.items():
        ckpt_path = RUNS_DIR / name.replace(" ", "_") / "model.pt"
        if not ckpt_path.exists():
            print(f"[reeval] WARN: no checkpoint at {ckpt_path}; skipping {name}")
            continue
        model = build()
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        m = evaluate_all_metrics(model, test_loader, stats, device)
        all_metrics[name] = m
        paper = PAPER_TABLE2.get(name, float("nan"))
        print(
            f"[reeval] {name:<20s}  mse_z={m['mse_z']:.3e}  "
            f"log10_mae_phys={m['log10_mae_phys']:.3e}  "
            f"hybrid={m['hybrid']:.3e}  paper={paper:.3e}"
        )

    write_reeval_outputs(all_metrics)

    # Identify which metric best matches paper Table 2 (smallest mean relative error).
    if all_metrics and all(arch in all_metrics for arch in PAPER_TABLE2):
        best_key = None
        best_score = float("inf")
        for k in REEVAL_METRIC_KEYS:
            rel = [
                abs(all_metrics[arch][k] - PAPER_TABLE2[arch]) / PAPER_TABLE2[arch]
                for arch in PAPER_TABLE2
            ]
            score = sum(rel) / len(rel)
            print(f"[reeval] mean |rel err| vs paper Table 2 under '{k}': {score:.3f}")
            if score < best_score:
                best_score = score
                best_key = k
        print(f"[reeval] best-matching metric for paper Table 2: '{best_key}' (mean rel err {best_score:.3f})")

    return 0


MULTISEED_ROOT = HERE / "runs_multiseed"

SPECIES_COLORS = {"x1": "#2E5BBA", "x2": "#D55E00", "x3": "#009E73"}


def plot_figure_3_multiseed(
    seeds: List[int],
    device: torch.device,
    aggregated: Dict[str, Dict[str, List[float]]],
    max_traj: int = None,  # accepted for caller compatibility; unused here
) -> None:
    """Single-panel training-curves Fig 3 (square aspect).

    Per-seed thin train curves + bold median train curve + dashed median val
    curve per architecture. X axis spans the actual number of trained epochs
    (read from metrics.csv) rather than the EPOCHS constant. Legend reports
    median test log10_mae_phys with the across-seed range.
    """
    del max_traj  # unused; kept for signature stability
    if STYLE_PATH.exists():
        try:
            plt.style.use(str(STYLE_PATH))
        except Exception:
            pass

    names = ["MLP", "Flow-map", "DeepONet"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.axvline(WARMUP_EPOCHS, color="gray", linestyle="--", linewidth=1.0, label="End of warmup")

    max_epoch_seen = 0
    using_hybrid_y = None  # True if csv has hybrid columns; warn once if not
    for name in names:
        train_curves: List[np.ndarray] = []
        val_curves: List[np.ndarray] = []
        epochs_arr: np.ndarray = None
        for seed in seeds:
            csv_path = MULTISEED_ROOT / f"seed{seed:02d}" / name.replace(" ", "_") / "metrics.csv"
            if not csv_path.exists():
                continue
            ep, tr, va = [], [], []
            with open(csv_path) as f:
                header = [h.strip() for h in next(f).strip().split(",")]
                has_hybrid = "train_hybrid" in header and "val_hybrid" in header
                if using_hybrid_y is None:
                    using_hybrid_y = has_hybrid
                    if not has_hybrid:
                        print(f"[fig] WARN: {csv_path.name} has no train_hybrid/val_hybrid; falling back to mse_z on y-axis")
                if has_hybrid:
                    # Derive log10_mae_phys per epoch from hybrid = log10_mae + 0.5*mse_z.
                    # Plotting log10_mae directly keeps the y-axis consistent with the
                    # legend (test log10-MAE) and with paper Table 2.
                    i_hyb_tr = header.index("train_hybrid")
                    i_hyb_va = header.index("val_hybrid")
                    i_mse_tr = header.index("train_mse_z")
                    i_mse_va = header.index("val_mse_z")
                else:
                    i_tr, i_va = header.index("train_mse_z"), header.index("val_mse_z")
                for line in f:
                    parts = line.strip().split(",")
                    ep.append(int(parts[0]))
                    if has_hybrid:
                        tr.append(float(parts[i_hyb_tr]) - 0.5 * float(parts[i_mse_tr]))
                        va.append(float(parts[i_hyb_va]) - 0.5 * float(parts[i_mse_va]))
                    else:
                        tr.append(float(parts[i_tr]))
                        va.append(float(parts[i_va]))
            epochs_arr = np.array(ep)
            train_curves.append(np.array(tr))
            val_curves.append(np.array(va))
        if not train_curves:
            continue

        color = ARCH_COLORS.get(name, None)
        mae_vals = np.array(aggregated.get(name, {}).get("log10_mae_phys", []), dtype=float)
        if mae_vals.size:
            mae_str = f"{np.median(mae_vals):.2e}"
        else:
            mae_str = "n/a"
        lab = f"{name}: test loss = {mae_str}"

        train_arr = np.stack(train_curves)
        val_arr = np.stack(val_curves)
        train_med = np.median(train_arr, axis=0)
        val_med = np.median(val_arr, axis=0)
        ax.plot(epochs_arr, train_med, color=color, linestyle="-", linewidth=1.8, label=lab)
        ax.plot(epochs_arr, val_med, color=color, linestyle="--", linewidth=1.1, alpha=0.85)
        max_epoch_seen = max(max_epoch_seen, int(epochs_arr.max()))

    ax.set_xlabel("Epoch")
    if using_hybrid_y:
        ax.set_ylabel(r"log$_{10}$ mean absolute error")
    else:
        ax.set_ylabel(r"Per-epoch MSE in z-scored space")
    ax.set_yscale("log")
    if max_epoch_seen > 0:
        ax.set_xlim(0, max_epoch_seen)
    ax.legend(loc="upper right", fontsize=13, frameon=False)

    fig.tight_layout()
    fig.savefig(LOCAL_FIG_PATH)
    print(f"[fig] saved {LOCAL_FIG_PATH}")
    if PAPER_FIG_PATH.parent.exists():
        fig.savefig(PAPER_FIG_PATH)
        print(f"[fig] saved {PAPER_FIG_PATH}")
    plt.close(fig)


def write_table2_multiseed(
    aggregated: Dict[str, Dict[str, List[float]]],
    n_params_by_arch: Dict[str, int],
) -> None:
    """Rewrite table2.txt using log10_mae_phys (mean +/- std across seeds)."""
    with open(TABLE2_PATH, "w") as f:
        f.write("# Robertson architecture benchmark (multi-seed, paper Table 2 metric)\n")
        f.write(f"# Metric: log10_mae_phys = mean(|log10(pred) - log10(truth)|) on test set\n")
        f.write(f"# Reported as mean +/- std across seeds; paper Table 2 in last column.\n")
        f.write(
            f"# {'Architecture':<22} {'Params':>8} {'Test log10-MAE (mean)':>24} "
            f"{'(std)':>12} {'Paper Table 2':>16}\n"
        )
        for arch, met in aggregated.items():
            vals = np.array(met.get("log10_mae_phys", []), dtype=float)
            mean = vals.mean() if vals.size else float("nan")
            std = vals.std(ddof=0) if vals.size else float("nan")
            params = n_params_by_arch.get(arch, 0)
            paper = PAPER_TABLE2.get(arch, float("nan"))
            f.write(
                f"{arch:<22} {params:>8,} {mean:>24.4e} {std:>12.2e} {paper:>16.4e}\n"
            )
    print(f"[table] saved {TABLE2_PATH}")


def write_multiseed_summary(
    aggregated: Dict[str, Dict[str, List[float]]], seeds: List[int],
) -> None:
    MULTISEED_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = MULTISEED_ROOT / "summary.csv"
    summary_md = MULTISEED_ROOT / "summary.md"

    keys = REEVAL_METRIC_KEYS

    with open(summary_csv, "w") as f:
        head = ["arch"]
        for k in keys:
            head += [f"{k}_mean", f"{k}_std"]
        head.append("paper_table2")
        f.write(",".join(head) + "\n")
        for arch, met in aggregated.items():
            row = [arch]
            for k in keys:
                vals = np.array(met.get(k, []), dtype=float)
                if vals.size:
                    row += [f"{vals.mean():.6e}", f"{vals.std(ddof=0):.6e}"]
                else:
                    row += ["nan", "nan"]
            row.append(f"{PAPER_TABLE2.get(arch, float('nan')):.6e}")
            f.write(",".join(row) + "\n")

    with open(summary_md, "w") as f:
        f.write(f"# Multi-seed Robertson benchmark (seeds: {seeds})\n\n")
        f.write("Values: mean +/- std across seeds. `paper` = paper Table 2 \"Test Set MSE Loss\".\n\n")
        head = ["arch"] + keys + ["paper"]
        f.write("| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n")
        for arch, met in aggregated.items():
            row = [arch]
            for k in keys:
                vals = np.array(met.get(k, []), dtype=float)
                if vals.size:
                    row.append(f"{vals.mean():.3e} +/- {vals.std(ddof=0):.2e}")
                else:
                    row.append("n/a")
            row.append(f"{PAPER_TABLE2.get(arch, float('nan')):.3e}")
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n## Per-seed log10_mae_phys (test set)\n\n")
        f.write("| arch | " + " | ".join(f"seed {s}" for s in seeds) + " |\n")
        f.write("|" + "|".join(["---"] * (1 + len(seeds))) + "|\n")
        for arch, met in aggregated.items():
            vals = met.get("log10_mae_phys", [])
            cells = [f"{v:.3e}" for v in vals]
            f.write(f"| {arch} | " + " | ".join(cells) + " |\n")

    print(f"[multiseed] wrote {summary_csv}")
    print(f"[multiseed] wrote {summary_md}")


def multiseed_main(seeds: List[int], device: torch.device, max_traj: int = None) -> int:
    MULTISEED_ROOT.mkdir(parents=True, exist_ok=True)
    data = generate_or_load_data()
    if max_traj is not None and max_traj < data["p1"].shape[0]:
        for k in ("p1", "p2", "p3"):
            data[k] = data[k][:max_traj]
        data["ys"] = data["ys"][:max_traj]
        print(f"[data] subsetting to first {max_traj} trajectories")
    n_total_traj = data["p1"].shape[0]

    builders = {
        "MLP": MLPModel,
        "Flow-map": lambda: FlowMapModel(residual=False),
        "DeepONet": DeepONetModel,
    }
    names = list(builders.keys())

    aggregated: Dict[str, Dict[str, List[float]]] = {name: {} for name in names}

    for s_idx, seed in enumerate(seeds):
        print(f"\n========== Seed {seed} ({s_idx+1}/{len(seeds)}) ==========")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_idx, val_idx, test_idx = trajectory_split(n_total_traj, seed=seed)
        X_train, Y_train = build_pairs(
            data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], train_idx,
        )
        X_val, Y_val = build_pairs(
            data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], val_idx,
        )
        X_test, Y_test = build_pairs(
            data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], test_idx,
        )
        stats = compute_stats(X_train, Y_train)
        Xtr, Ytr = apply_norm(X_train, Y_train, stats)
        Xva, Yva = apply_norm(X_val, Y_val, stats)
        Xte, Yte = apply_norm(X_test, Y_test, stats)

        def make_loader(X: np.ndarray, Y: np.ndarray, shuffle: bool) -> DataLoader:
            ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
            return DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
                pin_memory=(device.type == "cuda"), drop_last=shuffle,
            )

        train_loader = make_loader(Xtr, Ytr, shuffle=True)
        val_loader = make_loader(Xva, Yva, shuffle=False)
        test_loader = make_loader(Xte, Yte, shuffle=False)

        for name in names:
            out_dir = MULTISEED_ROOT / f"seed{seed:02d}" / name.replace(" ", "_")
            ckpt = out_dir / "model.pt"
            if ckpt.exists():
                print(f"[multiseed] seed={seed} {name}: existing checkpoint found at {ckpt}; skipping training")
            else:
                torch.manual_seed(seed)
                model = builders[name]()
                xavier_init_(model)
                if hasattr(model, "post_init_reset"):
                    model.post_init_reset()
                train_one_model(name, model, train_loader, val_loader, test_loader, stats, device, out_dir)

            model_eval = builders[name]()
            model_eval.load_state_dict(torch.load(ckpt, map_location=device))
            model_eval.to(device)
            metrics = evaluate_all_metrics(model_eval, test_loader, stats, device)
            for k, v in metrics.items():
                aggregated[name].setdefault(k, []).append(v)
            with open(out_dir / "metrics_all.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"[multiseed] seed={seed} {name:<20s}  "
                f"log10_mae_phys={metrics['log10_mae_phys']:.3e}  "
                f"mse_z={metrics['mse_z']:.3e}"
            )

        # Write running summary after each seed in case the job is interrupted.
        write_multiseed_summary(aggregated, seeds[: s_idx + 1])

    n_params_by_arch = {name: count_params(builders[name]()) for name in names}
    write_table2_multiseed(aggregated, n_params_by_arch)
    try:
        plot_figure_3_multiseed(seeds, device, aggregated, max_traj=max_traj)
    except Exception as e:
        print(f"[fig] WARN: figure generation failed: {e!r}")

    return 0


def load_existing_history(name: str) -> Dict:
    out_dir = RUNS_DIR / name.replace(" ", "_")
    csv_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"
    if not csv_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"No saved run for '{name}'. Train first.")
    with open(summary_path) as f:
        meta = json.load(f)
    history: Dict[str, List[float]] = {
        "epoch": [], "train_mse_z": [], "val_mse_z": [],
        "train_hybrid": [], "val_hybrid": [], "lr": [],
    }
    with open(csv_path) as f:
        header = [h.strip() for h in next(f).strip().split(",")]
        idx = {col: header.index(col) for col in header}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            history["epoch"].append(int(parts[idx["epoch"]]))
            history["train_mse_z"].append(float(parts[idx["train_mse_z"]]))
            history["val_mse_z"].append(float(parts[idx["val_mse_z"]]))
            history["lr"].append(float(parts[idx["lr"]]))
            if "train_hybrid" in idx:
                history["train_hybrid"].append(float(parts[idx["train_hybrid"]]))
                history["val_hybrid"].append(float(parts[idx["val_hybrid"]]))
    return {"name": name, "n_params": meta["n_params"], "test_mse_z": meta["test_mse_z"], "history": history}


# ====================================================================
# Main
# ====================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-train", action="store_true", help="Skip training; plot from existing runs/")
    parser.add_argument(
        "--reeval", action="store_true",
        help="Re-evaluate existing checkpoints under all candidate Test-MSE definitions; no training",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Multi-seed sweep: train all 4 archs once per listed seed. Writes runs_multiseed/.",
    )
    parser.add_argument(
        "--max-traj", type=int, default=None,
        help="Subset to the first N trajectories from the cache (no regen). For faster sweeps.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the training-epoch count (default 250).",
    )
    args = parser.parse_args()

    global EPOCHS
    if args.epochs is not None:
        EPOCHS = args.epochs
        print(f"[config] EPOCHS overridden -> {EPOCHS}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    names = ["MLP", "Flow-map", "DeepONet"]

    if args.reeval:
        return reeval_existing_checkpoints(device)

    if args.seeds is not None:
        return multiseed_main(args.seeds, device, max_traj=args.max_traj)

    if args.skip_train:
        results = [load_existing_history(n) for n in names]
        plot_figure_3(results)
        write_table2(results)
        return 0

    RUNS_DIR.mkdir(exist_ok=True)
    data = generate_or_load_data()
    n_total = data["p1"].shape[0]
    train_idx, val_idx, test_idx = trajectory_split(n_total)
    print(f"split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)} trajectories")

    print("[pairs] building (X, Y) pairs ...")
    X_train, Y_train = build_pairs(data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], train_idx)
    X_val,   Y_val   = build_pairs(data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], val_idx)
    X_test,  Y_test  = build_pairs(data["p1"], data["p2"], data["p3"], data["t_eval"], data["ys"], test_idx)
    stats = compute_stats(X_train, Y_train)
    Xtr, Ytr = apply_norm(X_train, Y_train, stats)
    Xva, Yva = apply_norm(X_val, Y_val, stats)
    Xte, Yte = apply_norm(X_test, Y_test, stats)
    print(f"[pairs] train={Xtr.shape[0]:,}  val={Xva.shape[0]:,}  test={Xte.shape[0]:,}")

    def make_loader(X: np.ndarray, Y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=shuffle,
        )

    train_loader = make_loader(Xtr, Ytr, shuffle=True)
    val_loader = make_loader(Xva, Yva, shuffle=False)
    test_loader = make_loader(Xte, Yte, shuffle=False)

    builders = {
        "MLP": MLPModel,
        "Flow-map": lambda: FlowMapModel(residual=False),
        "DeepONet": DeepONetModel,
    }
    results: List[Dict] = []
    for name in names:
        torch.manual_seed(SEED)  # reset for reproducible init across architectures
        model = builders[name]()
        xavier_init_(model)
        if hasattr(model, "post_init_reset"):
            model.post_init_reset()
        out_dir = RUNS_DIR / name.replace(" ", "_")
        results.append(train_one_model(name, model, train_loader, val_loader, test_loader, stats, device, out_dir))

    plot_figure_3(results)
    write_table2(results)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
