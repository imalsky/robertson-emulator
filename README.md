# instructions.txt

This folder reproduces Section 3.4 / Figure 3 / Table 2 of the Chemulator ApJ
paper: a 4-architecture benchmark on the stiff Robertson ODE.

It is a PyTorch implementation (matches the rest of the Chemulator codebase).
The previous JAX/Diffrax sandbox has been replaced because it used a
different experimental setup (fixed canonical rates, sampled initial state,
2 architectures) that did not match the paper.

──────────────────────────────────────────────────────────────────────────────
1) Problem setup (matches paper Eq. 5, L353, Table 2)
──────────────────────────────────────────────────────────────────────────────
Robertson ODE:
    dx1/dt = -p1*x1 + p3*x2*x3
    dx2/dt =  p1*x1 - p3*x2*x3 - p2*x2**2
    dx3/dt =  p2*x2**2

Fixed initial state:
    x(t0) = [1, 1e-30, 1e-30]

Reaction rates sampled log-uniformly PER TRAJECTORY:
    p1 in [2e-3, 6e-3]
    p2 in [1.5e7, 3.5e7]
    p3 in [5e3, 6e4]

10,000 trajectories. Each trajectory is evaluated at 100 log-spaced times
spanning t = 1e-5 to 1e5 s. The integrator is Radau IIA (5th order,
stiffly stable) via scipy.integrate.solve_ivp(method='Radau').

──────────────────────────────────────────────────────────────────────────────
2) Models (paper Table 2; ~7,650-7,683 parameters each)
──────────────────────────────────────────────────────────────────────────────
All three models take (p1, p2, p3, dt) as input and predict x(t0 + dt) in R^3.
SiLU activation throughout.

    MLP                 [4, 32, 32, 32, 32, 32, 32, 32, 32, 3]      7,651
    Flow-map            encoder    [3, 32, 32, 16]
                        propagator [17, 32, 32, 32, 32, 16] (3 skips
                                    between matched-width hidden layers)
                        decoder    [16, 32, 32, 3]
                        latent dim = 16  (strictly < layer width 32) 7,683
    DeepONet            branch [3, 32, 32, 32, 32, 16]
                        trunk  [1, 32, 32, 32, 32, 16]
                        head   Linear(16, 3)                          7,635

The flow-map latent dim is intentionally smaller than the hidden width.
With latent_dim = width, encoder + propagator + decoder is functionally
just a deeper MLP (no information bottleneck), so the architectural
distinction from the plain MLP baseline collapses.

The residual variant (z_next = z + F(z, dt)) of the flow-map is still
implemented in `FlowMapModel(residual=True)` but is not part of the
default benchmark; it was dropped after the Nz=16 redesign.

──────────────────────────────────────────────────────────────────────────────
3) Normalization (paper L353)
──────────────────────────────────────────────────────────────────────────────
- All inputs (rates and dt) are log10-transformed using train statistics.
- log10(rates) z-scored.
- log10(dt) scaled to [0, 1] using train-set min/max.
- Outputs y = log10(x + eps) with eps = 1e-30, then z-scored.

──────────────────────────────────────────────────────────────────────────────
4) Training
──────────────────────────────────────────────────────────────────────────────
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4, min_lr=5e-7.
- Schedule: linear warmup over 10 epochs, then cosine annealing.
- Batch size 1024, 250 epochs.
- Loss: log10-MAE in physical units + 0.5 * MSE in z-scored space
  (same hybrid loss as the main Chemulator model).
- Splits: 80/10/10 train/val/test, sampled at the trajectory level
  (no leakage across the three sets).

──────────────────────────────────────────────────────────────────────────────
5) Run
──────────────────────────────────────────────────────────────────────────────
    python -u train_robertson.py                  # data + 4 models + Fig 3
    python -u train_robertson.py --skip-train     # rebuild Fig 3 from runs/

Data generation is parallelized across CPU cores (10,000 stiff Radau IIA
integrations). Training defaults to GPU if available, otherwise MPS (Apple
Silicon) or CPU. Total wall time on a recent laptop: roughly 30 min for
data + 10 min per model.

──────────────────────────────────────────────────────────────────────────────
6) Outputs
──────────────────────────────────────────────────────────────────────────────
    robertson_paper_data.npz           cached trajectories
    runs/<arch>/metrics.csv            per-epoch train/val mse_z and lr
    runs/<arch>/model.pt               trained weights
    runs/<arch>/summary.json           {n_params, test_mse_z}
    Fig3.pdf                           training curves (also copied to
                                       ../../Chemulator_ApJ/Fig3.pdf if that
                                       directory exists)
    table2.txt                         architecture / params / test MSE
                                       (formatted for paper Table 2)
