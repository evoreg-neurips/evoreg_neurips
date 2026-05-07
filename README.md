# EvoReg: Versatile and Robust Point Cloud Registration via Multi-Stage Alignment

This repository contains the official anonymous code release for **EvoReg**, a
unified multi-stage coarse-to-fine pipeline that covers all four (rigid /
non-rigid) x (supervised / self-supervised) point-cloud registration settings
in a single architecture through staged decoupling.

## Method overview

EvoReg combines four stages into a coarse-to-fine cascade, each inheriting a
progressively tighter initialization and solving a progressively easier
subproblem:

- **Stage 0** -- Gradient-free CMA-ES pre-alignment over SE(3), giving a
  pose-agnostic initialization that escapes local minima.
- **Stage 1** -- Iterative Sinkhorn soft correspondences with
  confidence-weighted Kabsch refinement, tightening the rigid pose.
- **Stage 2** -- A residual MLP head correcting remaining rigid error.
- **Stage 3** -- A conditional VAE predicting a residual non-rigid deformation
  field on the rigidly aligned source, trained jointly with a
  diffusion-based score-matching objective.

At inference, four optional training-free modules compose selectively to trade
compute for accuracy without retraining:

1. Concentrated gradient-free SE(3) search (Stage~3 NIA)
2. Point-space diffusion denoising
3. Global Sinkhorn-EMD test-time optimization on the predicted (R, t)
   recovered via Kabsch
4. Per-point Sinkhorn-EMD test-time optimization on point displacements

## What's included

- **Source code** for the complete EvoReg pipeline -- model, data loaders,
  losses, evaluation metrics, training and evaluation entry points.
- **Baseline integrations and wrappers** for fifteen comparison methods:
  CPD (rigid and non-rigid), BCPD, NDP, DefTransNet, FLOT, ICP, FGR, RANSAC,
  GeoTransformer, PRNet, DCP, RPMNet, PointNetLK, DeepGMR, and iPCRNet.
- **Statistical significance utilities** -- paired Wilcoxon signed-rank tests
  (McNemar for binary recall outcomes), bootstrap 95% confidence intervals,
  and Holm-Bonferroni multiple-comparison correction.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The codebase is tested with Python 3.10+ and PyTorch 2.x.

## Datasets

The training and evaluation pipelines expect a top-level `data/` directory
populated by the user:

```
data/ModelNet40/
data/ShapeNetV1PointCloud/
data/MPI-FAUST/training/registrations/
data/3DMatch_test/data/test/
```

These datasets are publicly available from their respective providers.

## Training

EvoReg is trained on ModelNet40 with four variants (rigid / non-rigid x
supervised / self-supervised). A representative training command:

```bash
python3 train_evoreg.py \
  --dataset modelnet40 --train_data_dir data/ModelNet40 --val_data_dir data/ModelNet40 \
  --epochs 200 --batch_size 16 --lr 1e-3 --n_points 1024 --n_samples 9843 \
  --use_coarse_to_fine --use_rigid_head --use_correspondence_loss \
  --with_diffusion --diffusion_steps 1000 --diffusion_weight 1.0 \
  --use_pso --nia_type cmaes --pso_particles 50 --pso_iterations 30 \
  --use_inter_stage_nia \
  --inter_stage_nia_rot_s1 20.0 --inter_stage_nia_trans_s1 0.5 \
  --inter_stage_nia_rot_s2 10.0 --inter_stage_nia_trans_s2 0.2 \
  --inter_stage_nia_rot_s3  5.0 --inter_stage_nia_trans_s3 0.1 \
  --use_lr_schedule --lr_schedule_step 50 --lr_schedule_gamma 0.5 \
  --use_occlusion --occlusion_type dropout --occlusion_ratio 0.2 --occlusion_prob 0.5 \
  --rotation_range 45 --translation_range 1.0 --normalization UnitBall \
  --checkpoint_dir checkpoints
```

Add `--use_rt_supervision` for supervised variants and `--non_rigid` for
non-rigid variants.

## Evaluation

Run baselines and EvoReg on a chosen dataset:

```bash
python3 evaluate_baselines.py \
  --dataset modelnet40 --dir data/ModelNet40 \
  --recall_threshold 0.15 --normalization UnitBall \
  --rotation_range 45 --translation_range 0.2 \
  --n_points 1024 --n_samples 2468 \
  --models evoreg_c2f --model_checkpoint <path/to/checkpoint.pth>
```

Compose any of the four inference-time modules with `--use_eval_nia_s3`,
`--use_diffusion_refinement`, `--use_sinkhorn_tto`, `--use_kabsch`, and
`--use_sinkhorn_tto_rigid` (see `evaluate_baselines.py --help` for the full
flag list and per-module hyperparameters).

## Statistical Significance

The per-baseline, per-metric significance tables reported in the paper can be
reproduced via:

```bash
python3 statistical_tests.py --results <list-of-eval-jsons> --reference <method-name>
```

## Repository Layout

```
evoreg/                   EvoReg model, data loaders, losses, evaluation metrics
baselines/                Baseline integrations and wrappers (15 methods)
train_evoreg.py           EvoReg training entry point
evaluate_baselines.py     Unified evaluation pipeline
statistical_tests.py      Bootstrap CIs and paired significance tests
```

## License

MIT License. See `LICENSE`.
