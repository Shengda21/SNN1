English | [中文](README_zh.md)

# Code release: Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics

This directory holds the training, analysis, simulation, and figure code of the paper and its supplementary material.
The supplementary material refers to file names only through this README.

## Environment

Ubuntu 24.04.4, Python 3.12.3, PyTorch 2.13.0 (CUDA 13.0), MuJoCo 3.15.0, one NVIDIA RTX 3090; the CPU latencies of Table II are measured on one thread of an Intel Xeon W-2245.
Random seeds: data split 2025; training seeds 0-4 (D-M, B-L, B-A, B-N); `1000 + 10*s + i` for member `i` of seed dictionary `s`; `2000 + s` for snapshot runs.
Condition seeds are in `sarcos/data.py` and `sim/data.py`. The noise seeds of the closed-loop run E13 are fixed in `run_closedloop_aug.py` (N 900, O1-M 901, P1 902).

## Layout

| Path | Content |
|---|---|
| `common/` | surrogates, spiking models, training loop, calibration rules W-S/W-E/W-B/W-D/W-Q(k), metrics |
| `sarcos/data.py`, `sim/data.py` | datasets, splits, operating conditions N, O1-L/M/H, O2-B, P1 |
| `sim/gen_data.py`, `sim/panda_env.py` | MuJoCo Panda record generation and the batched closed-loop environment |
| `run_train.py` | training of all members and of the augmented, early-checkpoint, and reduced-capacity members |
| `dump_preds.py` | frozen per-member predictions on calibration and test sets |
| `launch_*.sh`, `post_*.sh`, `rerun_aug.sh`, `throttle.sh` | batch launchers, post-processing, and a load limiter for a shared machine |
| `figures/` | table and figure builders (`make_all.sh` regenerates all of them) |
| `physical/e08.py` | acquisition example for the 2-DOF physical recording |

## Experiments and result files

Result files are written to `results/<platform>/`, with `<platform>` being `sarcos` or `sim`.

| Experiment | What it computes | Script | Result files |
|---|---|---|---|
| E00 | identity checks of the simulated records; frozen development decisions (widths, strong surrogate, conditions) | `sim/gen_data.py`, `dev_decisions.py` | `runs_frozen/e00_checks.json`, `runs_frozen/frozen_<platform>.json` |
| E01 | test nMSE of every method and condition (Tables II, S2, S3) | `analysis_openloop.py`, `summarize_results.py` | `e01_summary.csv`, `e01_test_table.csv`, `calibration_weights.json`, `unit_nmse.npz` |
| E02 | residual geometry and the two-member witness | `analysis_openloop.py` | `e02_geometry.json` |
| E03 | calibration-budget `m` slice | `analysis_openloop.py` | `e03_m_slice.csv` |
| E04 | executed-member `k` slice and first-order optimality residuals (column `kkt`) | `analysis_openloop.py` | `e04_k_slice.csv` |
| E05 | per-joint error and absolute adequacy | `analysis_openloop.py` | `e05_boundary.csv` |
| E06 | closed-loop tracking under a shared feedback law | `run_closedloop.py` | `e06_episodes.csv`, `e06_config.json`, `e06_traces.npz` |
| E07 | density of gains | `analysis_openloop.py` | `e07_density.csv` |
| E08 | physical recording (supplement S5), arguments select split, normalization, and epochs | `physical/e08.py` | `results/physical/e08*.json` |
| E09 | attenuation controls: null candidate, estimator, augmented dictionaries | `analysis_aug.py`, `frontend_preds.py` | `e09_rows.csv`, `e09_summary.csv`, `e09_paired.csv`, `e09_weights.csv` |
| E10 | propagated and nominal parts of the residual | `analysis_aug.py` | `e10_members.csv`, `e10_dicts.csv` |
| E11 | estimator geometry and `m` slice | `frontend_preds.py`, `frontend_mslice.py` | `e11_frontend_geometry.json`, `e11_m_slice.csv` |
| E12 | forward perturbation response of every frozen member | `forward_response.py` | `forward_response.json`, `forward_response_aug.json` |
| E13 | closed loop with the attenuation candidates | `run_closedloop_aug.py`, `summarize_e13.py` | `e13_episodes_a.csv`, `e13_episodes_b.csv`, `e13_summary.csv`, `e13_paired.csv` |
| E13b | slow part of the open-loop residual with weights fitted on the full calibration set (reference only) | `analysis_lowfreq.py` | `e13b_lowfreq.csv` |
| E13c | total and slow part of the open-loop residual of every E13 feedforward model under the weights of the closed-loop run (ten trajectories per calibration group, estimator variances of the same group); Fig. 4b, c, Fig. 1e, Table S-XI, and the rank correlations use this file | `analysis_lowfreq_matched.py` | `e13c_lowfreq_matched.csv`; `e13_rank_correlations.json` from `figures/build_e13_tables.py` (also holds the full-calibration reference and per-seed values) |
| E14 | poor fit or surrogate: early checkpoints and reduced capacity | `analysis_underfit.py`, `run_train.py` | `e14_curve.csv`, `e14_rows.csv`, `e14_summary.csv`, `e14_paired.csv`, `e14_weights.csv`; `results/e14_numbers.json` |
| theory | bound constants of the finite-calibration certificate | `analysis_openloop.py` | `results/sim/theory_quantities.json` |

`results/results_manifest.csv` maps every number of the paper to its source run.
Per-member configurations (`config.json`) hold the mean spike counts per sample and layer, the epochs, the wall-clock time, and the development error.

## Regenerating tables and figures

From the repository root, `bash code/figures/make_all.sh` rebuilds Tables II and III, the tables of S1, S2, S3, S5, S8, and S9, Figs. 1-4, and the supplementary figures (`fig5.pdf`-`fig7.pdf`) from the frozen results; the output goes to `build/`.
`bash code/figures/make_all.sh figs` rebuilds the figures only. The interpreter is `$PY` (default `python3`).

## Notes

- The launch and post-processing scripts run from `code/` and expect `runs/`, `logs/`, `results/`, `data_sim/`, and `mujoco_menagerie/` next to it, with the SARCOS files in the parent directory; the interpreter is `$PY` (default `python3`). `throttle.sh` reads its concurrency limit from `logs/throttle_n`.

- E09-E12, E13, and E14 are post-hoc analyses on the frozen members, calibration sets, and test sets. They explain the results of P1-P4 and do not test them.
- The noise seed of E06 was derived from a string hash and is not reproducible across processes; E13 uses fixed seeds and reruns the baselines.
- Table II costs (`figures/tables.py`) are measured on the simulated-arm runs. The training column is the wall-clock time recorded in `config.json`, averaged over the five training seeds, of the deployed model or dictionary: for B-F the one `dm_sigmoid_*` run per seed (9.4 min), for D-S and D-M the five runs of the dictionary, for D-T the one run that yields the snapshots. Selecting the surrogate on the development set trained the five D-M members once (36.1 min); this cost is shared by B-F, B-A, D-S, and D-T and is stated in the table caption, not in the column. Execution is the measured batch-1 latency of the executed members on the RTX 3090 and on one CPU thread (`results/runtime/runtime.json`).
- Training times of the members added in E09-E14 depend on concurrency and throttling and should not be used for cost comparisons.
