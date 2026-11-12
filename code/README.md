English | [中文](README_zh.md)

# Code release: Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter

This directory holds the training, analysis, simulation, and figure code of the paper and its supplementary material.
It extends the earlier release (experiments E00-E14: `common/`, `sarcos/`, `sim/`, `run_*.py`, `analysis_*.py`, and so on) with the experiments **E15-E29**. The new files carry the prefixes `e15_` to `e29_` and `e2x_`, plus `theory_selfcheck.py`. No file of the earlier release is modified; the new scripts import its modules (`common/`, `sim/`, `sarcos/`) and read its frozen members.
The supplementary material refers to file names only through this README.

## Environment

Ubuntu 24.04.4, Python 3.12.3, PyTorch 2.13.0 (CUDA 13.0), MuJoCo 3.15.0, one NVIDIA RTX 3090; the CPU latencies of Table II are measured on one thread of an Intel Xeon W-2245.

Random seeds of the earlier release: data split 2025; training seeds 0-4 (D-M, B-L, B-A, B-N); `1000 + 10*s + i` for member `i` of seed dictionary `s`; `2000 + s` for snapshot runs.
Condition seeds are in `sarcos/data.py` and `sim/data.py`. The noise seeds of the closed-loop run E13 are fixed in `run_closedloop_aug.py` (N 900, O1-M 901, P1 902); E28 and E29 reuse them (`e27_common.NOISE_SEED`).

Seed rules of the new experiments:

- **E16-E18 members** (`w16_*`, `w16d_*`, `w16c_*`, `w17_*`, `w18_*`) use the training seeds 0, 1, 2 directly as the torch seed: `set_seed(seed)` is called before the model is built, so a new member has the same initialization and the same batch order as the earlier member of the same seed (for example `w16_tri_h0.25_s0` and `dm_triangular_s0`). The queue as run holds seeds 0-2; the ultra-wide `w16_*` members (HWHM 8-64) and the control members `w16c_*` have seed 0 only. Further seeds are added with `e16_tasks.py --extra "w16_tri_h1:3,4"`.
- **Detach variant** (`w16d_*`, including `w16d_ste_*`): the forward pass is unchanged; the subtractive reset is computed as `mem - spk.detach() * thr`, so the reset does not multiply the temporal credit by `(1 - thr * g(u))`. The threshold still receives its gradient through the reset. The initialization is identical to `common.models.SNN`.
- **E15**: the forward-response perturbation uses `rng(0)` and the same 2000 development rows as `forward_response.py`; the direction-resolved responses use `rng(15)`, shared by all members; the stationarity check uses 16,384 training rows drawn with seed 8 (`--seed`).
- **E21-E26**: the paired-noise grid draws one standard-normal array per role (seeds `dev` 2901, `cal` 2902, `test` 2903, `cal_P1` 2904, `test_P1` 2905; `e2x_common.XI_SEED`) and scales it by `rho`. The conditions of the earlier release are regenerated with `apply_condition` and are identical to the stored predictions. Frozen members: seeds 0-4. Cluster bootstrap over training seed and test unit, 1000 resamples by default.
- **E27-E29**: calibration uses the groups 0-2 of 10 trajectories (as E13); the frozen members have seeds 0-4. The external-torque records of E27 use the reference trajectories of the test role (seed 400), the event seed 2701, and the observation-noise seed 2702; the intention trajectories of E28 use seed 2801.

## Layout

Paths are relative to this directory unless stated otherwise. The scripts expect `runs25/` (the frozen members of the earlier release), `runs/` (the new E16-E18 members), `data_sim/`, `mujoco_menagerie/`, `logs/`, and `results/` next to `code/`, and the SARCOS files in the parent directory. `runs_frozen/` holds `config.json` and `history.json` of every member, including the E16-E18 members, whose weights are not distributed (see below).

| Path | Content |
|---|---|
| `common/` | surrogates, spiking models, training loop, calibration rules W-S/W-E/W-B/W-D/W-Q(k), metrics |
| `sarcos/data.py`, `sim/data.py` | datasets, splits, operating conditions N, O1-L/M/H, O2-B, P1 |
| `sim/gen_data.py`, `sim/panda_env.py` | MuJoCo Panda record generation and the batched closed-loop environment |
| `run_train.py` | training of all members and of the augmented, early-checkpoint, and reduced-capacity members |
| `dump_preds.py` | frozen per-member predictions on calibration and test sets |
| `launch_*.sh`, `post_*.sh`, `rerun_aug.sh`, `throttle.sh` | batch launchers, post-processing, and a load limiter for a shared machine |
| `figures/` | table and figure builders in two groups. Locked protocol (E00-E14), run by `make_all.sh`: `tables.py`, `fig1.py`-`fig6.py`, `build_*.py`. Revised paper (E15-E29), run by `make_all_revised.sh`: `common_style.py`, `fig1_overview.py`, `fig2_plane.py`, `fig3_width.py`, `fig4_ladder.py`, `fig5_path.py`, `fig6_task.py`, `figS_position.py`, `figS_reachable.py`, `tables_main.py`, `supp_tables_s3_s7.py`, `supp_tables_s5.py`, `supp_tables_s8.py`, `supp_table_s7_conformal.py` (see "Regenerating tables and figures") |
| `e15_*.py`, `e15_run.sh`, `e15_run_w16.sh` | E15 position diagnostics (no training). `e15_common.py`: data and member loading, state-recording forward pass, condition helpers. `e15_position.py`: signal gain, orthogonal noise power, identities, forward response, direction-resolved response, weight statistics. `e15_flip.py`: unit signal-to-noise ratios and spike-flip rates. `e15_stationarity.py`: stationarity residuals of the surrogate-gradient lemma. `e15_summary.py`: family means and checks against the earlier release. `e15_weights.py`: weight table only. `e15_width_summary.py`: summary over the E16-E18 members. `e15_run.sh`: driver for one platform |
| `e16_surrogates.py` | peak-normalized surrogate families (triangular, sigmoid, ATan, escape rate) on a common half-width-at-half-maximum (HWHM) axis, and the parametrization of the earlier release as a control |
| `e16_models.py` | `SNNX` (one surrogate per layer, training-time membrane noise, escape-noise forward, detach-reset) and `MLPAct` (tanh, clipped ReLU, binary straight-through activation); `load_member(dir)` rebuilds any member, new or earlier |
| `e16_tasks.py`, `e16_train.py`, `e16_launch.sh`, `e16_scale.sh` | priority-ordered task queue (JSON lines); training worker; worker launcher (workers share the queue through atomic claim files); helper that raises the number of workers after other jobs on the machine have finished |
| `e16_summary.py`, `e16_progress.py`, `e16_compare.py`, `e16_tables.py`, `e16_check_detach.py` | per-member and per-configuration summary, queue progress, markdown tables, detach-reset self-check |
| `e2x_common.py`, `e2x_dump_paired.py`, `e2x_launch_dumps.sh`, `e2x_launch_w16.sh` | per-unit second moments of the frozen predictions (the input of E21-E26); paired-noise and original-condition export; launchers |
| `e21_ladder.py`, `e21_w16.py` | weight-space ladder W0-W6 and attribution of the simplex margin; the same for the width dictionaries of E16-E18, plus the noise path of those members |
| `e22_path.py` | noise path of the calibrated weights, predicted and measured critical levels, risk crossings |
| `e23_dump.py`, `e23_labelfree.py` | calibration without target labels |
| `e24_certificate.py` | split-sample gain certificate |
| `e25_budget.py` | execution budget and per-joint supports |
| `e26_select.py` | selection of the weight space from calibration data |
| `e2x_verify.py`, `e2x_summary.py`, `e2x_launch_cpu1.sh`, `e2x_launch_stage2.sh` | checks against the files of the earlier release, key numbers, launchers |
| `e27_common.py`, `e27_candidates.json` | shared helpers and the candidate predictors of E27-E29 |
| `e27_gen_ext.py`, `e27_filtered.py`, `e27_ext_eval.py`, `e27_ext_summarize.py`, `e27_summarize_all.py` | external-torque records; filtered spurious torque; evaluation on the external-torque records; collection of the key numbers of E27-E29 |
| `e28_admittance.py`, `e28_summarize.py`, `e28_followup.sh`, `e28_followup_c50.sh` | admittance control in closed loop, summary, follow-up runs |
| `e29_tracking.py`, `e29_taskweighted.py` | tracking-task supplement, task-weighted prediction and calibration |
| `theory_selfcheck.py` | numerical check of the identities, lemmas, and propositions of the supplementary theory section |
| `../prereg/hypotheses.md` | preregistration of the hypotheses H1-H12 (see below) |

## Experiments and result files

Result files are written to `results/<platform>/`, with `<platform>` being `sarcos` or `sim`. E27-E29 run on the simulated arm and write to `results/sim/` only; entries marked `sim` below also exist for the simulated arm only.

### E00-E14 (earlier release)

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
| E09 | attenuation controls: null candidate, estimator, augmented dictionaries | `analysis_aug.py`, `frontend_preds.py` | `e09_rows.csv`, `e09_summary.csv`, `e09_paired.csv`, `e09_weights.csv` |
| E10 | propagated and nominal parts of the residual | `analysis_aug.py` | `e10_members.csv`, `e10_dicts.csv` |
| E11 | estimator geometry and `m` slice | `frontend_preds.py`, `frontend_mslice.py` | `e11_frontend_geometry.json`, `e11_m_slice.csv` |
| E12 | forward perturbation response of every frozen member | `forward_response.py` | `forward_response.json`, `forward_response_aug.json` |
| E13 | closed loop with the attenuation candidates | `run_closedloop_aug.py`, `summarize_e13.py` | `e13_episodes_a.csv`, `e13_episodes_b.csv`, `e13_summary.csv`, `e13_paired.csv` |
| E13b | slow part of the open-loop residual with weights fitted on the full calibration set (reference only) | `analysis_lowfreq.py` | `e13b_lowfreq.csv` |
| E13c | total and slow part of the open-loop residual of every E13 feedforward model under the weights of the closed-loop run (ten trajectories per calibration group, estimator variances of the same group); Fig. 4b, c, Fig. 1e, Table S-XI, and the rank correlations use this file | `analysis_lowfreq_matched.py` | `e13c_lowfreq_matched.csv`; `e13_rank_correlations.json` from `figures/build_e13_tables.py` (also holds the full-calibration reference and per-seed values) |
| E14 | poor fit or surrogate: early checkpoints and reduced capacity | `analysis_underfit.py`, `run_train.py` | `e14_curve.csv`, `e14_rows.csv`, `e14_summary.csv`, `e14_paired.csv`, `e14_weights.csv`; `results/e14_numbers.json` |
| theory | bound constants of the finite-calibration certificate | `analysis_openloop.py` | `results/sim/theory_quantities.json` |

### E15-E29 (new)

Step 1, position: where a frozen member sits in the plane of signal gain `beta` and orthogonal noise power `||nu||^2`.

| Experiment | What it computes | Script | Result files |
|---|---|---|---|
| E15 position | per member, role, and condition: risk `R`, `beta`, `||nu||^2`, slope, angle risk, scale cost, residuals of the identities, split into propagated and nominal parts, growth exponent `kappa`, forward response `r`; direction-resolved responses (top-5 and bottom-5 principal directions); weight statistics against the seed-reproduced initialization; learned widths of B-L | `e15_position.py` (`e15_common.py`; `e15_weights.py` regenerates the weight table alone) | `e15_members.csv`, `e15_direction.csv`, `e15_weights.csv`, `e15_bl_widths.csv`, `e15_position_meta.json` |
| E15 flip | unit signal-to-noise ratios of the first layer, three predictions of the spike-flip probability, measured spike-flip rates under input noise | `e15_flip.py` | `e15_flip.csv`, `e15_flip_neurons.npz`, `e15_flip_meta.json` |
| E15 stationarity | numerical check of the stationarity condition at the stored checkpoints (weighted and unweighted correlation of the back-propagated error with the presynaptic trace) | `e15_stationarity.py` | `e15_stationarity.csv` |
| E15 summary | family means; comparison with the tables of the earlier release; flip-rate rank correlations | `e15_summary.py` | `e15_summary.json` |
| E15 width summary | one row per E16-E18 member, mean and seed standard deviation per configuration; checks of H1, H3, H4 and `kappa` of the bounded-activation MLPs | `e15_width_summary.py` | `e15_width_members.csv`, `e15_width_summary.csv`, `e15_width_hypotheses.json` |
| E15 on E16 members (interim run) | the five E15 outputs above for the `w16_*` members existing at that time (suffix `_w16`) | `e15_run_w16.sh` | `e15_{members,flip,direction,weights,stationarity}_w16.csv`, `e15_flip_neurons_w16.npz`, `e15_{summary,position_meta,flip_meta}_w16.json` |
| E16 | peak-normalized width sweep: four families (triangular, sigmoid, ATan, escape rate) x HWHM {0.0625, 0.125, 0.25, 0.5, 1, 2, 4}, seeds 0-2, both platforms (`w16_*`); detach-reset variant at HWHM {0.0625, 0.25, 1, 4, 16} and STE (`w16d_*`); ultra-wide HWHM 8-64, seed 0; control with the parametrization of the earlier release (`w16c_*`: sigmoid and ATan, seed 0). The STE reference is the earlier member `dm_ste_s<seed>` | `e16_tasks.py`, `e16_train.py`, `e16_launch.sh`, `e16_surrogates.py`, `e16_models.py`, `e16_summary.py` | members in `runs/<platform>/`; `e16_members.csv`, `e16_agg.csv` |
| E17 | layer-wise placement of a wide triangular surrogate (HWHM 1 or 4 on layer 1 only, or on layers 2-3 only; the other layers at 0.25), seeds 0-2 (`w17_*`) | `e16_tasks.py`, `e16_train.py` | members in `runs/<platform>/`; rows of `e16_members.csv`, `e16_agg.csv` |
| E18 | noise injection and bounded activations: training-time membrane noise (`w18_mem*`), input augmentation (`w18_aug*`), tanh / clipped-ReLU / binary MLP (`w18_mlp_*`), escape-noise forward (`w18_esc_*`); seeds 0-2 | `e16_tasks.py`, `e16_train.py` | members in `runs/<platform>/`; rows of `e16_members.csv`, `e16_agg.csv` |
| E19 | analysis only: best single width at every noise level on the paired-noise grid (three seeds; seed 0 with the ultra-wide members; detached variant) | `e21_w16.py --parts e22` | `e22_bestwidth_w16.csv` |

Step 2, reachable region: weight spaces, from per-unit moments (no training).

| Experiment | What it computes | Script | Result files |
|---|---|---|---|
| moments | per-unit second moments of the frozen predictions: paired-noise grid (`--mode grid`) and the conditions of the earlier release, optionally behind the label-free (`wfe`) or oracle (`wfo`) input estimator (`--mode orig`); `--tag` selects the files of the new members | `e2x_dump_paired.py`, `e2x_common.py` | `results/<platform>/{paired,orig}/moments_*.npz` (about 7 GB, not part of the release), `check_preds_*.json` |
| check | recomputed risks against the files of the earlier release | `e2x_verify.py` | `e2x_verify_orig.csv`, `e2x_verify_paired.csv` |
| E21 | weight-space ladder W0 (simplex), W1 (simplex + null), W2 / W3 (per joint), W4 (full linear stacking, ridge), W5 (input estimator), W6 (augmented members) for D-S, D-M, D-T, B-F, B-L, B-A, B-N; paired D-M - D-S differences; attribution of the simplex margin into scale and angle parts; `m` slice (`sim`) | `e21_ladder.py` | `e21_ladder.csv`, `e21_ladder_seeds.csv`, `e21_paired.csv`, `e21_attribution.csv`, `e21_units.npz`; `sim`: `e21_mslice.csv`, `e21_mslice_rows.csv`; paired-noise version (`--source paired`): `e21_ladder_pairednoise.csv`, `e21_ladder_seeds_pairednoise.csv`, `e21_paired_pairednoise.csv`, `e21_attribution_pairednoise.csv`, `e21_units_pairednoise.npz` |
| E21 on E16-E18 members | the ladder for the width dictionaries D-W (four widths and STE per family, plain and detached) and for the single knob members | `e21_w16.py --parts e21` | `e21_ladder_w16.csv`, `e21_ladder_w16_seeds.csv`, `e21_paired_w16.csv`, `e21_attribution_w16.csv`, `e21_ladder_e18.csv` |
| E22 | noise path of the calibrated weights on the paired-noise grid; interpolation of the residual matrix; predicted and measured critical levels; growth exponent `kappa`; risk crossings; weight table `w*(rho)`; sensitivity run with a cubic family (`--family lqc`, suffix `_lqc`) | `e22_path.py` | `e22_path.csv`, `e22_C.npz`, `e22_fit.json`, `e22_critical.csv`, `e22_kappa.csv`, `e22_crossings.csv`, `e22_wtable.csv`, `e22_predicted_risk.csv`, `e22_pairwise_roots.csv`; `e22_critical_lqc.csv`, `e22_predicted_risk_lqc.csv`, `e22_pairwise_roots_lqc.csv` |
| E22 on E16 members | the path for the width dictionaries, risk of every member along `rho`, crossings | `e21_w16.py --parts e22` | `e22_path_w16.csv`, `e22_crossings_w16.csv`, `e22_bestwidth_w16.csv` |
| E23 | calibration without target labels: noise variance from unlabeled calibration inputs, from the true noise, from a static-robot recording, and from the weight table; misspecified noise (Student t, AR(1)) | `e23_dump.py`, `e23_labelfree.py` | `e23_labelfree.csv`, `e23_rows.csv`, `e23/e23_variances.json` (the moments in `e23/*.npz` are not part of the release) |
| E24 | split-sample gain certificate (t and bootstrap lower bounds; certification rate, coverage, smallest `m` with a rate of at least 0.9) | `e24_certificate.py` | `e24_certificate.csv`, `e24_min_m.csv`, `e24_rows.csv` |
| E25 | execution budget (at most `k` executed members plus a free null, against W-Q(k)), retention, per-joint supports | `e25_budget.py` | `e25_budget.csv`, `e25_budget_rows.csv`, `e25_retention.csv`, `e25_perjoint.csv` |
| E26 | selection of the weight space by cross-fitting on the calibration data; control with the split-conformal interval length | `e26_select.py`, `e26_conformal.py` | `e26_rows.csv`, `e26_select.csv`; `e26_conformal.csv`, `e26_conformal_check.csv`, `e26_conformal_summary.csv` |
| E21-E26 summary | key numbers of E21-E26 | `e2x_summary.py` | `e2x_summary.json` |

Step 3, needed: the task layer, on the simulated arm. The candidate predictors are listed in `e27_candidates.json`.

| Experiment | What it computes | Script | Result files |
|---|---|---|---|
| E27 filtered | spurious interaction torque of every candidate under N, O1-M, and P1 after causal low-pass filters; crossing cut-off frequencies against D-S; slow / white split of the residual; check against E13c | `e27_filtered.py` (`e27_common.py`) | `e27_filtered.csv`, `e27_filtered_fine.csv`, `e27_crossing.csv`, `e27_slowwhite.csv`, `e27_e13c_check.json` |
| E27 external torque | 40 test trajectories with external joint torques (steps and ramps), evaluated under N and O1-M: estimation error, ROC / AUC, event detection with a fixed and a calibrated threshold; acceleration from the record (setting `a`) or from the differenced noisy velocity (setting `b`) | `e27_gen_ext.py`, `e27_ext_eval.py`, `e27_ext_summarize.py` | `data_sim/ext_test.npz` (written by `e27_gen_ext.py`), `e27_ext_episodes.csv`, `e27_ext_summary.csv`, `e27_ext_roc.csv`, `e27_ext_meta_a.json`, `e27_ext_meta_b.json`, `e27_ext_meta_c.json`, `e27_ext_test_info.json` |
| E28 | joint-space admittance control in closed loop with the frozen predictor as interaction-torque estimator: reference drift without a human, transparency and intention following with a human; main setting `c = 20`, true-state feedback diagnostic, damping `c = 50`; payload P1 with and without recalibration | `e28_admittance.py`, `e28_followup.sh`, `e28_followup_c50.sh`, `e28_summarize.py` | `e28_episodes.csv`, `e28_summary.csv`, `e28_paired.csv`, `e28_params_c20.json`, `e28_params_c20_truefb.json`, `e28_params_c50.json`; per-seed parts in `e28_parts/` |
| E29 | tracking supplement on the E13 setting: reference-trajectory feedforward, task-weighted calibration, reproduction of the E13 episodes; task-weighted prediction of the tracking error from the slow and white parts of the residual | `e29_tracking.py`, `e29_taskweighted.py` | `e29_tracking.csv`, `e29_tracking_summary.csv`, `e29_tracking_paired.csv`, `e29_taskweighted.csv`, `e29_taskweighted_meta.json` |
| E27-E29 summary | key numbers of E27-E29 and the checks of H11 and H12 | `e27_summarize_all.py` | `e27_29_summary.json` |
| theory | numerical check of the identities, lemmas, and propositions of the supplementary theory section | `theory_selfcheck.py` | printed output (no result file) |

`results/results_manifest.csv` maps every number quoted in the text of the paper to its source: one row per number, with the section and paragraph, the quantity, the value, platform and condition, the source file, the column or key, the selector (role, condition, group, averaging), a note, and a `verified` column (`PASS` or `FAIL` from the check that recomputes the value from the source file).
Per-member configurations (`config.json`) hold the mean spike counts per sample and layer, the epochs, the wall-clock time, and the development error.

## Running order

All commands run from `code/`. The `.sh` launchers change to their own directory and take the interpreter from `$PY` (default `python3`); the results of the earlier release are read from `../results/<platform>/`. The earlier release (E00-E14, with its frozen members in `runs25/` and `runs25/<platform>/frozen.json`) is a prerequisite.

1. **Moments of the frozen predictions** (paired-noise grid and original conditions; the members of the earlier release):
   `bash e2x_launch_dumps.sh > ../logs/e2x_dumps.log 2>&1`.
2. **E15 on the earlier members**, per platform:
   `bash e15_run.sh sim ../runs25/sim` and `bash e15_run.sh sarcos ../runs25/sarcos` (position, flip, stationarity), then `python e15_summary.py --platform sim` and `python e15_summary.py --platform sarcos`. The Python scripts take `--runs`, `--members` (glob), and `--suffix`.
3. **E21-E26**:
   `bash e2x_launch_cpu1.sh > ../logs/e2x_cpu1.log 2>&1` (E21 SARCOS, check, E24-E26 on both platforms) and `python e21_ladder.py --platform sim --res ../results/sim --mslice` (E21 on the simulated arm; no launcher calls it); then `bash e2x_launch_stage2.sh` (waits for the two logs; E23 dumps, E22, E21 on the paired noise, checks, E23, E24 again); finally `python e2x_summary.py --res_root ../results`.
4. **E16-E18 training**:
   `python e16_tasks.py --out ../logs/e16_tasks.jsonl` writes the queue; `bash e16_launch.sh 5` starts five workers, which exit when the queue is empty (`python e16_progress.py` shows the progress; `bash e16_scale.sh 7` raises the number of workers once the machine is free); `python e16_summary.py` writes `e16_members.csv` and `e16_agg.csv`; `python e16_tables.py` and `python e16_compare.py detach` or `wide` print markdown tables; `python e16_check_detach.py` checks the detach variant.
5. **E15 and E21 / E22 on the new members**, per platform (`<p>` is `sim` or `sarcos`):
   `bash e15_run.sh <p> ../runs/<p> 'w1*' _new`, then `python e15_summary.py --platform <p> --suffix _new`, and once `python e15_width_summary.py --results ../results --suffix _new`;
   `bash e2x_launch_w16.sh` (moments of the new members; pass `--overwrite` as the first argument to replace existing files after more seeds have been trained), then `python e21_w16.py --platform <p> --res ../results/<p>` (`--parts e21` or `--parts e22` for one part).
6. **E27-E29** (simulated arm; the candidates and their weights follow E13 and read `results/sim/e21_ladder_seeds.csv` and the E13 result files of the earlier release):
   `python e27_gen_ext.py --xml_dir ../mujoco_menagerie/franka_emika_panda --data_dir ../data_sim --out ../data_sim/ext_test.npz` (external-torque records);
   `python e27_filtered.py --data_dir ../data_sim --runs ../runs25/sim --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json --out_dir ../results/sim`;
   `python e27_ext_eval.py ...` (arguments in the module docstring), run in parts with `--tag _a`, `_b`, `_c`, then `python e27_ext_summarize.py ../results/sim`;
   `python e28_admittance.py ...` for the main setting (`--c_bd 20`, one output file per seed in `results/sim/e28_parts/`), then `bash e28_followup.sh` and `bash e28_followup_c50.sh`, then `python e28_summarize.py ../results/sim`;
   `python e29_tracking.py ...` in parts that write `e29_tracking_a.csv`, `e29_tracking_b.csv`, `e29_tracking_c.csv`, then `python e29_taskweighted.py`;
   `python e27_summarize_all.py ../results/sim`.
7. **Tables and figures**: `bash figures/make_all_revised.sh` (see below).

## Regenerating tables and figures

Earlier release: from the repository root, `bash code/figures/make_all.sh` rebuilds Tables II and III, the tables of S1, S2, S3, S5, S8, and S9, Figs. 1-4, and the supplementary figures (`fig5.pdf`-`fig7.pdf`) from the frozen results; the output goes to `build/locked/`.
`bash code/figures/make_all.sh figs` rebuilds the figures only. The interpreter is `$PY` (default `python3`).

E15-E29: `bash code/figures/make_all_revised.sh` (from any directory) runs the scripts below one after the other. They read `results/` and write vector PDFs to `build/fig/` and LaTeX tables to `build/tables/` at the repository root; the root can be changed with the environment variable `SNN29_ROOT`, and `SNN29_PNG=<dir>` also writes 300-dpi PNG previews. The script first unpacks the result files stored as `.csv.gz` (`gunzip -k`). Every script imports `common_style.py` (colours, paths, helpers) and can also be run on its own (`python code/figures/<script>.py`); `figS_position.py` also imports `supp_tables_s5.py`. The tables of the locked protocol (E00-E14) are written by `make_all.sh` to `build/locked/`, so the two builds do not overwrite each other:

| Script | Output | Reads |
|---|---|---|
| `fig1_overview.py` | Fig. 1 (schematic: question, plane, three steps) | no result file (illustrative values) |
| `fig2_plane.py` | Fig. 2 (signal-gain / orthogonal-noise plane) | `e15_members.csv`, `e15_members_w16.csv`, `e21_ladder.csv`, `e21_ladder_e18.csv` |
| `fig3_width.py` | Fig. 3 (surrogate width and the straight-through estimator: response, gain, and error against HWHM; spike-flip prediction against measurement) | `e15_members.csv`, `e15_members_new.csv`, `e15_flip.csv`, `e15_flip_new.csv`, `e15_flip_neurons.npz`, `e15_width_summary.csv`, `e16_members.csv` (and the `_w16` files) |
| `fig4_ladder.py` | Fig. 4 (weight-space ladder, attribution, calibration budget) | `e21_ladder.csv`, `e21_attribution.csv`, `e21_mslice.csv` |
| `fig5_path.py` | Fig. 5 (noise path, label-free calibration, certificate) | `e22_path.csv`, `e22_critical.csv`, `e22_crossings.csv`, `e23_labelfree.csv`, `e24_certificate.csv` |
| `fig6_task.py` | Fig. 6 (task layer, simulated arm: filtered spurious torque, external-torque estimation and admittance drift, task-weighted prediction of the tracking error) | `e27_filtered_fine.csv`, `e27_crossing.csv`, `e27_ext_summary.csv`, `e28_paired.csv`, `e29_taskweighted.csv`, `e29_taskweighted_meta.json` |
| `figS_position.py` | `figS4.pdf` (E16 width sweep), `figS5.pdf` (E15 flips and direction-resolved response) | E15 tables through `supp_tables_s5.load` (`e15_*.csv`, `e15_*_new.csv`), `e16_members.csv` |
| `figS_reachable.py` | `figS1.pdf` (noise path), `figS2.pdf` (certificate) | `e22_path.csv`, `e24_certificate.csv` |
| `tables_main.py` | `table2.tex`, `table3.tex`, `table4.tex` | `e21_ladder.csv`, `e21_ladder_w16.csv`, `e21_attribution.csv`, `e24_certificate.csv`, `e25_*.csv`; the latency rows come from `results/runtime/runtime.json` |
| `supp_tables_s3_s7.py` | `s3_*.tex`, `s7_*.tex` | `e21_*.csv`, `e22_*.csv`, `e23_labelfree.csv`, `e24_*.csv`, `e25_*.csv`, `e26_select.csv` |
| `supp_table_s7_conformal.py` | `s7_conformal.tex` (supplement S7: conformal selection control) | `e26_conformal.csv`, `e26_conformal_summary.csv` |
| `supp_tables_s5.py` | `s5n_*.tex` (supplement S5: member diagnostics, width, reset factor, geometry and flips, stationarity, layer placement, noise injection) | `e15_*.csv`, `e15_*_new.csv`, `e16_members.csv`, `e21_ladder{,_w16,_e18}.csv`, `e22_bestwidth_w16.csv` |
| `supp_tables_s8.py` | `s8n_*.tex` (supplement S8: task filters, interaction torque, admittance, tracking) | `e27_*.csv`, `e27_29_summary.json`, `e28_*.csv` / `.json`, `e29_*.csv` / `.json` |

## Preregistration

`prereg/hypotheses.md` (a Chinese and an English version in one file) states the twelve hypotheses H1-H12, the experiments that test them, the decision criteria, and whether each endpoint is primary or secondary. It is dated 2026-11-11. Its first declaration is that the criteria were fixed before the results of E15-E30 were seen: the criteria come from the research plan (whose SHA-256 the file records), and the file may not be changed after the runs; a change can only be appended as a dated amendment, with the original criterion and the reason reported together.
Six primary endpoints (the rank correlation between width and forward response in H1, the paired risk differences in H2, the scale share in H6, the risk difference between label-free and labelled calibration in H9, the certification rate in H10, and the drift difference in H11) are tested with a Holm correction; all other endpoints are secondary and labelled as such.
The file also lists which experiments are preregistered (E15, E16, E17, E18, E19, E21, E22, E23, E24, E27, E28), which are exploratory or descriptive (E20, E25, E26, E29, E30, and E09-E14 of the earlier release), and which results keep the status of confirmatory results under the locked protocol of the earlier release (P1-P4). Operationalizations that the research plan did not specify are marked as supplements in the file. A table in the file fixes how each pattern of outcomes is to be written up.
The file records the SHA-256 of `code/e27_candidates.json` at registration time; after registration the candidate list was extended with the E21 optima (D-S and D-M under W1 and W3); the shipped `e27_candidates.json` is the extended list, and the amendment, with the new SHA-256, is recorded in `prereg/hypotheses.md`. The optional experiments E20 and E30 are not part of this release.

## Notes

- The launch and post-processing scripts of the earlier release run from `code/` and expect `runs/`, `logs/`, `results/`, `data_sim/`, and `mujoco_menagerie/` next to it, with the SARCOS files in the parent directory; the interpreter is `$PY` (default `python3`). `throttle.sh` reads its concurrency limit from `logs/throttle_n`. In this release `runs25/` holds the frozen members of the earlier release and `runs/` the new members (the environment variable `E16_RUNS` overrides the directory used by `e16_train.py`).
- **Loading E16-E18 members.** Use `e16_models.load_member(member_dir)`; it rebuilds any member, new or earlier, from its `config.json`. Rebuilding a new member with `common.models.SNN` gives a wrong backward pass (peak-normalized surrogates `sigmoid_pn` / `atan_pn`, layer-wise widths `triangular_layerwise`, `detach_reset`) and, for `model_class = MLPAct`, a wrong forward pass. The E15 and E21-E26 scripts build new members through `e16_models.build_from_config`.
- **Suffixes of result files.**
  `_pairednoise`: E21 computed on the paired-noise moments (`--source paired`) instead of the noise regenerated as in the earlier release. The main E21 results use the regenerated noise, so they can be compared number by number with the earlier files; the paired-noise version is a robustness check.
  `_w16`: E21 and E22 for the E16 members (`w16_*`; D-M and D-S of the same seed as references), with the width dictionaries D-W built from seeds 0-2 (`seeds` column of `e21_ladder_w16.csv`). The `e15_*_w16` files come from an earlier interim run on the 65 `w16_*` members per platform that existed at that time; they are superseded by the `_new` files.
  `_new`: E15 rerun on all E16-E18 members (`w16_*`, `w16d_*`, `w16c_*`, `w17_*`, `w18_*`) after training finished (step 5 of the running order); it is the default suffix of `e15_width_summary.py`.
  `_lqc`: E22 sensitivity run with a cubic polynomial family for the interpolation of the residual matrix.
- Result files larger than about 100 KB are stored as `<name>.csv.gz` (decompress with `gunzip -k`; `make_all_revised.sh` does it for the figure and table scripts, and the analysis scripts that read such a file, for example `e27_ext_summarize.py` and `e28_summarize.py`, need the unpacked file). The per-neuron arrays `e15_flip_neurons_new.npz` and `e15_flip_neurons_w16.npz` of E15 are not part of the release; `e15_flip.py` writes them. Some `source_file` entries of `results/results_manifest.csv` name the `.csv` file of a stored `.csv.gz` file.
- The moment files are about 7 GB and are not part of the release. `bash e2x_launch_dumps.sh` and `bash e2x_launch_w16.sh` regenerate them; with `--check` every export compares the recomputed predictions with the stored `pred_*.npy` (`check_preds_*.json`). `e2x_common.MomFile` reads them.
- E09-E12, E13, and E14 are post-hoc analyses on the frozen members, calibration sets, and test sets. They explain the results of P1-P4 and do not test them. Which of E15-E29 are preregistered and which are exploratory is stated in `prereg/hypotheses.md`.
- E15 and E21-E29 train nothing; only E16-E18 train new members. E27-E29 and `e27_candidates.json` refer to the members of the earlier release by name (`dm_*`, `ds_*`, `bn_*`, `ba_*`) and read the E13 result files of the earlier release (`e13_episodes_*.csv`, `e13c_lowfreq_matched.csv`).
- `e28_followup.sh` also contains a `B_d = 50` call with the human stiffness scaled with `B_d` (`K_h = 5 B_d`). That setting was unstable in every intention episode and was discarded; `e28_followup_c50.sh` writes the same output files with the human model of the main runs (`--kh_ratio 2 --bh_ratio 0.13333333`) and has to be run after `e28_followup.sh`.
- `e28_summarize.py` merges the files `results/sim/e28_parts/e28_episodes*_s*.csv` with their `*_params.json` files. The example output path in the docstring of `e28_admittance.py` (`../results/sim/e28_episodes_a.csv`) does not match this pattern; write the per-seed outputs to `e28_parts/`.
- `theory_selfcheck.py` imports `common.models` and `common.surrogates` from the directory it is in (`code/`); run `python theory_selfcheck.py` from there. The check of Lemma 1 is skipped when PyTorch is missing. It runs in about one to three minutes on a laptop CPU.
- `e27_29_summary.json` contains `NaN` written in the default style of the Python `json` module (not strict JSON).
- Table II costs (`figures/tables.py`) are measured on the simulated-arm runs. The training column is the wall-clock time recorded in `config.json`, averaged over the five training seeds, of the deployed model or dictionary: for B-F the one `dm_sigmoid_*` run per seed (9.4 min), for D-S and D-M the five runs of the dictionary, for D-T the one run that yields the snapshots. Selecting the surrogate on the development set trained the five D-M members once (36.1 min); this cost is shared by B-F, B-A, D-S, and D-T and is stated in the table caption, not in the column. Execution is the measured batch-1 latency of the executed members on the RTX 3090 and on one CPU thread (`results/runtime/runtime.json`).
- Training times of the members added in E09-E14 and of the E16-E18 members depend on concurrency and throttling and should not be used for cost comparisons.
