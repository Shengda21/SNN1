English | [中文](README_zh.md)

# Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter

Code, run configurations, result files, and preregistration of the paper *Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter*. Every table and figure of the paper and of its supplementary material can be regenerated from the files in this repository, and every number can be traced to the file that holds it.

## Contents

| Path | Content |
|---|---|
| `code/` | training, analysis, simulation, and figure code. `code/README.md` lists every experiment (E00-E14 of the locked protocol and E15-E29 of the new analyses) with its script and its result files, the seeds, and the running order. |
| `code/figures/` | table and figure builders in two groups: `make_all.sh` for the locked protocol (E00-E14) and `make_all_revised.sh` for the new analyses (Figs. 1-6, Tables II-IV, and the supplementary figures and tables of E15-E29). |
| `results/` | the result files behind every table and figure: `results/sarcos/` and `results/sim/` for the two platforms, `results/runtime/` for the latency measurements. `results/results_manifest.csv` maps every number quoted in the text of the paper to its source file, column, selector, and a verification flag; `results/results_manifest_locked_runs.csv` does the same for the locked-protocol runs; `results/e15_width_*` summarize the width sweep over both platforms; `results/summary_all_results.txt` is a plain-text summary of the locked-protocol result files. Large tables are stored as `.csv.gz`. |
| `runs_frozen/` | `config.json` and `history.json` of every trained member (hyperparameters, selected width, epochs, wall-clock time, development error, spike counts per layer, training curve), the frozen development decisions `frozen_sarcos.json` and `frozen_sim.json`, the simulation checks `e00_checks.json`, and the SARCOS split manifest. The members of the width, placement, and noise-injection experiments (E16-E18) are the directories `w16*`, `w17*`, and `w18*` of `runs_frozen/sarcos/` and `runs_frozen/sim/`. |
| `prereg/` | `hypotheses.md`, the preregistration of the hypotheses and their decision criteria. |

## Release assets

Three files are too large for the git tree and are attached to the release of this repository (https://github.com/Shengda21/SNN1/releases). They hold the members of the locked protocol (E00-E14), which the experiments E15-E29 take as their frozen predictors:

| Release asset | Size | Content | Location after download |
|---|---|---|---|
| `members_sarcos.tar.gz` | 271 MB | trained weights (`model.pt`), configuration, and history of every SARCOS member, with the frozen decisions of the platform | extract into `runs25/` (gives `runs25/sarcos/<member>/`) |
| `members_sim.tar.gz` | 183 MB | the same for the simulated arm | extract into `runs25/` (gives `runs25/sim/<member>/`) |
| `unit_nmse.npz` | 139 MB | per-test-row nMSE of every SARCOS method, seed, and condition, read by the cluster bootstrap of `code/summarize_results.py`; the arm file (per test trajectory) is in the tree | `results/sarcos/unit_nmse.npz` |

The scripts of E15-E29 read these members from `runs25/`. The Python scripts of E00-E14 take the directory through `--runs`; their launchers (`launch_*.sh`, `post_*.sh`) use `runs/`, where retraining writes.

The weights of the members of E16-E18 are not attached. Their configurations and training histories are in `runs_frozen/`, and they can be retrained with the same seeds (the same initialization and batch order) by `code/e16_tasks.py` and `code/e16_train.py`. From `code/`, with the release members in `runs25/` and the data prepared as described below:

```bash
python e16_tasks.py --out ../logs/e16_tasks.jsonl     # priority-ordered queue of all E16-E18 members
bash e16_launch.sh 5                                  # five workers share the queue; members are written to ../runs/<platform>/
```

`python e16_progress.py` shows the progress of the queue, and `python e16_summary.py` rebuilds `e16_members.csv` and `e16_agg.csv`. The running order in `code/README.md` (steps 4 and 5) continues with the analyses that read the new members.

## Regenerating the tables and figures

Needs Python 3 with `numpy`, `pandas`, and `matplotlib`. From the repository root:

```bash
bash code/figures/make_all.sh            # locked protocol (E00-E14): tables and figures -> build/locked/
bash code/figures/make_all.sh figs       # the same, figures only
bash code/figures/make_all_revised.sh    # new analyses (E15-E29): Figs. 1-6, tables, supplementary figures and tables -> build/fig/, build/tables/
```

`make_all.sh` rebuilds the tables of the locked protocol that the supplementary material keeps; `make_all_revised.sh` builds everything that is new (Figs. 1-6, Tables II-IV, and the supplementary figures and tables of E15-E29). The two builds write to different directories. `make_all_revised.sh` unpacks the `.csv.gz` result files first. The interpreter is taken from `$PY` (default `python3`).

## Reproducing the experiments

Environment of the reported runs: Ubuntu 24.04.4, Python 3.12.3, PyTorch 2.13.0 (CUDA 13.0), MuJoCo 3.15.0, one NVIDIA RTX 3090; the CPU latencies are measured on one thread of an Intel Xeon W-2245. Python packages: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `mujoco`.

### Data

- **SARCOS.** Download `sarcos_inv.mat` and `sarcos_inv_test.mat` from the GPML data page (http://gaussianprocess.org/gpml/data/) and place them in the repository root. The scripts refer to this directory as `--data_dir ..` from `code/`. The split is recorded in `runs_frozen/sarcos_split_manifest.json`.
- **Simulated arm.** Clone the MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie) into the repository root; the plant is `franka_emika_panda/panda_nohand.xml` with the actuators replaced by direct torque inputs. Generate the records with

  ```bash
  python code/sim/gen_data.py --xml_dir mujoco_menagerie/franka_emika_panda --out data_sim
  ```

  which writes the training, development, calibration, test, payload, and closed-loop reference records and the identity checks `e00_checks.json` into `data_sim/`.

### Pipeline

The scripts run from `code/` and expect `runs25/` (the members of the release), `runs/` (members trained here), `logs/`, `results/`, `data_sim/`, and `mujoco_menagerie/` next to it. `<p>` is `sarcos` or `sim`, and `<d>` is `..` or `../data_sim`. The header of every script states its arguments.

- **Locked protocol (E00-E14).**
  1. Width grid on the development set: `python run_train.py --platform <p> --data_dir <d> --block width --seed 0 --out ../runs/<p>`.
  2. Development decisions and formal training: `bash launch_formal.sh <p> <d>` freezes the widths, the strong surrogate, and the noise levels (`dev_decisions.py` -> `runs/<p>/frozen.json`) and trains the blocks D-M, D-S, D-T, B-L, B-A, and B-N for seeds 0-4. `throttle.sh` limits the number of concurrent training processes on a shared machine.
  3. Open-loop analysis: `bash post_train.sh` dumps the frozen predictions and runs `analysis_openloop.py` (E01-E05, E07) and `forward_response.py` (E12). `python profile_runtime.py --out ../results/runtime/runtime.json` measures the execution cost.
  4. Closed loop E06: `python run_closedloop.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim`.
  5. Attenuation analyses E09-E12: `bash launch_aug.sh <p> <d> all` (and `qqd` on the arm), `bash launch_aug2.sh <p> <d> all`, then `bash post_aug.sh` and `bash rerun_aug.sh`; `frontend_mslice.py` computes the calibration-budget slice of E11.
  6. Closed loop E13: `python run_closedloop_aug.py` with the arguments of step 4, then `python summarize_e13.py ../results/sim`, `analysis_lowfreq_matched.py` (E13c), and `analysis_lowfreq.py` (E13b).
  7. Underfitting control E14: `bash launch_e14.sh`, `bash launch_e14_cap.sh`, `bash post_e14.sh`.
- **New analyses (E15-E29).** The running order is in `code/README.md`: moments of the frozen predictions, E15 position diagnostics, E21-E26 weight-space analyses, E16-E18 training of the new members, and E27-E29 on the task layer.

Random seeds: data split 2025; training seeds 0-4; `1000 + 10*s + i` for member `i` of seed dictionary `s`; `2000 + s` for the snapshot runs; the condition seeds are in `code/sarcos/data.py` and `code/sim/data.py`; the noise seeds of the closed-loop runs are fixed in `code/run_closedloop_aug.py` and `code/e27_common.py`. The seeds of E15-E29 are listed in `code/README.md`.

## Preregistration

`prereg/hypotheses.md` states the hypotheses H1-H12, the experiments that test them, and their decision criteria, fixed before the results of the new experiments were seen.

## Citation

```bibtex
@article{frozen_spiking_inverse_dynamics_2026,
  title  = {Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter},
  author = {<authors>},
  year   = {2026},
  note   = {<journal or preprint details>}
}
```

## License

MIT License; see `LICENSE`.
