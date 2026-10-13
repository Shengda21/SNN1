# Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics

Code, frozen run configurations, and result files of the paper *Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics*. Every table and figure of the paper and of its supplementary material can be regenerated from the files in this repository, and every number can be traced to the run that produced it.

## Contents

| Path | Content |
|---|---|
| `code/` | training, analysis, simulation, and figure code. `code/README.md` lists every experiment (E00-E14) with its script and its result files. |
| `results/` | the frozen result files behind every table and figure: `results/sarcos/` and `results/sim/` for the two platforms, `results/physical/` for the 2-DOF recording of the supplement, `results/runtime/` for the latency measurements. `results/results_manifest.csv` maps every number of the paper to its source file, method, condition, and value; `results/summary_all_results.txt` is a plain-text summary of all result files. |
| `runs_frozen/` | `config.json` and `history.json` of every trained member (hyperparameters, selected width, epochs, wall-clock time, development error, spike counts per layer, training curve; the E14 replay runs keep their history only), the frozen development decisions `frozen_sarcos.json` and `frozen_sim.json`, the simulation checks `e00_checks.json`, and the SARCOS split manifest. |

Three files are too large for the git tree and are attached to the release of this repository:

| Release asset | Size | Content | Location after download |
|---|---|---|---|
| `members_sarcos.tar.gz` | 271 MB | trained weights (`model.pt`), configuration, and history of every SARCOS member, with the frozen decisions of the platform | extract into `runs/` (gives `runs/sarcos/<member>/`) |
| `members_sim.tar.gz` | 183 MB | the same for the simulated arm | extract into `runs/` (gives `runs/sim/<member>/`) |
| `unit_nmse.npz` | 139 MB | per-test-row nMSE of every SARCOS method, seed, and condition, read by the cluster bootstrap of `code/summarize_results.py`; the arm file (per test trajectory) is in the tree | `results/sarcos/unit_nmse.npz` |

The trained members make every analysis script runnable without retraining; the per-member predictions they produce (`code/dump_preds.py`) are not stored because of their size.

## Regenerating the tables and figures

Needs Python 3 with `numpy`, `pandas`, and `matplotlib`. From the repository root:

```bash
bash code/figures/make_all.sh          # Tables II, III, S1-S3, S5, S8, S9 and Figs. 1-7 -> build/
bash code/figures/make_all.sh figs     # figures only
```

The interpreter is taken from `$PY` (default `python3`).

## Reproducing the experiments

Environment of the reported runs: Ubuntu 24.04.4, Python 3.12.3, PyTorch 2.13.0 (CUDA 13.0), MuJoCo 3.15.0, one NVIDIA RTX 3090; the CPU latencies of Table II are measured on one thread of an Intel Xeon W-2245. Python packages: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `mujoco`.

### Data

- **SARCOS.** Download `sarcos_inv.mat` and `sarcos_inv_test.mat` from the GPML data page (http://gaussianprocess.org/gpml/data/) and place them in the repository root. The scripts refer to this directory as `--data_dir ..` from `code/`. The split is recorded in `runs_frozen/sarcos_split_manifest.json`.
- **Simulated arm.** Clone the MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie) into the repository root; the plant is `franka_emika_panda/panda_nohand.xml` with the actuators replaced by direct torque inputs. Generate the records with

  ```bash
  python code/sim/gen_data.py --xml_dir mujoco_menagerie/franka_emika_panda --out data_sim
  ```

  which writes the training, development, calibration, test, payload, and closed-loop reference records and the identity checks `e00_checks.json` into `data_sim/`.
- **Physical recording.** The 151-sample 2-DOF record of the supplement is analyzed by `code/physical/e08.py`; its result files are in `results/physical/`.

### Pipeline

The scripts run from `code/` and expect `runs/`, `logs/`, `results/`, `data_sim/`, and `mujoco_menagerie/` next to it. `<p>` is `sarcos` or `sim`, and `<d>` is `..` or `../data_sim`. The header of every script states its arguments; `code/README.md` gives the script and the result files of every experiment.

1. Width grid on the development set: `python run_train.py --platform <p> --data_dir <d> --block width --seed 0 --out ../runs/<p>`.
2. Development decisions and formal training: `bash launch_formal.sh <p> <d>` freezes the widths, the strong surrogate, and the noise levels (`dev_decisions.py` -> `runs/<p>/frozen.json`) and trains the blocks D-M, D-S, D-T, B-L, B-A, and B-N for seeds 0-4. `throttle.sh` limits the number of concurrent training processes on a shared machine.
3. Open-loop analysis: `bash post_train.sh` waits for both platforms, dumps the frozen predictions, and runs `analysis_openloop.py` (E01-E05, E07) and `forward_response.py` (E12). `python profile_runtime.py --out ../results/runtime/runtime.json` measures the execution cost.
4. Closed loop E06: `python run_closedloop.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim`.
5. Attenuation analyses E09-E12: `bash launch_aug.sh <p> <d> all` (and `qqd` on the arm), `bash launch_aug2.sh <p> <d> all`, then `bash post_aug.sh` and `bash rerun_aug.sh`; `frontend_mslice.py` computes the calibration-budget slice of E11.
6. Closed loop E13: `python run_closedloop_aug.py` with the arguments of step 4, then `python summarize_e13.py ../results/sim`, `python analysis_lowfreq_matched.py --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim/e13c_lowfreq_matched.csv` (E13c), and `analysis_lowfreq.py` with the same arguments for the full-calibration reference (E13b).
7. Underfitting control E14: `bash launch_e14.sh`, `bash launch_e14_cap.sh`, `bash post_e14.sh`.
8. Tables and figures: `bash code/figures/make_all.sh` from the repository root.

Random seeds: data split 2025; training seeds 0-4; `1000 + 10*s + i` for member `i` of seed dictionary `s`; `2000 + s` for the snapshot runs; the condition seeds are in `code/sarcos/data.py` and `code/sim/data.py`; the noise seeds of E13 are fixed in `code/run_closedloop_aug.py`.

## Citation

```bibtex
@article{attenuation_surrogate_diversity_2026,
  title  = {Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics},
  author = {<authors>},
  year   = {2026},
  note   = {<journal or preprint details>}
}
```

## License

MIT License; see `LICENSE`.
