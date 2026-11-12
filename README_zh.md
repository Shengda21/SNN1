[English](README.md) | 中文

# Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter

论文 *Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter* 的代码、运行配置、结果文件和预注册。论文及其补充材料中的每张表、每幅图都可以由本仓库的文件重新生成，每个数值都能追溯到保存它的文件。

## 内容

| 路径 | 内容 |
|---|---|
| `code/` | 训练、分析、仿真和作图代码。`code/README_zh.md` 列出每个实验（锁定协议的 E00–E14 与新分析的 E15–E29）对应的脚本与结果文件、随机种子和运行顺序。 |
| `code/figures/` | 表与图的生成脚本，分两组：`make_all.sh` 对应锁定协议（E00–E14），`make_all_revised.sh` 对应新分析（图 1–6、表 II–IV，以及 E15–E29 的补充图表）。 |
| `results/` | 每张表、每幅图背后的结果文件：`results/sarcos/` 与 `results/sim/` 对应两个平台，`results/runtime/` 对应时延测量。`results/results_manifest.csv` 把论文正文中的每个数值对应到其来源文件、列、选取条件和校验标记；`results/results_manifest_locked_runs.csv` 对锁定协议的运行做同样的对应；`results/e15_width_*` 汇总两个平台上的宽度扫描；`results/summary_all_results.txt` 是锁定协议结果文件的纯文本汇总。较大的表以 `.csv.gz` 存放。 |
| `runs_frozen/` | 每个训练成员的 `config.json` 与 `history.json`（超参数、所选宽度、轮数、墙钟时间、开发集误差、各层脉冲数、训练曲线），冻结的开发阶段决策 `frozen_sarcos.json` 与 `frozen_sim.json`，仿真检查 `e00_checks.json`，以及 SARCOS 的划分清单。宽度、位置和噪声注入实验（E16–E18）的成员是 `runs_frozen/sarcos/` 与 `runs_frozen/sim/` 中的 `w16*`、`w17*`、`w18*` 目录。 |
| `prereg/` | `hypotheses.md`，假设及其判据的预注册。 |

## Release 附件

有三个文件超出 git 树的体积限制，作为本仓库的 Release 附件提供（https://github.com/Shengda21/SNN1/releases）。它们是锁定协议（E00–E14）的成员，也是 E15–E29 使用的冻结预测器：

| Release 附件 | 大小 | 内容 | 下载后放置位置 |
|---|---|---|---|
| `members_sarcos.tar.gz` | 271 MB | SARCOS 每个成员的训练权重（`model.pt`）、配置和历史，以及该平台的冻结决策 | 解压到 `runs25/`（得到 `runs25/sarcos/<member>/`） |
| `members_sim.tar.gz` | 183 MB | 仿真机械臂的同类文件 | 解压到 `runs25/`（得到 `runs25/sim/<member>/`） |
| `unit_nmse.npz` | 139 MB | SARCOS 每个方法、随机种子和条件的逐测试行 nMSE，供 `code/summarize_results.py` 的聚类自助法读取；机械臂的对应文件（按测试轨迹）已在树中 | `results/sarcos/unit_nmse.npz` |

E15–E29 的脚本从 `runs25/` 读取这些成员。E00–E14 的 Python 脚本通过 `--runs` 指定目录；它们的启动脚本（`launch_*.sh`、`post_*.sh`）使用 `runs/`，重新训练的结果也写入该目录。

E16–E18 新成员的权重未随附。它们的配置和训练历史在 `runs_frozen/` 中，可用相同的种子（相同的初始化和批次顺序）由 `code/e16_tasks.py` 与 `code/e16_train.py` 重新训练。在 `code/` 下执行，要求发布的成员已放在 `runs25/`，数据已按下文准备好：

```bash
python e16_tasks.py --out ../logs/e16_tasks.jsonl     # 生成全部 E16–E18 成员的优先级队列
bash e16_launch.sh 5                                  # 五个工作进程共享队列；成员写入 ../runs/<platform>/
```

`python e16_progress.py` 显示队列进度，`python e16_summary.py` 重建 `e16_members.csv` 与 `e16_agg.csv`。`code/README_zh.md` 的运行顺序（第 4、5 步）接着给出读取新成员的分析。

## 重新生成表与图

需要 Python 3 及 `numpy`、`pandas`、`matplotlib`。在仓库根目录执行：

```bash
bash code/figures/make_all.sh            # 锁定协议（E00–E14）：表和图 -> build/locked/
bash code/figures/make_all.sh figs       # 同上，只生成图
bash code/figures/make_all_revised.sh    # 新分析（E15–E29）：图 1–6、表、补充图表 -> build/fig/、build/tables/
```

`make_all.sh` 重新生成补充材料中保留的锁定协议的表；`make_all_revised.sh` 生成全部新内容（图 1–6、表 II–IV，以及 E15–E29 的补充图表）。两套构建写入不同的目录。`make_all_revised.sh` 会先解压 `.csv.gz` 结果文件。解释器由环境变量 `$PY` 指定（默认 `python3`）。

## 复现实验

所报告运行的环境：Ubuntu 24.04.4、Python 3.12.3、PyTorch 2.13.0（CUDA 13.0）、MuJoCo 3.15.0、一块 NVIDIA RTX 3090；CPU 时延在 Intel Xeon W-2245 的单线程上测得。Python 包：`torch`、`numpy`、`pandas`、`scipy`、`matplotlib`、`mujoco`。

### 数据

- **SARCOS。** 从 GPML 数据页（http://gaussianprocess.org/gpml/data/）下载 `sarcos_inv.mat` 与 `sarcos_inv_test.mat`，放在仓库根目录。脚本从 `code/` 以 `--data_dir ..` 引用该目录。划分记录在 `runs_frozen/sarcos_split_manifest.json`。
- **仿真机械臂。** 把 MuJoCo Menagerie（https://github.com/google-deepmind/mujoco_menagerie）克隆到仓库根目录；被控对象为 `franka_emika_panda/panda_nohand.xml`，其执行器替换为直接力矩输入。用下面的命令生成记录：

  ```bash
  python code/sim/gen_data.py --xml_dir mujoco_menagerie/franka_emika_panda --out data_sim
  ```

  它把训练、开发、校准、测试、负载和闭环参考记录以及一致性检查 `e00_checks.json` 写入 `data_sim/`。

### 流水线

脚本在 `code/` 中运行，并要求 `runs25/`（发布的成员）、`runs/`（本地训练的成员）、`logs/`、`results/`、`data_sim/` 和 `mujoco_menagerie/` 位于其旁边。`<p>` 为 `sarcos` 或 `sim`，`<d>` 为 `..` 或 `../data_sim`。每个脚本的文件头说明其参数。

- **锁定协议（E00–E14）。**
  1. 开发集上的宽度网格：`python run_train.py --platform <p> --data_dir <d> --block width --seed 0 --out ../runs/<p>`。
  2. 开发决策与正式训练：`bash launch_formal.sh <p> <d>` 冻结宽度、强替代函数和噪声水平（`dev_decisions.py` -> `runs/<p>/frozen.json`），并训练随机种子 0–4 的 D-M、D-S、D-T、B-L、B-A、B-N 各块。`throttle.sh` 在共享机器上限制并发训练进程数。
  3. 开环分析：`bash post_train.sh` 导出冻结预测，并运行 `analysis_openloop.py`（E01–E05、E07）与 `forward_response.py`（E12）。`python profile_runtime.py --out ../results/runtime/runtime.json` 测量执行成本。
  4. 闭环 E06：`python run_closedloop.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim`。
  5. 衰减分析 E09–E12：`bash launch_aug.sh <p> <d> all`（机械臂上另加 `qqd`）、`bash launch_aug2.sh <p> <d> all`，然后 `bash post_aug.sh` 与 `bash rerun_aug.sh`；`frontend_mslice.py` 计算 E11 的校准预算切片。
  6. 闭环 E13：以第 4 步的参数运行 `python run_closedloop_aug.py`，然后 `python summarize_e13.py ../results/sim`、`analysis_lowfreq_matched.py`（E13c）和 `analysis_lowfreq.py`（E13b）。
  7. 欠拟合对照 E14：`bash launch_e14.sh`、`bash launch_e14_cap.sh`、`bash post_e14.sh`。
- **新分析（E15–E29）。** 运行顺序见 `code/README_zh.md`：冻结预测的矩、E15 位置诊断、E21–E26 权重空间分析、E16–E18 新成员的训练，以及任务层上的 E27–E29。

随机种子：数据划分 2025；训练种子 0–4；随机种子字典 `s` 的成员 `i` 使用 `1000 + 10*s + i`；快照运行使用 `2000 + s`；条件种子见 `code/sarcos/data.py` 与 `code/sim/data.py`；闭环运行的噪声种子固定在 `code/run_closedloop_aug.py` 与 `code/e27_common.py` 中。E15–E29 的种子列在 `code/README_zh.md` 中。

## 预注册

`prereg/hypotheses.md` 给出假设 H1–H12、检验它们的实验及其判据，这些内容在看到新实验的结果之前已经确定。

## 引用

```bibtex
@article{frozen_spiking_inverse_dynamics_2026,
  title  = {Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter},
  author = {<authors>},
  year   = {2026},
  note   = {<journal or preprint details>}
}
```

## 许可

MIT 许可证，见 `LICENSE`。
