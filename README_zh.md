[English](README.md) | 中文

# Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics

论文 *Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics* 的代码、冻结的运行配置和结果文件。论文及其补充材料中的每张表、每幅图都可以由本仓库的文件重新生成，每个数值都能追溯到产生它的运行。

## 内容

| 路径 | 内容 |
|---|---|
| `code/` | 训练、分析、仿真和作图代码。`code/README_zh.md` 列出每个实验（E00–E14）对应的脚本与结果文件。 |
| `results/` | 每张表、每幅图背后的冻结结果文件：`results/sarcos/` 与 `results/sim/` 对应两个平台，`results/physical/` 对应补充材料中的二自由度记录，`results/runtime/` 对应时延测量。`results/results_manifest.csv` 把论文中的每个数值对应到其来源文件、方法、条件和取值；`results/summary_all_results.txt` 是全部结果文件的纯文本汇总。 |
| `runs_frozen/` | 每个训练成员的 `config.json` 与 `history.json`（超参数、所选宽度、轮数、墙钟时间、开发集误差、各层脉冲数、训练曲线；E14 的回放运行只保留 history），冻结的开发阶段决策 `frozen_sarcos.json` 与 `frozen_sim.json`，仿真检查 `e00_checks.json`，以及 SARCOS 的划分清单。 |

有三个文件超出 git 树的体积限制，作为本仓库的 Release 附件提供（https://github.com/Shengda21/SNN1/releases）：

| Release 附件 | 大小 | 内容 | 下载后放置位置 |
|---|---|---|---|
| `members_sarcos.tar.gz` | 271 MB | SARCOS 每个成员的训练权重（`model.pt`）、配置和历史，以及该平台的冻结决策 | 解压到 `runs/`（得到 `runs/sarcos/<member>/`） |
| `members_sim.tar.gz` | 183 MB | 仿真机械臂的同类文件 | 解压到 `runs/`（得到 `runs/sim/<member>/`） |
| `unit_nmse.npz` | 139 MB | SARCOS 每个方法、随机种子和条件的逐测试行 nMSE，供 `code/summarize_results.py` 的聚类自助法读取；机械臂的对应文件（按测试轨迹）已在树中 | `results/sarcos/unit_nmse.npz` |

有了训练好的成员，每个分析脚本都可以直接运行而无需重新训练；它们生成的逐成员预测（`code/dump_preds.py`）因体积过大未存入仓库。

## 重新生成表与图

需要 Python 3 及 `numpy`、`pandas`、`matplotlib`。在仓库根目录执行：

```bash
bash code/figures/make_all.sh          # 表 II、III、S1–S3、S5、S8、S9 与图 1–7 -> build/
bash code/figures/make_all.sh figs     # 只生成图
```

解释器由环境变量 `$PY` 指定（默认 `python3`）。

## 复现实验

论文所报告运行的环境：Ubuntu 24.04.4、Python 3.12.3、PyTorch 2.13.0（CUDA 13.0）、MuJoCo 3.15.0、一块 NVIDIA RTX 3090；表 II 的 CPU 时延在 Intel Xeon W-2245 的单线程上测得。Python 包：`torch`、`numpy`、`pandas`、`scipy`、`matplotlib`、`mujoco`。

### 数据

- **SARCOS。** 从 GPML 数据页（http://gaussianprocess.org/gpml/data/）下载 `sarcos_inv.mat` 与 `sarcos_inv_test.mat`，放在仓库根目录。脚本从 `code/` 以 `--data_dir ..` 引用该目录。划分记录在 `runs_frozen/sarcos_split_manifest.json`。
- **仿真机械臂。** 把 MuJoCo Menagerie（https://github.com/google-deepmind/mujoco_menagerie）克隆到仓库根目录；被控对象为 `franka_emika_panda/panda_nohand.xml`，其执行器替换为直接力矩输入。用下面的命令生成记录：

  ```bash
  python code/sim/gen_data.py --xml_dir mujoco_menagerie/franka_emika_panda --out data_sim
  ```

  它把训练、开发、校准、测试、负载和闭环参考记录以及一致性检查 `e00_checks.json` 写入 `data_sim/`。
- **物理记录。** 补充材料中 151 个样本的二自由度记录由 `code/physical/e08.py` 分析，其结果文件在 `results/physical/`。

### 流水线

脚本在 `code/` 中运行，并要求 `runs/`、`logs/`、`results/`、`data_sim/` 和 `mujoco_menagerie/` 位于其旁边。`<p>` 为 `sarcos` 或 `sim`，`<d>` 为 `..` 或 `../data_sim`。每个脚本的文件头说明其参数；`code/README_zh.md` 给出每个实验的脚本与结果文件。

1. 开发集上的宽度网格：`python run_train.py --platform <p> --data_dir <d> --block width --seed 0 --out ../runs/<p>`。
2. 开发决策与正式训练：`bash launch_formal.sh <p> <d>` 冻结宽度、强替代函数和噪声水平（`dev_decisions.py` -> `runs/<p>/frozen.json`），并训练随机种子 0–4 的 D-M、D-S、D-T、B-L、B-A、B-N 各块。`throttle.sh` 在共享机器上限制并发训练进程数。
3. 开环分析：`bash post_train.sh` 等待两个平台完成，导出冻结预测，并运行 `analysis_openloop.py`（E01–E05、E07）与 `forward_response.py`（E12）。`python profile_runtime.py --out ../results/runtime/runtime.json` 测量执行成本。
4. 闭环 E06：`python run_closedloop.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim`。
5. 衰减分析 E09–E12：`bash launch_aug.sh <p> <d> all`（机械臂上另加 `qqd`）、`bash launch_aug2.sh <p> <d> all`，然后 `bash post_aug.sh` 与 `bash rerun_aug.sh`；`frontend_mslice.py` 计算 E11 的校准预算切片。
6. 闭环 E13：以第 4 步的参数运行 `python run_closedloop_aug.py`，然后 `python summarize_e13.py ../results/sim`、`python analysis_lowfreq_matched.py --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim/e13c_lowfreq_matched.csv`（E13c），以及以相同参数运行 `analysis_lowfreq.py` 得到完整校准集权重下的参考值（E13b）。
7. 欠拟合对照 E14：`bash launch_e14.sh`、`bash launch_e14_cap.sh`、`bash post_e14.sh`。
8. 表与图：在仓库根目录执行 `bash code/figures/make_all.sh`。

随机种子：数据划分 2025；训练种子 0–4；随机种子字典 `s` 的成员 `i` 使用 `1000 + 10*s + i`；快照运行使用 `2000 + s`；条件种子见 `code/sarcos/data.py` 与 `code/sim/data.py`；E13 的噪声种子固定在 `code/run_closedloop_aug.py` 中。

## 引用

```bibtex
@article{attenuation_surrogate_diversity_2026,
  title  = {Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics},
  author = {<authors>},
  year   = {2026},
  note   = {<journal or preprint details>}
}
```

## 许可

MIT 许可证，见 `LICENSE`。
