[English](README.md) | 中文

# 代码发布：Attenuation Explains the Noise Gain of Surrogate Diversity in Spiking Robot Inverse Dynamics

本目录包含论文及其补充材料的训练、分析、仿真和绘图代码。
补充材料仅通过本 README 引用文件名。

## 环境

Ubuntu 24.04.4，Python 3.12.3，PyTorch 2.13.0（CUDA 13.0），MuJoCo 3.15.0，一块 NVIDIA RTX 3090；Table II 中的 CPU 延迟在 Intel Xeon W-2245 的一个线程上测得。
随机种子：数据划分为 2025；训练随机种子为 0-4（D-M、B-L、B-A、B-N）；种子字典 `s` 的第 `i` 个成员使用 `1000 + 10*s + i`；快照运行使用 `2000 + s`。
各条件的随机种子位于 `sarcos/data.py` 和 `sim/data.py`。闭环运行 E13 的噪声随机种子固定在 `run_closedloop_aug.py` 中（N 为 900，O1-M 为 901，P1 为 902）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `common/` | 替代函数、脉冲模型、训练循环、校准规则 W-S/W-E/W-B/W-D/W-Q(k)、指标 |
| `sarcos/data.py`, `sim/data.py` | 数据集、划分、运行条件 N、O1-L/M/H、O2-B、P1 |
| `sim/gen_data.py`, `sim/panda_env.py` | MuJoCo Panda 记录生成和批量闭环环境 |
| `run_train.py` | 训练所有成员，以及增广成员、早期检查点成员和缩减容量成员 |
| `dump_preds.py` | 冻结的各成员在校准集和测试集上的预测 |
| `launch_*.sh`, `post_*.sh`, `rerun_aug.sh`, `throttle.sh` | 批量启动脚本、后处理脚本，以及用于共享机器的负载限制器 |
| `figures/` | 表格和图的生成脚本（`make_all.sh` 重新生成全部） |
| `physical/e08.py` | 2 自由度物理记录的采集示例 |

## 实验与结果文件

结果文件写入 `results/<platform>/`，其中 `<platform>` 为 `sarcos` 或 `sim`。

| 实验 | 计算内容 | 脚本 | 结果文件 |
|---|---|---|---|
| E00 | 仿真记录的恒等式检查；冻结的开发决策（宽度、强替代函数、条件） | `sim/gen_data.py`, `dev_decisions.py` | `runs_frozen/e00_checks.json`, `runs_frozen/frozen_<platform>.json` |
| E01 | 每种方法和每个条件的测试 nMSE（Tables II, S2, S3） | `analysis_openloop.py`, `summarize_results.py` | `e01_summary.csv`, `e01_test_table.csv`, `calibration_weights.json`, `unit_nmse.npz` |
| E02 | 残差几何和双成员见证 | `analysis_openloop.py` | `e02_geometry.json` |
| E03 | 校准预算 `m` 切片 | `analysis_openloop.py` | `e03_m_slice.csv` |
| E04 | 执行成员 `k` 切片和一阶最优性残差（列 `kkt`） | `analysis_openloop.py` | `e04_k_slice.csv` |
| E05 | 逐关节误差和绝对充分性 | `analysis_openloop.py` | `e05_boundary.csv` |
| E06 | 共享反馈律下的闭环跟踪 | `run_closedloop.py` | `e06_episodes.csv`, `e06_config.json`, `e06_traces.npz` |
| E07 | 增益密度 | `analysis_openloop.py` | `e07_density.csv` |
| E08 | 物理记录（补充材料 S5），参数选择划分、归一化和轮数 | `physical/e08.py` | `results/physical/e08*.json` |
| E09 | 衰减对照：空候选、估计器、增广字典 | `analysis_aug.py`, `frontend_preds.py` | `e09_rows.csv`, `e09_summary.csv`, `e09_paired.csv`, `e09_weights.csv` |
| E10 | 残差的传播部分和名义部分 | `analysis_aug.py` | `e10_members.csv`, `e10_dicts.csv` |
| E11 | 估计器几何和 `m` 切片 | `frontend_preds.py`, `frontend_mslice.py` | `e11_frontend_geometry.json`, `e11_m_slice.csv` |
| E12 | 每个冻结成员的前向扰动响应 | `forward_response.py` | `forward_response.json`, `forward_response_aug.json` |
| E13 | 使用衰减候选的闭环 | `run_closedloop_aug.py`, `summarize_e13.py` | `e13_episodes_a.csv`, `e13_episodes_b.csv`, `e13_summary.csv`, `e13_paired.csv` |
| E13b | 使用全部校准集拟合权重时开环残差的慢变部分（仅作参考） | `analysis_lowfreq.py` | `e13b_lowfreq.csv` |
| E13c | 在闭环运行所用权重下，每个 E13 前馈模型开环残差的总量和慢变部分（每个校准组十条轨迹，同一组的估计器方差）；Fig. 4b, c、Fig. 1e、Table S-XI 和秩相关均使用此文件 | `analysis_lowfreq_matched.py` | `e13c_lowfreq_matched.csv`；`e13_rank_correlations.json` 由 `figures/build_e13_tables.py` 生成（同时包含全部校准集的参考值和各随机种子的值） |
| E14 | 欠拟合还是替代函数：早期检查点和缩减容量 | `analysis_underfit.py`, `run_train.py` | `e14_curve.csv`, `e14_rows.csv`, `e14_summary.csv`, `e14_paired.csv`, `e14_weights.csv`；`results/e14_numbers.json` |
| 理论 | 有限校准证书的界常数 | `analysis_openloop.py` | `results/sim/theory_quantities.json` |

`results/results_manifest.csv` 将论文中的每个数字对应到其来源运行。
各成员的配置（`config.json`）记录每个样本每层的平均脉冲数、轮数、墙钟时间和开发集误差。

## 重新生成表格和图

在仓库根目录下，`bash code/figures/make_all.sh` 根据冻结的结果重新生成 Tables II 和 III、S1、S2、S3、S5、S8 和 S9 的表格、Figs. 1-4 以及补充图（`fig5.pdf`-`fig7.pdf`）；输出位于 `build/`。
`bash code/figures/make_all.sh figs` 只重新生成图。解释器为 `$PY`（默认 `python3`）。

## 说明

- 启动脚本和后处理脚本从 `code/` 运行，并要求 `runs/`、`logs/`、`results/`、`data_sim/` 和 `mujoco_menagerie/` 位于其旁边，SARCOS 文件位于上一级目录；解释器为 `$PY`（默认 `python3`）。`throttle.sh` 从 `logs/throttle_n` 读取并发上限。

- E09-E12、E13 和 E14 是对冻结成员、校准集和测试集的事后分析。它们解释 P1-P4 的结果，而不检验这些结果。
- E06 的噪声随机种子由字符串哈希导出，在不同进程之间不可复现；E13 使用固定随机种子并重新运行基线。
- Table II 的成本（`figures/tables.py`）在仿真机械臂的运行上测得。训练列是 `config.json` 中记录的墙钟时间，对五个训练随机种子取平均，对象为所部署的模型或字典：B-F 取每个随机种子的一次 `dm_sigmoid_*` 运行（9.4 min），D-S 和 D-M 取该字典的五次运行，D-T 取产生快照的那一次运行。在开发集上选择替代函数时，五个 D-M 成员训练了一次（36.1 min）；这一成本由 B-F、B-A、D-S 和 D-T 共同承担，在表的标题中说明，不计入该列。执行列是在 RTX 3090 和一个 CPU 线程上测得的、所执行成员的批大小为 1 的延迟（`results/runtime/runtime.json`）。
- E09-E14 中新增成员的训练时间取决于并发和限速，不应用于成本比较。
