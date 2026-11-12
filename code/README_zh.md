[English](README.md) | 中文

# 代码发布：Using Frozen Spiking Inverse-Dynamics Predictors Under Sensor Noise: Signal Gain, Weight Constraints, and the Task Filter

本目录包含论文及其补充材料的训练、分析、仿真和绘图代码。
它在原稿发布（实验 E00-E14：`common/`、`sarcos/`、`sim/`、`run_*.py`、`analysis_*.py` 等）的基础上新增实验 **E15-E29**。新增文件的前缀为 `e15_` 至 `e29_` 和 `e2x_`，另有 `theory_selfcheck.py`。原稿发布中的文件均未修改；新脚本导入其模块（`common/`、`sim/`、`sarcos/`）并读取其冻结成员。
补充材料仅通过本 README 引用文件名。

## 环境

Ubuntu 24.04.4，Python 3.12.3，PyTorch 2.13.0（CUDA 13.0），MuJoCo 3.15.0，一块 NVIDIA RTX 3090；Table II 中的 CPU 延迟在 Intel Xeon W-2245 的一个线程上测得。

原稿发布的随机种子：数据划分为 2025；训练随机种子为 0-4（D-M、B-L、B-A、B-N）；种子字典 `s` 的第 `i` 个成员使用 `1000 + 10*s + i`；快照运行使用 `2000 + s`。
各条件的随机种子位于 `sarcos/data.py` 和 `sim/data.py`。闭环运行 E13 的噪声随机种子固定在 `run_closedloop_aug.py` 中（N 为 900，O1-M 为 901，P1 为 902）；E28 和 E29 沿用这些种子（`e27_common.NOISE_SEED`）。

新实验的种子规则：

- **E16-E18 成员**（`w16_*`、`w16d_*`、`w16c_*`、`w17_*`、`w18_*`）直接以训练种子 0、1、2 作为 torch 种子：构建模型之前调用 `set_seed(seed)`，因此新成员与原稿中同一种子的成员具有相同的初始化和相同的批次顺序（例如 `w16_tri_h0.25_s0` 与 `dm_triangular_s0`）。实际运行的队列包含种子 0-2；超宽的 `w16_*` 成员（HWHM 8-64）和对照成员 `w16c_*` 只有种子 0。需要更多种子时用 `e16_tasks.py --extra "w16_tri_h1:3,4"` 追加。
- **detach 变体**（`w16d_*`，含 `w16d_ste_*`）：前向不变；减法复位写作 `mem - spk.detach() * thr`，使复位不再把时间信用乘以 `(1 - thr * g(u))`。阈值仍通过复位项得到梯度。初始化与 `common.models.SNN` 相同。
- **E15**：前向响应的扰动使用 `rng(0)` 和与 `forward_response.py` 相同的 2000 行开发集；方向分辨响应使用 `rng(15)`，所有成员共用；驻点检验使用以种子 8（`--seed`）抽取的 16,384 行训练数据。
- **E21-E26**：配对噪声网格对每个角色抽取一组标准正态数组（种子 `dev` 2901、`cal` 2902、`test` 2903、`cal_P1` 2904、`test_P1` 2905；`e2x_common.XI_SEED`），再乘以 `rho`。原稿的条件用 `apply_condition` 重新生成，与已存的预测逐元素相同。冻结成员的种子为 0-4。聚类自助法在训练种子和测试单位两级上进行，默认重采样 1000 次。
- **E27-E29**：校准使用每组 10 条轨迹的第 0-2 组（与 E13 相同）；冻结成员的种子为 0-4。E27 的外力矩记录使用测试角色的参考轨迹（种子 400）、事件种子 2701 和观测噪声种子 2702；E28 的意图轨迹使用种子 2801。

## 目录结构

除另有说明外，路径均相对于本目录。脚本要求 `runs25/`（原稿发布的冻结成员）、`runs/`（E16-E18 新成员）、`data_sim/`、`mujoco_menagerie/`、`logs/` 和 `results/` 位于 `code/` 旁边，SARCOS 文件位于上一级目录。`runs_frozen/` 存放每个成员的 `config.json` 与 `history.json`，包括 E16-E18 的新成员；这些新成员的权重不随仓库分发（见下文）。

| 路径 | 内容 |
|---|---|
| `common/` | 替代函数、脉冲模型、训练循环、校准规则 W-S/W-E/W-B/W-D/W-Q(k)、指标 |
| `sarcos/data.py`, `sim/data.py` | 数据集、划分、运行条件 N、O1-L/M/H、O2-B、P1 |
| `sim/gen_data.py`, `sim/panda_env.py` | MuJoCo Panda 记录生成和批量闭环环境 |
| `run_train.py` | 训练所有成员，以及增广成员、早期检查点成员和缩减容量成员 |
| `dump_preds.py` | 冻结的各成员在校准集和测试集上的预测 |
| `launch_*.sh`, `post_*.sh`, `rerun_aug.sh`, `throttle.sh` | 批量启动脚本、后处理脚本，以及用于共享机器的负载限制器 |
| `figures/` | 表格和图的生成脚本，分两组。锁定协议（E00-E14），由 `make_all.sh` 运行：`tables.py`、`fig1.py`-`fig6.py`、`build_*.py`。新稿（E15-E29），由 `make_all_revised.sh` 运行：`common_style.py`、`fig1_overview.py`、`fig2_plane.py`、`fig3_width.py`、`fig4_ladder.py`、`fig5_path.py`、`fig6_task.py`、`figS_position.py`、`figS_reachable.py`、`tables_main.py`、`supp_tables_s3_s7.py`、`supp_tables_s5.py`、`supp_tables_s8.py`、`supp_table_s7_conformal.py`（见“重新生成表格和图”） |
| `e15_*.py`, `e15_run.sh`, `e15_run_w16.sh` | E15 位置诊断（不训练）。`e15_common.py`：数据与成员加载、记录状态的前向、条件辅助函数。`e15_position.py`：信号增益、正交噪声功率、恒等式、前向响应、方向分辨响应、权重统计。`e15_flip.py`：单元信噪比和脉冲翻转率。`e15_stationarity.py`：替代梯度引理的驻点残差。`e15_summary.py`：族均值以及与原稿发布的核对。`e15_weights.py`：只生成权重表。`e15_width_summary.py`：对 E16-E18 成员的汇总。`e15_run.sh`：单个平台的驱动脚本 |
| `e16_surrogates.py` | 峰值归一的替代函数族（三角、sigmoid、ATan、逃逸率），统一在半高半宽（HWHM）轴上；以及作为对照的原稿参数化 |
| `e16_models.py` | `SNNX`（逐层替代函数、训练期膜噪声、逃逸噪声随机前向、detach 复位）和 `MLPAct`（tanh、截断 ReLU、二值直通激活）；`load_member(dir)` 重建任何成员，新旧皆可 |
| `e16_tasks.py`, `e16_train.py`, `e16_launch.sh`, `e16_scale.sh` | 按优先级排序的任务队列（JSON lines）；训练工作进程；工作进程启动脚本（各进程通过原子的认领文件共享队列）；机器上其他作业结束后增加工作进程数的辅助脚本 |
| `e16_summary.py`, `e16_progress.py`, `e16_compare.py`, `e16_tables.py`, `e16_check_detach.py` | 逐成员和逐配置的汇总、队列进度、markdown 表、detach 复位的自检 |
| `e2x_common.py`, `e2x_dump_paired.py`, `e2x_launch_dumps.sh`, `e2x_launch_w16.sh` | 冻结预测的逐单位二阶矩（E21-E26 的输入）；配对噪声和原稿条件的导出；启动脚本 |
| `e21_ladder.py`, `e21_w16.py` | 权重空间阶梯 W0-W6 与单纯形边际的归因；对 E16-E18 的宽度字典做同样的分析，并计算这些成员的噪声路径 |
| `e22_path.py` | 校准权重的噪声路径、预测与实测的临界水平、风险交叉 |
| `e23_dump.py`, `e23_labelfree.py` | 无目标标签的校准 |
| `e24_certificate.py` | 分样本增益证书 |
| `e25_budget.py` | 执行预算与逐关节支撑 |
| `e26_select.py` | 由校准数据选择权重空间 |
| `e2x_verify.py`, `e2x_summary.py`, `e2x_launch_cpu1.sh`, `e2x_launch_stage2.sh` | 与原稿发布结果文件的核对、关键数字汇总、启动脚本 |
| `e27_common.py`, `e27_candidates.json` | 共用辅助函数和 E27-E29 的候选预测器 |
| `e27_gen_ext.py`, `e27_filtered.py`, `e27_ext_eval.py`, `e27_ext_summarize.py`, `e27_summarize_all.py` | 外力矩记录；滤波后的虚假力矩；在外力矩记录上的评估；E27-E29 关键数字的汇总 |
| `e28_admittance.py`, `e28_summarize.py`, `e28_followup.sh`, `e28_followup_c50.sh` | 导纳控制闭环、汇总、补充运行 |
| `e29_tracking.py`, `e29_taskweighted.py` | 跟踪任务补充、任务加权预测与校准 |
| `theory_selfcheck.py` | 补充材料理论部分的恒等式、引理和命题的数值核对 |
| `../prereg/hypotheses.md` | 假设 H1-H12 的预注册（见下文） |

## 实验与结果文件

结果文件写入 `results/<platform>/`，其中 `<platform>` 为 `sarcos` 或 `sim`。E27-E29 在仿真机械臂上运行，只写入 `results/sim/`；下表中标注 `sim` 的文件同样只存在于仿真机械臂。

### E00-E14（原稿发布）

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
| E09 | 衰减对照：空候选、估计器、增广字典 | `analysis_aug.py`, `frontend_preds.py` | `e09_rows.csv`, `e09_summary.csv`, `e09_paired.csv`, `e09_weights.csv` |
| E10 | 残差的传播部分和名义部分 | `analysis_aug.py` | `e10_members.csv`, `e10_dicts.csv` |
| E11 | 估计器几何和 `m` 切片 | `frontend_preds.py`, `frontend_mslice.py` | `e11_frontend_geometry.json`, `e11_m_slice.csv` |
| E12 | 每个冻结成员的前向扰动响应 | `forward_response.py` | `forward_response.json`, `forward_response_aug.json` |
| E13 | 使用衰减候选的闭环 | `run_closedloop_aug.py`, `summarize_e13.py` | `e13_episodes_a.csv`, `e13_episodes_b.csv`, `e13_summary.csv`, `e13_paired.csv` |
| E13b | 使用全部校准集拟合权重时开环残差的慢变部分（仅作参考） | `analysis_lowfreq.py` | `e13b_lowfreq.csv` |
| E13c | 在闭环运行所用权重下，每个 E13 前馈模型开环残差的总量和慢变部分（每个校准组十条轨迹，同一组的估计器方差）；Fig. 4b, c、Fig. 1e、Table S-XI 和秩相关均使用此文件 | `analysis_lowfreq_matched.py` | `e13c_lowfreq_matched.csv`；`e13_rank_correlations.json` 由 `figures/build_e13_tables.py` 生成（同时包含全部校准集的参考值和各随机种子的值） |
| E14 | 欠拟合还是替代函数：早期检查点和缩减容量 | `analysis_underfit.py`, `run_train.py` | `e14_curve.csv`, `e14_rows.csv`, `e14_summary.csv`, `e14_paired.csv`, `e14_weights.csv`；`results/e14_numbers.json` |
| 理论 | 有限校准证书的界常数 | `analysis_openloop.py` | `results/sim/theory_quantities.json` |

### E15-E29（新增）

第一步，位置：冻结成员在信号增益 `beta` 与正交噪声功率 `||nu||^2` 所成平面上的落点。

| 实验 | 计算内容 | 脚本 | 结果文件 |
|---|---|---|---|
| E15 位置 | 逐成员、角色和条件：风险 `R`、`beta`、`||nu||^2`、斜率、角度风险、尺度代价、恒等式残差、传播部分与名义部分的拆分、增长指数 `kappa`、前向响应 `r`；方向分辨响应（前 5 和后 5 个主方向）；相对于按种子重现的初始化的权重统计；B-L 学到的宽度 | `e15_position.py`（`e15_common.py`；`e15_weights.py` 单独重新生成权重表） | `e15_members.csv`, `e15_direction.csv`, `e15_weights.csv`, `e15_bl_widths.csv`, `e15_position_meta.json` |
| E15 翻转 | 第一层的单元信噪比、脉冲翻转概率的三种预测、输入噪声下实测的脉冲翻转率 | `e15_flip.py` | `e15_flip.csv`, `e15_flip_neurons.npz`, `e15_flip_meta.json` |
| E15 驻点 | 在已存检查点上对驻点条件做数值检验（反传误差与突触前迹的加权相关和未加权相关） | `e15_stationarity.py` | `e15_stationarity.csv` |
| E15 汇总 | 族均值；与原稿发布表格的核对；翻转率的秩相关 | `e15_summary.py` | `e15_summary.json` |
| E15 宽度汇总 | 每个 E16-E18 成员一行，逐配置的均值和种子标准差；H1、H3、H4 的检验以及有界激活 MLP 的 `kappa` | `e15_width_summary.py` | `e15_width_members.csv`, `e15_width_summary.csv`, `e15_width_hypotheses.json` |
| E16 成员上的 E15（阶段性运行） | 对当时已有的 `w16_*` 成员生成上述五类 E15 输出（后缀 `_w16`） | `e15_run_w16.sh` | `e15_{members,flip,direction,weights,stationarity}_w16.csv`, `e15_flip_neurons_w16.npz`, `e15_{summary,position_meta,flip_meta}_w16.json` |
| E16 | 峰值归一的宽度扫描：四个族（三角、sigmoid、ATan、逃逸率）x HWHM {0.0625, 0.125, 0.25, 0.5, 1, 2, 4}，种子 0-2，两个平台（`w16_*`）；HWHM {0.0625, 0.25, 1, 4, 16} 和 STE 的 detach 复位变体（`w16d_*`）；超宽 HWHM 8-64，种子 0；采用原稿参数化的对照（`w16c_*`：sigmoid 与 ATan，种子 0）。STE 参照为原稿成员 `dm_ste_s<seed>` | `e16_tasks.py`, `e16_train.py`, `e16_launch.sh`, `e16_surrogates.py`, `e16_models.py`, `e16_summary.py` | 成员位于 `runs/<platform>/`；`e16_members.csv`, `e16_agg.csv` |
| E17 | 宽三角替代函数的逐层放置（HWHM 1 或 4，只放在第 1 层，或只放在第 2-3 层；其余层为 0.25），种子 0-2（`w17_*`） | `e16_tasks.py`, `e16_train.py` | 成员位于 `runs/<platform>/`；`e16_members.csv`、`e16_agg.csv` 中的相应行 |
| E18 | 噪声注入与有界激活：训练期膜噪声（`w18_mem*`）、输入增广（`w18_aug*`）、tanh / 截断 ReLU / 二值 MLP（`w18_mlp_*`）、逃逸噪声随机前向（`w18_esc_*`）；种子 0-2 | `e16_tasks.py`, `e16_train.py` | 成员位于 `runs/<platform>/`；`e16_members.csv`、`e16_agg.csv` 中的相应行 |
| E19 | 仅分析：配对噪声网格上各噪声水平下的最优单一宽度（三个种子；种子 0 另含超宽成员；detach 变体） | `e21_w16.py --parts e22` | `e22_bestwidth_w16.csv` |

第二步，可达区域：权重空间，由逐单位矩计算（不训练）。

| 实验 | 计算内容 | 脚本 | 结果文件 |
|---|---|---|---|
| 矩 | 冻结预测的逐单位二阶矩：配对噪声网格（`--mode grid`）和原稿发布的条件，可选地放在无标签（`wfe`）或真方差（`wfo`）输入估计器之后（`--mode orig`）；`--tag` 选择新成员的文件 | `e2x_dump_paired.py`, `e2x_common.py` | `results/<platform>/{paired,orig}/moments_*.npz`（约 7 GB，不在发布中），`check_preds_*.json` |
| 核对 | 重新计算的风险与原稿发布文件的对照 | `e2x_verify.py` | `e2x_verify_orig.csv`, `e2x_verify_paired.csv` |
| E21 | 权重空间阶梯：W0（单纯形）、W1（单纯形加空候选）、W2 / W3（逐关节）、W4（全线性堆叠，岭惩罚）、W5（输入估计器）、W6（增广成员），对象为 D-S、D-M、D-T、B-F、B-L、B-A、B-N；D-M 与 D-S 的配对差；单纯形边际按尺度部分与角度部分的归因；`m` 切片（`sim`） | `e21_ladder.py` | `e21_ladder.csv`, `e21_ladder_seeds.csv`, `e21_paired.csv`, `e21_attribution.csv`, `e21_units.npz`；`sim`：`e21_mslice.csv`, `e21_mslice_rows.csv`；配对噪声版本（`--source paired`）：`e21_ladder_pairednoise.csv`, `e21_ladder_seeds_pairednoise.csv`, `e21_paired_pairednoise.csv`, `e21_attribution_pairednoise.csv`, `e21_units_pairednoise.npz` |
| E16-E18 成员上的 E21 | 宽度字典 D-W（每族四个宽度加 STE，普通与 detach 版）和单个旋钮成员的阶梯 | `e21_w16.py --parts e21` | `e21_ladder_w16.csv`, `e21_ladder_w16_seeds.csv`, `e21_paired_w16.csv`, `e21_attribution_w16.csv`, `e21_ladder_e18.csv` |
| E22 | 配对噪声网格上校准权重的噪声路径；残差矩阵的插值；预测与实测的临界水平；增长指数 `kappa`；风险交叉；权重表 `w*(rho)`；三次函数族的敏感性运行（`--family lqc`，后缀 `_lqc`） | `e22_path.py` | `e22_path.csv`, `e22_C.npz`, `e22_fit.json`, `e22_critical.csv`, `e22_kappa.csv`, `e22_crossings.csv`, `e22_wtable.csv`, `e22_predicted_risk.csv`, `e22_pairwise_roots.csv`；`e22_critical_lqc.csv`, `e22_predicted_risk_lqc.csv`, `e22_pairwise_roots_lqc.csv` |
| E16 成员上的 E22 | 宽度字典的路径、每个成员的风险随 `rho` 的变化、交叉点 | `e21_w16.py --parts e22` | `e22_path_w16.csv`, `e22_crossings_w16.csv`, `e22_bestwidth_w16.csv` |
| E23 | 无目标标签的校准：噪声方差取自无标签校准输入、真实噪声、静止机器人记录和权重表；误设的噪声（Student t、AR(1)） | `e23_dump.py`, `e23_labelfree.py` | `e23_labelfree.csv`, `e23_rows.csv`, `e23/e23_variances.json`（`e23/*.npz` 中的矩不在发布中） |
| E24 | 分样本增益证书（t 下界和自助法下界；认证率、覆盖率、认证率不低于 0.9 的最小 `m`） | `e24_certificate.py` | `e24_certificate.csv`, `e24_min_m.csv`, `e24_rows.csv` |
| E25 | 执行预算（至多 `k` 个执行成员加一个免费空候选，对照 W-Q(k)）、保留比例、逐关节支撑 | `e25_budget.py` | `e25_budget.csv`, `e25_budget_rows.csv`, `e25_retention.csv`, `e25_perjoint.csv` |
| E26 | 在校准数据上交叉拟合以选择权重空间；以分裂共形区间长度为对照 | `e26_select.py`, `e26_conformal.py` | `e26_rows.csv`, `e26_select.csv`；`e26_conformal.csv`, `e26_conformal_check.csv`, `e26_conformal_summary.csv` |
| E21-E26 汇总 | E21-E26 的关键数字 | `e2x_summary.py` | `e2x_summary.json` |

第三步，所需：仿真机械臂上的任务层。候选预测器列于 `e27_candidates.json`。

| 实验 | 计算内容 | 脚本 | 结果文件 |
|---|---|---|---|
| E27 滤波 | N、O1-M、P1 下各候选经因果低通滤波后的虚假交互力矩；相对 D-S 的交叉截止频率；残差的慢/白拆分；与 E13c 的核对 | `e27_filtered.py`（`e27_common.py`） | `e27_filtered.csv`, `e27_filtered_fine.csv`, `e27_crossing.csv`, `e27_slowwhite.csv`, `e27_e13c_check.json` |
| E27 外力矩 | 40 条带外关节力矩（阶跃和斜坡）的测试轨迹，在 N 和 O1-M 下评估：估计误差、ROC / AUC、固定阈值与校准阈值的事件检测；加速度取自记录（设定 `a`）或来自带噪速度的差分（设定 `b`） | `e27_gen_ext.py`, `e27_ext_eval.py`, `e27_ext_summarize.py` | `data_sim/ext_test.npz`（由 `e27_gen_ext.py` 写出）, `e27_ext_episodes.csv`, `e27_ext_summary.csv`, `e27_ext_roc.csv`, `e27_ext_meta_a.json`, `e27_ext_meta_b.json`, `e27_ext_meta_c.json`, `e27_ext_test_info.json` |
| E28 | 以冻结预测器作交互力矩估计器的关节空间导纳控制闭环：无人时的参考漂移，有人时的透明度和意图跟随；主设定 `c = 20`、真状态反馈诊断、阻尼 `c = 50`；负载 P1 下重新校准与否 | `e28_admittance.py`, `e28_followup.sh`, `e28_followup_c50.sh`, `e28_summarize.py` | `e28_episodes.csv`, `e28_summary.csv`, `e28_paired.csv`, `e28_params_c20.json`, `e28_params_c20_truefb.json`, `e28_params_c50.json`；逐种子的分块文件位于 `e28_parts/` |
| E29 | E13 设定上的跟踪补充：参考轨迹前馈、任务加权校准、E13 回合的重现；由残差的慢变部分和白噪声部分对跟踪误差做任务加权预测 | `e29_tracking.py`, `e29_taskweighted.py` | `e29_tracking.csv`, `e29_tracking_summary.csv`, `e29_tracking_paired.csv`, `e29_taskweighted.csv`, `e29_taskweighted_meta.json` |
| E27-E29 汇总 | E27-E29 的关键数字以及 H11、H12 的核对 | `e27_summarize_all.py` | `e27_29_summary.json` |
| 理论 | 补充材料理论部分的恒等式、引理和命题的数值核对 | `theory_selfcheck.py` | 终端输出（无结果文件） |

`results/results_manifest.csv` 将论文正文中引用的每个数值对应到其来源：每个数值一行，含章节和段落、量的名称、数值、平台和条件、来源文件、列名或键、筛选条件（角色、条件、组、平均方式）、备注，以及 `verified` 列（由根据来源文件重新计算该数值的核对给出 `PASS` 或 `FAIL`）。
各成员的配置（`config.json`）记录每个样本每层的平均脉冲数、轮数、墙钟时间和开发集误差。

## 运行顺序

所有命令均在 `code/` 下运行。`.sh` 启动脚本会切换到自身所在目录，并从 `$PY` 读取解释器（默认 `python3`）；原稿发布的结果从 `../results/<platform>/` 读取。原稿发布（E00-E14，冻结成员位于 `runs25/` 和 `runs25/<platform>/frozen.json`）是前提。

1. **冻结预测的矩**（配对噪声网格和原稿条件；原稿发布的成员）：
   `bash e2x_launch_dumps.sh > ../logs/e2x_dumps.log 2>&1`。
2. **原稿成员上的 E15**，每个平台各一次：
   `bash e15_run.sh sim ../runs25/sim` 和 `bash e15_run.sh sarcos ../runs25/sarcos`（位置、翻转、驻点），然后 `python e15_summary.py --platform sim` 和 `python e15_summary.py --platform sarcos`。Python 脚本接受 `--runs`、`--members`（glob）和 `--suffix`。
3. **E21-E26**：
   `bash e2x_launch_cpu1.sh > ../logs/e2x_cpu1.log 2>&1`（SARCOS 的 E21、核对、两个平台的 E24-E26）和 `python e21_ladder.py --platform sim --res ../results/sim --mslice`（仿真机械臂的 E21；没有启动脚本调用它）；然后 `bash e2x_launch_stage2.sh`（等待上述两个日志；E23 导出、E22、配对噪声上的 E21、核对、E23、再次运行 E24）；最后 `python e2x_summary.py --res_root ../results`。
4. **E16-E18 训练**：
   `python e16_tasks.py --out ../logs/e16_tasks.jsonl` 生成队列；`bash e16_launch.sh 5` 启动五个工作进程，队列清空后它们自行退出（`python e16_progress.py` 查看进度；机器空闲后 `bash e16_scale.sh 7` 增加工作进程数）；`python e16_summary.py` 生成 `e16_members.csv` 和 `e16_agg.csv`；`python e16_tables.py` 以及 `python e16_compare.py detach` 或 `wide` 输出 markdown 表；`python e16_check_detach.py` 检查 detach 变体。
5. **新成员上的 E15 与 E21 / E22**，每个平台各一次（`<p>` 为 `sim` 或 `sarcos`）：
   `bash e15_run.sh <p> ../runs/<p> 'w1*' _new`，然后 `python e15_summary.py --platform <p> --suffix _new`，并且运行一次 `python e15_width_summary.py --results ../results --suffix _new`；
   `bash e2x_launch_w16.sh`（新成员的矩；训练更多种子后若要替换已有文件，把 `--overwrite` 作为第一个参数），然后 `python e21_w16.py --platform <p> --res ../results/<p>`（只运行其中一部分时用 `--parts e21` 或 `--parts e22`）。
6. **E27-E29**（仿真机械臂；候选及其权重遵循 E13，并读取 `results/sim/e21_ladder_seeds.csv` 和原稿发布的 E13 结果文件）：
   `python e27_gen_ext.py --xml_dir ../mujoco_menagerie/franka_emika_panda --data_dir ../data_sim --out ../data_sim/ext_test.npz`（外力矩记录）；
   `python e27_filtered.py --data_dir ../data_sim --runs ../runs25/sim --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json --out_dir ../results/sim`；
   `python e27_ext_eval.py ...`（参数见模块 docstring），用 `--tag _a`、`_b`、`_c` 分块运行，然后 `python e27_ext_summarize.py ../results/sim`；
   主设定运行 `python e28_admittance.py ...`（`--c_bd 20`，每个种子一个输出文件，位于 `results/sim/e28_parts/`），然后 `bash e28_followup.sh` 和 `bash e28_followup_c50.sh`，再运行 `python e28_summarize.py ../results/sim`；
   分块运行 `python e29_tracking.py ...`，写出 `e29_tracking_a.csv`、`e29_tracking_b.csv`、`e29_tracking_c.csv`，然后 `python e29_taskweighted.py`；
   `python e27_summarize_all.py ../results/sim`。
7. **表格和图**：运行 `bash figures/make_all_revised.sh`（见下文）。

## 重新生成表格和图

原稿发布：在仓库根目录下，`bash code/figures/make_all.sh` 根据冻结的结果重新生成 Tables II 和 III、S1、S2、S3、S5、S8 和 S9 的表格、Figs. 1-4 以及补充图（`fig5.pdf`-`fig7.pdf`）；输出位于 `build/locked/`。
`bash code/figures/make_all.sh figs` 只重新生成图。解释器为 `$PY`（默认 `python3`）。

E15-E29：`bash code/figures/make_all_revised.sh`（可在任意目录运行）依次运行下列脚本。它们读取 `results/`，把矢量 PDF 写入仓库根目录下的 `build/fig/`，把 LaTeX 表写入 `build/tables/`；根目录可用环境变量 `SNN29_ROOT` 更改，设置 `SNN29_PNG=<dir>` 时还会写出 300 dpi 的 PNG 预览。该脚本先解压以 `.csv.gz` 存放的结果文件（`gunzip -k`）。每个脚本都导入 `common_style.py`（颜色、路径、辅助函数），也可单独运行（`python code/figures/<script>.py`）；`figS_position.py` 还会导入 `supp_tables_s5.py`。锁定协议（E00-E14）的表由 `make_all.sh` 写入 `build/locked/`，两套构建互不覆盖：

| 脚本 | 输出 | 读取 |
|---|---|---|
| `fig1_overview.py` | Fig. 1（示意图：问题、平面、三步） | 不读结果文件（示意性数值） |
| `fig2_plane.py` | Fig. 2（信号增益 / 正交噪声平面） | `e15_members.csv`, `e15_members_w16.csv`, `e21_ladder.csv`, `e21_ladder_e18.csv` |
| `fig3_width.py` | Fig. 3（替代函数宽度与直通估计器：响应、增益、误差随 HWHM 的变化；脉冲翻转的预测与实测） | `e15_members.csv`, `e15_members_new.csv`, `e15_flip.csv`, `e15_flip_new.csv`, `e15_flip_neurons.npz`, `e15_width_summary.csv`, `e16_members.csv`（以及 `_w16` 文件） |
| `fig4_ladder.py` | Fig. 4（权重空间阶梯、归因、校准预算） | `e21_ladder.csv`, `e21_attribution.csv`, `e21_mslice.csv` |
| `fig5_path.py` | Fig. 5（噪声路径、无标签校准、证书） | `e22_path.csv`, `e22_critical.csv`, `e22_crossings.csv`, `e23_labelfree.csv`, `e24_certificate.csv` |
| `fig6_task.py` | Fig. 6（仿真机械臂的任务层：滤波后的虚假力矩、外力矩估计与导纳漂移、跟踪误差的任务加权预测） | `e27_filtered_fine.csv`, `e27_crossing.csv`, `e27_ext_summary.csv`, `e28_paired.csv`, `e29_taskweighted.csv`, `e29_taskweighted_meta.json` |
| `figS_position.py` | `figS4.pdf`（E16 宽度扫描）、`figS5.pdf`（E15 翻转与方向分辨响应） | 通过 `supp_tables_s5.load` 读取 E15 表（`e15_*.csv`, `e15_*_new.csv`），以及 `e16_members.csv` |
| `figS_reachable.py` | `figS1.pdf`（噪声路径）、`figS2.pdf`（证书） | `e22_path.csv`, `e24_certificate.csv` |
| `tables_main.py` | `table2.tex`、`table3.tex`、`table4.tex` | `e21_ladder.csv`, `e21_ladder_w16.csv`, `e21_attribution.csv`, `e24_certificate.csv`, `e25_*.csv`；延迟行取自 `results/runtime/runtime.json` |
| `supp_tables_s3_s7.py` | `s3_*.tex`、`s7_*.tex` | `e21_*.csv`, `e22_*.csv`, `e23_labelfree.csv`, `e24_*.csv`, `e25_*.csv`, `e26_select.csv` |
| `supp_table_s7_conformal.py` | `s7_conformal.tex`（补充材料 S7：共形选择对照） | `e26_conformal.csv`, `e26_conformal_summary.csv` |
| `supp_tables_s5.py` | `s5n_*.tex`（补充材料 S5：成员诊断、宽度、复位因子、几何与翻转、驻点、逐层放置、噪声注入） | `e15_*.csv`, `e15_*_new.csv`, `e16_members.csv`, `e21_ladder{,_w16,_e18}.csv`, `e22_bestwidth_w16.csv` |
| `supp_tables_s8.py` | `s8n_*.tex`（补充材料 S8：任务滤波、交互力矩、导纳、跟踪） | `e27_*.csv`, `e27_29_summary.json`, `e28_*.csv` / `.json`, `e29_*.csv` / `.json` |

## 预注册

`prereg/hypotheses.md`（中文版和英文版在同一文件中）给出十二个假设 H1-H12、检验各假设的实验、判据，以及各终点属于主要还是次要终点。文件日期为 2026-11-11。其第一条声明是：判据在看到 E15-E30 的结果之前已经固定——判据取自研究方案（文件记录了方案的 SHA-256），运行之后不得修改；如需修改，只能以带日期的修订条目追加，并同时报告原判据和修订理由。
六个主要终点（H1 中宽度与前向响应的秩相关、H2 的配对风险差、H6 的尺度占比、H9 中无标签校准与有标签校准的风险差、H10 的认证率、H11 的漂移差）使用 Holm 校正；其余终点为次要终点，如实标注。
文件还列出哪些实验属于预注册（E15、E16、E17、E18、E19、E21、E22、E23、E24、E27、E28），哪些属于探索性或描述性（E20、E25、E26、E29、E30，以及原稿发布的 E09-E14），哪些结果保持原稿锁定协议下确认性结果的地位（P1-P4）。研究方案没有给出的操作化在文件中标为补充。文件中的一张表规定了各种结果模式的写法。
文件记录了注册时 `code/e27_candidates.json` 的 SHA-256；注册之后，候选清单追加了 E21 的最优解（D-S 和 D-M 在 W1、W3 下的解）；发布的 `e27_candidates.json` 是追加后的清单，修订记录及新的 SHA-256 见 `prereg/hypotheses.md`。可选实验 E20 和 E30 不在本次发布中。

## 说明

- 原稿发布的启动脚本和后处理脚本从 `code/` 运行，并要求 `runs/`、`logs/`、`results/`、`data_sim/` 和 `mujoco_menagerie/` 位于其旁边，SARCOS 文件位于上一级目录；解释器为 `$PY`（默认 `python3`）。`throttle.sh` 从 `logs/throttle_n` 读取并发上限。在本次发布中，`runs25/` 存放原稿发布的冻结成员，`runs/` 存放新成员（环境变量 `E16_RUNS` 可改变 `e16_train.py` 使用的目录）。
- **加载 E16-E18 成员。** 请使用 `e16_models.load_member(member_dir)`；它根据 `config.json` 重建任何成员，新旧皆可。用 `common.models.SNN` 重建新成员会得到错误的反向（峰值归一的替代函数 `sigmoid_pn` / `atan_pn`、逐层宽度 `triangular_layerwise`、`detach_reset`），对 `model_class = MLPAct` 还会得到错误的前向。E15 和 E21-E26 的脚本通过 `e16_models.build_from_config` 构建新成员。
- **结果文件的后缀。**
  `_pairednoise`：E21 在配对噪声矩上计算（`--source paired`），而不是原稿那样重新生成的噪声。E21 的主要结果使用重新生成的噪声，因此可与原稿文件逐数对照；配对噪声版本是稳健性核对。
  `_w16`：对 E16 成员（`w16_*`；以同一种子的 D-M 和 D-S 为参照）的 E21 和 E22，其中宽度字典 D-W 由种子 0-2 构成（见 `e21_ladder_w16.csv` 的 `seeds` 列）。`e15_*_w16` 文件来自更早的阶段性运行，每个平台只含当时已有的 65 个 `w16_*` 成员，已被 `_new` 文件取代。
  `_new`：训练结束后对全部 E16-E18 成员（`w16_*`、`w16d_*`、`w16c_*`、`w17_*`、`w18_*`）重跑 E15（运行顺序第 5 步）；它是 `e15_width_summary.py` 的默认后缀。
  `_lqc`：E22 的敏感性运行，残差矩阵的插值采用三次多项式函数族。
- 大于约 100 KB 的结果文件以 `<name>.csv.gz` 存放（用 `gunzip -k` 解压；`make_all_revised.sh` 会为图表脚本自动解压，而读取这类文件的分析脚本，例如 `e27_ext_summarize.py` 和 `e28_summarize.py`，需要先解压）。E15 的逐神经元数组 `e15_flip_neurons_new.npz` 和 `e15_flip_neurons_w16.npz` 不在发布中，可由 `e15_flip.py` 生成。`results/results_manifest.csv` 中部分 `source_file` 条目写的是以 `.csv.gz` 存放的文件的 `.csv` 名称。
- 矩文件约 7 GB，不在发布中。`bash e2x_launch_dumps.sh` 和 `bash e2x_launch_w16.sh` 可重新生成；使用 `--check` 时，每次导出都会把重新计算的预测与已存的 `pred_*.npy` 比较（`check_preds_*.json`）。`e2x_common.MomFile` 用于读取矩文件。
- E09-E12、E13 和 E14 是对冻结成员、校准集和测试集的事后分析。它们解释 P1-P4 的结果，而不检验这些结果。E15-E29 中哪些属于预注册、哪些属于探索性，见 `prereg/hypotheses.md`。
- E15 和 E21-E29 不训练模型；只有 E16-E18 训练新成员。E27-E29 和 `e27_candidates.json` 按名称引用原稿发布的成员（`dm_*`、`ds_*`、`bn_*`、`ba_*`），并读取原稿发布的 E13 结果文件（`e13_episodes_*.csv`、`e13c_lowfreq_matched.csv`）。
- `e28_followup.sh` 还包含一个 `B_d = 50` 的调用，其中人的刚度随 `B_d` 放大（`K_h = 5 B_d`）。该设定在每个意图回合中都失稳，已被废弃；`e28_followup_c50.sh` 采用主运行的人模型（`--kh_ratio 2 --bh_ratio 0.13333333`）写出同名输出文件，必须在 `e28_followup.sh` 之后运行。
- `e28_summarize.py` 合并 `results/sim/e28_parts/e28_episodes*_s*.csv` 及其 `*_params.json`。`e28_admittance.py` 的 docstring 中的示例输出路径（`../results/sim/e28_episodes_a.csv`）不符合此模式；请把逐种子的输出写入 `e28_parts/`。
- `theory_selfcheck.py` 从其所在目录（`code/`）导入 `common.models` 和 `common.surrogates`；请在该目录下运行 `python theory_selfcheck.py`。缺少 PyTorch 时跳过引理 1 的检查。在笔记本 CPU 上运行约需一到三分钟。
- `e27_29_summary.json` 中的 `NaN` 按 Python `json` 模块的默认方式写出（不是严格的 JSON）。
- Table II 的成本（`figures/tables.py`）在仿真机械臂的运行上测得。训练列是 `config.json` 中记录的墙钟时间，对五个训练随机种子取平均，对象为所部署的模型或字典：B-F 取每个随机种子的一次 `dm_sigmoid_*` 运行（9.4 min），D-S 和 D-M 取该字典的五次运行，D-T 取产生快照的那一次运行。在开发集上选择替代函数时，五个 D-M 成员训练了一次（36.1 min）；这一成本由 B-F、B-A、D-S 和 D-T 共同承担，在表的标题中说明，不计入该列。执行列是在 RTX 3090 和一个 CPU 线程上测得的、所执行成员的批大小为 1 的延迟（`results/runtime/runtime.json`）。
- E09-E14 中新增成员以及 E16-E18 成员的训练时间取决于并发和限速，不应用于成本比较。
