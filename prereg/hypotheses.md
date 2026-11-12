# 预注册：传感噪声下冻结脉冲逆动力学预测器的使用（假设 H1–H12）

**时间戳：2026-11-11**
**来源**：`research_plan.md` 第 4 节（假设与判据）、5 节（实验判据）、5.5 节（不同结果下的写法）、7.5 节（统计与预注册）。该方案文件的 SHA-256 为 `1be5b2c13ad2b3e4e5040780d84ac1926f3ec917206e6681103a2c3834c93a41`；E27 候选清单 `code/e27_candidates.json` 的 SHA-256 为 `455d0e2c59ef5d3c2321969f4d4775d0444d1f0d980c720801ebc3fd245a8fc0`。

**修订（2026-11-11，E21 完成后）**：候选清单在注册后追加了 E21 在 W1、W3 下 D-S、D-M 的最优解作为额外候选（方案 5.4 节原本就要求加入），追加后的 SHA-256 为 `20591c9f3f5898a0f34b04e3be99055a79c1dddc2b44968e90d7d210ae7999c6`；原有候选与判据未改。另：E16 每个配置训练了 3 个种子，方案中“进入主要结论的配置补到 5 个”未执行，因为没有任何中间宽度进入主要结论；STE 成员沿用原稿的 5 个种子。

文中标有【补充】的内容是本文件在整理时对方案原文所作的操作化补充（方案没有给出数值或规则）。它们在冻结前须由作者确认，确认之前不应视为预注册内容。其余内容均来自方案原文。

---

# 中文版

## 0 声明

1. **判据先于结果。** H1–H12 的陈述与判据在 `research_plan.md`（第 4、5、7.5 节）中于任何 E15–E30 运行开始之前写定。本文件整理于 2026-11-11，当日首批 E15、E16、E27 作业已在远程机器上启动；整理本文件时没有读取、汇总或查看它们的任何输出，本地 `results/` 目录为空。预注册的含义是：判据在看到新实验结果之前固定。
2. **哪些数据已被看到。** 方案中标【已核算】的数值（附录 A）由原稿的表格反推，属于原稿的数据，不是 E15–E30 的结果。H2 的依据（STE 的风险在仿真臂 ρ≈0.84、SARCOS ρ≈0.59 起低于最好的窗口成员）、H6 中"尺度占比"的预期（32%–100%）、H12 的交叉点区间（12–21 Hz）都来自这些已核算的数值，因此这三个假设相对原稿数据不是盲的，它们的预注册意义仅限于：新实验（E16、E19、E21、E27）的结果在判据固定之前没有被看到。
3. **判据在实验运行前固定。** 本文件所列判据在 E15–E30 运行前固定；运行后不得修改判据。若须修改，只能以带日期的修订条目追加，并在论文中同时报告原判据和修订理由。
4. **与原稿的关系。** 原稿的 P1–P4 保留为"原稿锁定协议下的确认性结果"；E09–E14 仍标为探索性；E15–E30 中在运行前已写明判据的实验，标为"新稿预注册"（见第 3 节）。

## 1 总体设计

- 论证链三步：位置（成员在"信号增益 β – 正交噪声功率 ‖ν‖²"平面上的落点，H1–H5）→ 可达（权重空间决定的可达区域，H6–H10）→ 所需（任务滤波器决定的最优点，H11–H12）。
- 数据范围：SARCOS 公开数据与 MuJoCo 仿真臂，两个平台。不含新采集的实机数据。
- 条件：N 名义；O1-L/M/H 逐步独立的高斯观测噪声（仿真臂 ρ=0.35/0.7/1.0，加在 q、q̇；SARCOS ρ=0.5/0.7，加在全部 21 个通道）；O2-B 每条轨迹一个常值偏置；P1 1 kg 负载。
- 种子：每个配置 3 个训练种子，进入主要结论的配置补到 5 个（E16）；原稿的成员为 5 个种子。

## 2 假设、检验、判据与终点标记

终点标记：**主要**＝方案 7.5 节的六个主要终点之一，参与 Holm 校正；**次要**＝如实标注、不参与校正。

### H1　宽度–增益律　【主要终点：宽度与前向响应 r 的秩相关】

- **陈述**：前向响应 r 与噪声下的信号增益 β 随代理的有效宽度单调下降；四个代理族（三角、sigmoid、ATan 型、逃逸率）在统一的宽度轴（半高半宽 HWHM）上落到同一条曲线；STE 是宽度趋于无穷的极限。
- **检验实验**：E16（宽度网格 HWHM ∈ {0.0625, 0.125, 0.25, 0.5, 1, 2, 4} 加 STE；每配置 3 个种子，主要结论配置补到 5 个；两个平台）。
- **判据**：每个族内宽度与 r 的 Spearman 秩相关 ≤ −0.8，且同一宽度下族间差异小于种子标准差的两倍。
- **终点定义**：主要终点为各族、各平台的宽度与 r 的秩相关。【补充】八个（族 × 平台）单元全部满足判据时记为 H1 成立；β 随宽度的单调性按描述性结果报告，不进入判据。
- **统计**：秩相关及其 95% 聚类自助法区间；族间差异用各宽度下的种子标准差作尺度。

### H2　匹配宽度　【主要终点：O1-M 下最优中间宽度相对原稿选定宽度与相对 STE 的配对风险差】

- **陈述**：噪声水平 ρ 下风险最小的宽度随 ρ 单调增大，量级与诱导输入噪声 σ_in(ρ) 一致；在 O1-M 下存在一个中间宽度，同时优于原稿选定的宽度和 STE。
- **检验实验**：E16、E19（由 E16 的模型计算各噪声水平下的最优宽度，与 σ_in(ρ) 的中位数比较；在 E22 的噪声网格上重复）。
- **判据**：两个平台上，风险最小的宽度随 ρ 单调不减且至少出现三个不同的最优宽度；O1-M 下最优中间宽度相对原稿选定宽度和相对 STE 的配对差的 95% 区间都不含零。
- **终点定义**：主要终点为 O1-M 下上述两个配对风险差（nMSE 单位）。【补充】两个平台、两个比较全部满足时记为成立。"量级与 σ_in(ρ) 一致"没有数值判据，按描述性结果报告。
- **统计**：配对差，聚类自助法（见 3.1）。

### H3　层级　【次要终点】

- **陈述**：输入增益主要由第一层的代理决定。
- **检验实验**：E17（三角族，一个窄宽度和两个宽宽度；三种放置：只第一层宽、只第二三层宽、全部宽；3 个种子，两个平台）。
- **判据**：只改第一层能复现全部改宽时 r 变化的 70% 以上，只改第二三层不超过 30%。

### H4　只在反向的噪声先验效率低　【次要终点】

- **陈述**：在相同的信号增益下，前向注入噪声（膜噪声或输入增广）得到的正交噪声更低；宽代理的兑换率最差。
- **检验实验**：E18（五种方式：只在反向的宽代理；前向膜噪声 σ∈{0.25,0.5,1,2}；前向输入噪声；随机前向；有界激活对照 tanh、截断 ReLU、二值激活加直通估计）。
- **判据**：在匹配的 β 处，前向膜噪声与前向输入噪声两种方式的风险低于只在反向的方式，且配对区间不含零。若 tanh MLP 的增长指数 κ 与脉冲成员相当，则"饱和式衰减"归因于有界激活而非脉冲。
- **统计**：配对差，聚类自助法；"匹配 β"的具体做法（插值或最近邻）【补充】须在运行前确定。

### H5　机理　【次要终点】

- **陈述**：宽度通过第一层单元的信噪比分布起作用：宽代理使 SNR_k 更高、翻转率更低；方向分辨的响应差集中在输入的低方差方向。
- **检验实验**：E15（零训练，对原稿全部成员）、E16。
- **判据**：预测翻转率与实测翻转率在成员间的秩相关 ≥ 0.8，才把 SNR_k 作为机理写进正文。"响应差集中在低方差方向"没有数值判据，按描述性结果报告。

### H6　约束代价　【主要终点：D-M 相对 D-S 的单纯形边际中的尺度占比】

- **陈述**：所有噪声条件下，每个干净训练的方法（含 MLP）尺度代价为正，名义条件下约为零；松弛后 D-M 相对 D-S 的边际在仿真臂上消失，在 SARCOS 上保留角度部分。
- **检验实验**：E21（权重空间阶梯 W0–W6；单模型基线 B-F、B-L、B-A、B-N；字典 D-S、D-M、D-T）。
- **判据**：方案原文给出陈述和预期数（原稿反推：SARCOS O1-M 32%、O1-H 42%，仿真臂 O1-M 100%、O1-H 95%），没有数值阈值。【补充】操作化如下：（a）每个干净训练方法在每个噪声条件下的尺度代价的 95% 区间下界大于零；名义条件下区间含零或点估计小于噪声条件下对应值的 10%；（b）加 null 后（W1）D-M 减 D-S 的配对差，仿真臂的 95% 区间含零，SARCOS 的区间不含零且角度部分大于零。
- **终点定义**：主要终点为单纯形（W0）下 D-M 减 D-S 的边际中尺度部分所占比例（式 I2 的分解），在 O1-M、O1-H 两个条件和两个平台上报告。【补充】判定以 (a)(b) 为准。
- **统计**：配对差与分解，聚类自助法。

### H7　各向异性　【次要终点】

- **陈述**：SARCOS 上保留的角度部分在逐关节权重或输入估计器下消失 90% 以上，并与 E15 的方向分辨响应一致。
- **检验实验**：E21、E15。
- **判据**：角度部分在逐关节权重（W2/W3）或输入估计器（W5）下消失 90% 以上。与 E15 一致的判据没有数值，按描述性结果报告。

### H8　噪声路径　【次要终点】

- **陈述**：由三个噪声水平拟合的 C(ρ) 预测的临界水平，与实测的支撑变化点相差不超过一个网格步长。
- **检验实验**：E22（ρ 从 0 到 1.2，SARCOS 到 0.9，步长 0.05）。
- **判据**：预测与实测的临界水平之差不超过 0.05（一个网格步长）。

### H9　无标签校准　【主要终点：无目标标签校准与有标签校准的测试风险之差】

- **陈述**：观测噪声条件下，无目标标签校准与有标签校准的测试风险之差不超过种子标准差；负载条件下显著变差。
- **检验实验**：E23（五种校准来源；条件 O1-L/M/H、O2-B，以及作为反例的 P1）。
- **判据**：O1-L/M/H 与 O2-B 下，来源 2（名义带标签数据加模拟噪声，噪声方差由目标条件的无标签输入估计）相对来源 1（目标条件带标签记录）的 nMSE 差不超过种子标准差；P1 下该差显著为正（配对区间不含零）。
- **终点定义**：【补充】主要终点取 O1-M 下来源 2 减来源 1 的 nMSE 差，两个平台分别判定，均满足时记为成立；其余条件（O1-L、O1-H、O2-B、P1）与来源 3–5 为次要。"种子标准差"取来源 1 风险在 5 个训练种子间的标准差。
- **统计**：配对差，聚类自助法。

### H10　证书　【主要终点：分样本下界在 10 条轨迹时对 O1-M 的增益认证率】

- **陈述**：分样本下界在 10 条轨迹时对 O1-M 的增益认证率 ≥ 90%，经验覆盖率不低于名义水平。
- **检验实验**：E24（校准轨迹 5/5 分样本并交叉互换；单侧 95% 下界；SARCOS 以 100 行为一个块）。
- **判据**：认证率 ≥ 90%；经验覆盖率不低于名义水平（单侧 95%，即不低于 95%）。
- **终点定义**：主要终点为 m=10、O1-M 下的认证率；【补充】两个平台分别判定，均满足时记为成立。覆盖率为同一假设的一部分，与认证率一并判定，但不单独计入 Holm。
- **统计**：认证率为二项比例；用对种子与校准分组的聚类自助法给出区间。

### H11　虚假力矩与漂移　【主要终点：导纳控制下无人作用时的参考位置漂移差】

- **陈述**：O1-M 下，做了衰减的候选经 5 Hz 滤波后的虚假交互力矩和导纳漂移大于种子字典，每个种子都成立。
- **检验实验**：E27（开环交互力矩估计）、E28（导纳控制闭环：无人作用时参考位置 10 s 内的漂移；条件 N、O1-M、P1；每个条件 30 段）。
- **判据**：O1-M 下，每个衰减型候选的虚假力矩（5 Hz 滤波后）与导纳漂移都大于种子字典，在每个种子上都成立。"做了衰减的候选"以 `code/e27_candidates.json` 中标记 `attenuating` 的候选为准（见上方 SHA-256）。
- **终点定义**：主要终点为 E28 在 O1-M 下无人作用时的 10 s 参考位置漂移，衰减型候选减种子字典。【补充】"每个种子都成立"按种子逐一比较，全部种子、全部衰减型候选满足时记为成立；虚假力矩（E27）作为次要终点同时报告。
- **统计**：配对差，种子 × 段两个方向的自助法，闭环区间沿用原稿（对三个校准分组在种子–段单元内取平均）。

### H12　任务滤波器决定排名　【次要终点】

- **陈述**：截止频率从 2 Hz 扫到 40 Hz，衰减型候选相对种子字典的力矩估计误差由更差变为更好，交叉点落在 L_eq 预测的 12–21 Hz 附近。
- **检验实验**：E27（零成本部分：对原稿已有的测试残差，按截止频率 2–40 Hz 逐一滤波；新仿真：40 条带外力矩的测试轨迹）。
- **判据**：出现由"更差"到"更好"的符号变化，且交叉点的截止频率落在 12–21 Hz 内。"附近"没有给出容差，【补充】取 12–21 Hz 区间本身，不加容差。

## 3 统计方法

### 3.1 抽样与区间

- 沿用原稿的聚类自助法：训练种子与测试单位（轨迹）独立重采样，方法之间共享下标；权重在每个条件的完整校准集上拟合；区间为 95% 百分位区间。比较在方法之间共享校准分组、测试单位、参考轨迹和噪声实现。重复次数沿用原稿协议。
- 采样单位是轨迹，不是行；SARCOS 上以 100 行为一个块。

### 3.2 主要终点与多重性

- **主要终点六个**：H1 的秩相关、H2 的配对差、H6 的尺度占比、H9 的风险差、H10 的认证率、H11 的漂移差。
- 六个终点之间做 **Holm 校正**，家族错误率 α=0.05。【补充】对阈值型判据（H1、H6 尺度占比、H10），把"判据在其边界值上成立"作为原假设，由聚类自助法得到单侧 p 值后做 Holm 步降；对配对差型判据（H2、H9、H11），p 值由配对差的聚类自助法得到。文中报告的 95% 区间是未校正的展示区间，判定以 Holm 校正后的结果为准。
- 一个主要终点涉及多个平台或多个比较时，【补充】要求全部满足（交并检验），不再对其内部做校正。H2 的方案原文已明确"两个平台上"。
- **其余为次要终点**（H3、H4、H5、H7、H8、H12，以及各主要终点之外的条件与来源），如实标注，不做校正。

### 3.3 预注册范围

| 类别 | 内容 |
|---|---|
| 新稿预注册（判据在运行前已写明） | E15（SNR 机理判据）、E16、E17、E18、E19、E21、E22、E23、E24、E27、E28 |
| 探索性或描述性（方案没有判据） | E20（可选，仅当 H1 在 E16 成立才做）、E25、E26、E29、E30（可选）；E09–E14 |
| 原稿锁定协议下的确认性结果 | P1–P4 |

## 4 不同结果下的写法（承诺）

| 结果 | 正文写法 |
|---|---|
| H1、H2 都成立 | 标题可用 "Surrogate width sets signal gain"；宽度–增益律与匹配宽度作为 C1 的主结果 |
| H1 成立、H2 不成立 | 报告宽度–增益律；说明宽代理是低效的衰减器，最优宽度不随噪声移动 |
| H1 不成立（非单调或依赖形状） | 报告"训练条件之外形状有影响"，这是对 [ZV21] 结论适用范围的直接补充；保留"有窗口对无窗口"的二分和引理 SG-1 |
| E18 显示有界 MLP 也次二次 | 把"饱和式衰减"写成有界激活的性质，脉冲网络是其中一例 |
| H6 成立、H7 不成立（角度部分不被逐关节权重或输入估计器消除） | 报告为代理多样性保留的、不能由衰减替代的部分；对多代理字典有利，论证链不变 |
| H11 成立、H12 的交叉点偏离预测 | 报告实测交叉点；L_eq 的预测降为定性说明，并用 E29 的逐关节分析解释偏差 |

三步中任何一步的结果变化都只改变该步的结论强度，不改变章节结构。

## 5 需作者在冻结前确认的事项

1. 声明 0.1 的时间顺序（判据先于结果）。
2. 所有标【补充】的操作化：H1 的单元判定；H2 的双平台、双比较；H4 的"匹配 β"；H6 的 (a)(b)；H9 的主要条件与"种子标准差"；H10 的双平台；H11 的逐种子判定；H12 的容差；Holm 的 p 值构造；交并检验规则；自助法重复次数。
3. 方案没有给出数值判据的描述性陈述：H2 的量级一致、H5 的低方差方向、H7 与 E15 的一致性。

---

# English version

## 0 Declarations

1. **Criteria precede results.** The statements and criteria of H1–H12 were fixed in `research_plan.md` (Sections 4, 5, 7.5) before any E15–E30 run was started. This file was compiled on 2026-11-11, on which day the first E15, E16 and E27 jobs had already been launched on the remote machine; none of their outputs had been read, summarized or inspected when this file was written, and the local `results/` directory was empty. Preregistration here means that the criteria were fixed before any result of the new experiments was seen.
2. **Which data have already been seen.** Values marked "computed" in the plan (Appendix A) are back-calculated from the tables of the original manuscript; they are data of the original manuscript, not results of E15–E30. The rationale of H2 (the STE member's risk falls below the best window member from rho about 0.84 on the simulated arm and about 0.59 on SARCOS), the expected scale share in H6 (32% to 100%), and the crossing interval in H12 (12 to 21 Hz) come from these computed values. These three hypotheses are therefore not blind with respect to the original data. Their preregistration only guarantees that the results of the new experiments (E16, E19, E21, E27) were not seen when the criteria were fixed.
3. **Criteria are fixed before the experiments run.** The criteria in this file are fixed before E15–E30 are run and must not be changed afterwards. Any change must be appended as a dated amendment, and the paper must report both the original criterion and the reason for the change.
4. **Relation to the original manuscript.** P1 to P4 of the original manuscript stay as "confirmatory results under the locked protocol". E09 to E14 stay exploratory. Among E15 to E30, experiments whose criteria were written down before running are labelled "preregistered for the new manuscript" (Section 3.3).

Items marked [Added] are operationalizations added while preparing this file, where the plan gives no number or rule. They must be confirmed by the authors before freezing and are not preregistered until then. Everything else is taken from the plan.

## 1 Overall design

- Three-step argument chain: position (where a member lies in the plane of signal gain beta versus orthogonal-noise power ||nu||^2; H1 to H5), reach (the region reachable under a weight space; H6 to H10), requirement (the optimum set by the task filter; H11 and H12).
- Data scope: the public SARCOS data and a MuJoCo simulated arm, two platforms. No newly collected real-robot data.
- Conditions: N nominal; O1-L/M/H independent Gaussian observation noise at every step (simulated arm rho = 0.35/0.7/1.0 on q and q-dot; SARCOS rho = 0.5/0.7 on all 21 channels); O2-B a constant per-trajectory offset; P1 a 1 kg payload.
- Seeds: 3 training seeds per configuration, raised to 5 for configurations entering the main conclusions (E16); the original members have 5 seeds.

## 2 Hypotheses, tests, criteria and endpoint labels

Endpoint labels: **primary** means one of the six primary endpoints of plan Section 7.5, included in the Holm correction; **secondary** means reported as such and not corrected.

### H1  Width-gain law  [primary endpoint: rank correlation between width and forward response r]

- **Statement**: The forward response r and the signal gain beta under noise decrease monotonically with the effective surrogate width; the four surrogate families (triangular, sigmoid, ATan-type, escape-rate) fall on one curve on the common width axis (half width at half maximum, HWHM); STE is the limit of infinite width.
- **Test**: E16 (grid HWHM in {0.0625, 0.125, 0.25, 0.5, 1, 2, 4} plus STE; 3 seeds per configuration, raised to 5 for main-conclusion configurations; both platforms).
- **Criterion**: Within each family the Spearman rank correlation between width and r is at most -0.8, and the difference between families at equal width is smaller than twice the seed standard deviation.
- **Endpoint definition**: The primary endpoint is the rank correlation per family and platform. [Added] H1 holds when all eight (family x platform) cells meet the criterion. The monotonicity of beta in width is reported descriptively and is not part of the criterion.
- **Statistics**: Rank correlation with a 95% cluster-bootstrap interval; between-family differences scaled by the seed standard deviation at each width.

### H2  Matched width  [primary endpoint: paired risk differences at O1-M between the best intermediate width and (i) the width selected in the original manuscript, (ii) STE]

- **Statement**: The width with minimal risk at noise level rho increases monotonically with rho and its magnitude agrees with the induced input noise sigma_in(rho); under O1-M there is an intermediate width that beats both the width selected in the original manuscript and STE.
- **Test**: E16 and E19 (compute the optimal width per noise level from the E16 models and compare with the median of sigma_in(rho); repeat on the E22 noise grid).
- **Criterion**: On both platforms the risk-minimizing width is non-decreasing in rho and at least three distinct optimal widths occur; under O1-M the 95% intervals of the paired differences between the best intermediate width and the originally selected width, and between it and STE, both exclude zero.
- **Endpoint definition**: The primary endpoint is these two paired risk differences (nMSE units) at O1-M. [Added] H2 holds when both comparisons hold on both platforms. "Magnitude agrees with sigma_in(rho)" has no numerical criterion and is reported descriptively.
- **Statistics**: Paired differences, cluster bootstrap (Section 3.1).

### H3  Layer hierarchy  [secondary]

- **Statement**: The input gain is determined mainly by the surrogate of the first layer.
- **Test**: E17 (triangular family; one narrow and two wide widths; three placements: wide only in layer 1, wide only in layers 2 and 3, wide everywhere; 3 seeds, both platforms).
- **Criterion**: Changing only the first layer reproduces at least 70% of the change in r obtained by widening everywhere, and changing only layers 2 and 3 reproduces at most 30%.

### H4  Backward-only noise prior is inefficient  [secondary]

- **Statement**: At equal signal gain, injecting noise in the forward pass (membrane noise or input augmentation) yields lower orthogonal noise; wide surrogates have the worst exchange rate.
- **Test**: E18 (five modes: backward-only wide surrogate; forward membrane noise sigma in {0.25, 0.5, 1, 2}; forward input noise; stochastic forward; bounded-activation controls tanh, clipped ReLU, binary activation with straight-through estimator).
- **Criterion**: At matched beta, the risk of forward membrane noise and of forward input noise is lower than that of the backward-only mode, with paired intervals excluding zero. If the growth exponent kappa of the tanh MLP is comparable to that of the spiking members, the saturating attenuation is attributed to bounded activations rather than to spiking.
- **Statistics**: Paired differences, cluster bootstrap. [Added] The way "matched beta" is obtained (interpolation or nearest neighbour) must be fixed before running.

### H5  Mechanism  [secondary]

- **Statement**: Width acts through the signal-to-noise ratio distribution of first-layer units: wide surrogates give higher SNR_k and lower flip rates; direction-resolved response differences concentrate in low-variance input directions.
- **Test**: E15 (no training, all original members) and E16.
- **Criterion**: The rank correlation between predicted and measured flip rates across members is at least 0.8; only then is SNR_k written into the main text as the mechanism. "Concentrate in low-variance directions" has no numerical criterion and is reported descriptively.

### H6  Cost of the constraint  [primary endpoint: scale share of the D-M minus D-S simplex margin]

- **Statement**: Under all noise conditions every cleanly trained method (including the MLP) has a positive scale cost, and about zero under the nominal condition; after relaxation the margin of D-M over D-S vanishes on the simulated arm and retains an angular part on SARCOS.
- **Test**: E21 (weight-space ladder W0 to W6; single-model baselines B-F, B-L, B-A, B-N; dictionaries D-S, D-M, D-T).
- **Criterion**: The plan gives the statement and expected numbers (back-calculated from the original manuscript: SARCOS O1-M 32%, O1-H 42%; simulated arm O1-M 100%, O1-H 95%) but no numerical threshold. [Added] Operationalization: (a) for each cleanly trained method and each noise condition, the lower end of the 95% interval of the scale cost is above zero; under the nominal condition the interval contains zero or the point estimate is below 10% of the corresponding noisy value; (b) for the paired difference D-M minus D-S after adding a null candidate (W1), the 95% interval contains zero on the simulated arm, and excludes zero with a positive angular part on SARCOS.
- **Endpoint definition**: The primary endpoint is the share of the scale part in the D-M minus D-S margin under the simplex (W0), using the decomposition of identity (I2), reported at O1-M and O1-H on both platforms. [Added] The decision is made by (a) and (b).
- **Statistics**: Paired differences and the decomposition, cluster bootstrap.

### H7  Anisotropy  [secondary]

- **Statement**: On SARCOS the retained angular part vanishes by more than 90% under per-joint weights or under the input estimator, consistent with the direction-resolved response of E15.
- **Test**: E21 and E15.
- **Criterion**: The angular part vanishes by more than 90% under per-joint weights (W2/W3) or the input estimator (W5). Consistency with E15 has no numerical criterion and is reported descriptively.

### H8  Noise path  [secondary]

- **Statement**: The critical noise level predicted from C(rho) fitted at three noise levels differs from the measured support change point by at most one grid step.
- **Test**: E22 (rho from 0 to 1.2, to 0.9 on SARCOS, step 0.05).
- **Criterion**: The absolute difference between predicted and measured critical level is at most 0.05 (one grid step).

### H9  Label-free calibration  [primary endpoint: test-risk difference between label-free and labelled calibration]

- **Statement**: Under observation noise, the test-risk difference between calibration without target labels and calibration with labels does not exceed the seed standard deviation; it is significantly worse under the payload condition.
- **Test**: E23 (five calibration sources; conditions O1-L/M/H, O2-B, and P1 as the counterexample).
- **Criterion**: Under O1-L/M/H and O2-B, the nMSE difference of source 2 (nominal labelled data plus simulated noise whose variance is estimated from unlabelled target inputs) relative to source 1 (labelled recording in the target condition) does not exceed the seed standard deviation; under P1 the difference is significantly positive (paired interval excludes zero).
- **Endpoint definition**: [Added] The primary endpoint is the nMSE difference source 2 minus source 1 at O1-M, judged separately on each platform and required on both. The other conditions (O1-L, O1-H, O2-B, P1) and sources 3 to 5 are secondary. "Seed standard deviation" is the standard deviation of the source-1 risk over the 5 training seeds.
- **Statistics**: Paired differences, cluster bootstrap.

### H10  Certificate  [primary endpoint: certification rate of the split-sample lower bound at 10 trajectories under O1-M]

- **Statement**: With 10 trajectories the split-sample lower bound certifies the gain under O1-M in at least 90% of cases, and the empirical coverage is not below the nominal level.
- **Test**: E24 (calibration trajectories split 5/5 with swap; one-sided 95% lower bound; on SARCOS a block of 100 rows is the unit).
- **Criterion**: Certification rate at least 90%; empirical coverage not below the nominal level (one-sided 95%, that is, at least 95%).
- **Endpoint definition**: The primary endpoint is the certification rate at m = 10 under O1-M. [Added] Judged separately on each platform and required on both. Coverage is part of the same hypothesis and is judged together with the rate, but is not counted separately in the Holm correction.
- **Statistics**: The rate is a binomial proportion; intervals from a cluster bootstrap over seeds and calibration groups.

### H11  Spurious torque and drift  [primary endpoint: reference-position drift difference under admittance control without human input]

- **Statement**: Under O1-M, attenuating candidates produce larger spurious interaction torque (after 5 Hz filtering) and larger admittance drift than the seed dictionary, in every seed.
- **Test**: E27 (open-loop interaction-torque estimation) and E28 (admittance closed loop: reference-position drift over 10 s without human action; conditions N, O1-M, P1; 30 episodes per condition).
- **Criterion**: Under O1-M, each attenuating candidate has larger spurious torque (5 Hz filtered) and larger admittance drift than the seed dictionary, in every seed. "Attenuating candidates" are those flagged `attenuating` in `code/e27_candidates.json` (SHA-256 above).
- **Endpoint definition**: The primary endpoint is the 10 s reference-position drift of E28 at O1-M, attenuating candidate minus seed dictionary. [Added] "Every seed" is judged seed by seed, and H11 holds when all seeds and all attenuating candidates satisfy it. Spurious torque (E27) is reported as a secondary endpoint alongside.
- **Statistics**: Paired differences; bootstrap over seeds and episodes; closed-loop intervals follow the original manuscript (averaging the three calibration groups within each seed-episode cell).

### H12  The task filter decides the ranking  [secondary]

- **Statement**: As the cutoff frequency is swept from 2 Hz to 40 Hz, the torque-estimation error of attenuating candidates relative to the seed dictionary changes from worse to better, with the crossing near the 12 to 21 Hz predicted by L_eq.
- **Test**: E27 (zero-cost part: filter the existing test residuals of the original manuscript at cutoffs 2 to 40 Hz; new simulation: 40 test trajectories with external torque).
- **Criterion**: A sign change from worse to better occurs and the crossing cutoff lies within 12 to 21 Hz. "Near" has no stated tolerance; [Added] the interval 12 to 21 Hz is taken as is, with no tolerance.

## 3 Statistical methods

### 3.1 Resampling and intervals

- Cluster bootstrap as in the original manuscript: training seeds and test units (trajectories) are resampled independently with shared indices across methods; weights are fitted on the full calibration set of each condition; intervals are 95% percentile intervals. Methods share calibration groups, test units, reference trajectories and noise realizations. The number of replicates follows the original protocol.
- The sampling unit is the trajectory, not the row; on SARCOS a block of 100 rows is the unit.

### 3.2 Primary endpoints and multiplicity

- **Six primary endpoints**: the rank correlation of H1, the paired difference of H2, the scale share of H6, the risk difference of H9, the certification rate of H10, the drift difference of H11.
- **Holm correction** across the six endpoints at family-wise alpha = 0.05. [Added] For threshold-type criteria (H1, the H6 scale share, H10) the null is that the criterion holds exactly at its boundary value, with one-sided p-values from the cluster bootstrap followed by Holm step-down; for paired-difference criteria (H2, H9, H11) the p-values come from the cluster bootstrap of the paired difference. The 95% intervals shown in the paper are uncorrected display intervals; decisions follow the Holm-corrected results.
- When one primary endpoint involves several platforms or comparisons, [Added] all of them must be satisfied (intersection-union rule) and no further correction is applied inside the endpoint. The plan already says "on both platforms" for H2.
- **All others are secondary endpoints** (H3, H4, H5, H7, H8, H12, and the conditions and sources outside each primary endpoint), labelled as such and not corrected.

### 3.3 Preregistration scope

| Category | Content |
|---|---|
| Preregistered for the new manuscript (criteria written before running) | E15 (SNR mechanism criterion), E16, E17, E18, E19, E21, E22, E23, E24, E27, E28 |
| Exploratory or descriptive (no criterion in the plan) | E20 (optional, only if H1 holds in E16), E25, E26, E29, E30 (optional); E09 to E14 |
| Confirmatory under the locked protocol of the original manuscript | P1 to P4 |

## 4 Commitments on how results are written up

| Result | How the main text will read |
|---|---|
| H1 and H2 both hold | The title may read "Surrogate width sets signal gain"; the width-gain law and matched width are the main result of contribution C1 |
| H1 holds, H2 does not | Report the width-gain law; state that wide surrogates are inefficient attenuators and the optimal width does not move with noise |
| H1 fails (non-monotone or shape-dependent) | Report that shape matters outside the training condition, a direct addition to the scope of the conclusion of [ZV21]; keep the window versus no-window dichotomy and Lemma SG-1 |
| E18 shows a bounded MLP is also sub-quadratic | Write the saturating attenuation as a property of bounded activations, of which spiking networks are one instance |
| H6 holds, H7 does not (the angular part is not removed by per-joint weights or the input estimator) | Report it as the part retained by surrogate diversity that attenuation cannot replace; this favours the multi-surrogate dictionary and leaves the argument chain unchanged |
| H11 holds, the crossing of H12 deviates from the prediction | Report the measured crossing; downgrade the L_eq prediction to a qualitative remark and explain the deviation with the per-joint analysis of E29 |

A change of result in any of the three steps changes only the strength of that step's conclusion, not the section structure.

## 5 Items for the authors to confirm before freezing

1. The timing stated in Declaration 0.1 (criteria fixed before results).
2. All [Added] operationalizations: the cell rule of H1; the two-platform, two-comparison rule of H2; "matched beta" in H4; (a) and (b) of H6; the primary condition and "seed standard deviation" of H9; the two-platform rule of H10; the seed-by-seed rule of H11; the tolerance of H12; the construction of p-values for Holm; the intersection-union rule; the number of bootstrap replicates.
3. Descriptive statements for which the plan gives no numerical criterion: the magnitude agreement of H2, the low-variance directions of H5, the consistency of H7 with E15.
