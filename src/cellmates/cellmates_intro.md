# CellMates 项目架构与核心算法详解（基于 `cellmates --input reproducibility/demo.h5ad --num-processors 4 --predict-cn --n-states 8`）

> 本文是对上一版回答的**扩展细化版**，目标是：
> 1. 讲清每个目录/文件的职责；
> 2. 讲清默认命令一次执行时跨文件的调用关系；
> 3. 精读 JCB + Neighbor Joining + HMM-EM 的核心实现，并把主要数学公式推导还原出来。

---

## 0. 先给一个总览：这个项目到底在做什么？

`cellmates` 的核心任务是：

1. 从单细胞拷贝数相关数据（通常是 AnnData `.h5ad`）中读取观测；
2. 对每一对细胞 \((v,w)\) 做一个“隐变量四元组模型（r,u,v,w）”的 EM 推断，估计 3 条边长：
   - \(l_{ru}\)：root 到 pair 共同祖先/centroid 的长度
   - \(l_{uv}\)：centroid 到细胞 \(v\) 的长度
   - \(l_{uw}\)：centroid 到细胞 \(w\) 的长度
3. 把所有 pair 的三元长度汇总成 CTR 距离张量 \((n\times n\times3)\)；
4. 用邻接式合并（neighbor joining 风格）重建整棵树；
5. 可选：在树上继续做内部节点 CN profile 的 Viterbi 预测。

换句话说，它是一个“**pairwise EM 估计 + 全局树重建 + 可选内部状态重建**”流水线。

---

## 1. 文件结构（逐目录逐文件细化说明）

以下基于 `src/cellmates/` 主包。

---

### 1.1 顶层包

#### `src/cellmates/__init__.py`
- 作用：定义项目根路径 `ROOT_DIR`。
- 用途：在某些场景中，供其他模块做相对路径定位。

---

### 1.2 命令行入口 `bin/`

#### `src/cellmates/bin/__init__.py`
- 空包初始化文件。

#### `src/cellmates/bin/core.py`
- **命令行入口主文件（最关键入口）**。
- 主要函数：
  - `parse_args()`：定义 CLI 参数（`--input`, `--n-states`, `--predict-cn`, `--num-processors`, `--jc-correction`, `--alpha` 等）
  - `main()`：
    1. 解析参数；
    2. 当 `num_processors > 1` 时设置 `multiprocessing` 启动方式为 `spawn`；
    3. 调用 `run_inference_pipeline(**vars(args))`。
- 由打包入口点 `cellmates = cellmates.bin.core:main` 直接调用。

---

### 1.3 推断主流程 `inference/`

#### `src/cellmates/inference/__init__.py`
- 空文件。

#### `src/cellmates/inference/pipeline.py`
- **端到端 pipeline 调度器（最重要流程控制文件）**。
- 关键函数：
  - `load_and_prepare_adata(...)`：读 `.h5ad`，检查 layer，必要时做 CNAsim readcount 校正。
  - `prepare_observations(...)`：抽取观测矩阵、染色体分段边界、细胞名，构建观测模型对象（`NormalModel` / `JitterCopy`）。
  - `run_em_inference(...)`：创建 `JCBModel` + `EM`，执行 EM 并返回结果对象。
  - `save_results(...)`：保存距离矩阵、Newick 树、cell names。
  - `predict_cn_profiles(...)`：在已重建树上对内部节点做 Viterbi CN 预测。
  - `run_inference_pipeline(...)`：把所有步骤串联起来。

#### `src/cellmates/inference/em.py`
- **EM 算法主体实现**（pairwise quadruplet）。
- 核心类：`EM`
  - `fit(...)`：对所有细胞对 `(s,t)` 执行推断，支持单进程/多进程。
- 关键函数：
  - `fit_quadruplet(...)`：单个 pair 的 EM 循环（E-step + M-step + 收敛判断）。
  - `_fit_em`, `_fit_copy_obs`, `_fit_copy_obs_async`：不同并行/调度实现。
  - `estimate_theta_from_cn(...)`：可从 CN profile 初始化三段长度参数。

#### `src/cellmates/inference/neighbor_joining.py`
- **树构建模块**。
- 关键函数：
  - `build_tree(ctr_table, ...)`：从 `n x n x 3` 的三元距离估计树。
  - `_build_tree_rec(...)`：递归合并 OTU，维护中间距离结构并生成边。

---

### 1.4 模型定义 `models/`

#### `src/cellmates/models/__init__.py`
- 空文件。

#### `src/cellmates/models/obs.py`
- **观测模型层（发射概率）**。
- 抽象基类：`ObsModel`
  - 定义 `log_emission`, `log_emission_split`, `update`, `M_step`, `initialize`, `new` 等接口。
- 主要实现：
  - `NormalModel`：高斯读数模型（均值与 CN 成线性关系，精度为 `tau`）
  - `PoissonModel`：泊松读数模型
  - `JitterCopy`：离散 CN 观测噪声模型（用于 copy-number 直接输入时）
  - `NegBinomialModel`：负二项模型（可扩展）

#### `src/cellmates/models/quadruplet.py`
- 四元组相关建模/工具（在当前主命令链中不是最核心入口）。

#### `src/cellmates/models/evo/basefunc.py`
- **进化模型基础概率函数**。
- 包括：
  - `p_delta_change(...)`：JCB 下“变化/不变化”概率；
  - `p_delta_trans_mat(...)`, `p_delta_start_prob(...)`：构建转移张量和初始概率；
  - `h_eps(...)` 等：CopyTree 模型相关 zipping 概率构造。

#### `src/cellmates/models/evo/__init__.py`
- **进化模型主实现（非常核心）**。
- 抽象基类：`EvoModel`
  - `forward_backward`, `forward_pass`, `viterbi_path`
  - `_expected_changes`, `multi_chr_expected_changes`
- 具体模型：
  - `JCBModel`：你提到的 JCB 模型（长度参数化 + 闭式 M-step）
  - `CopyTree`：另一套参数化模型（epsilon）
  - `SimulationEvoModel`：用于仿真生成 CN 演化数据

---

### 1.5 工具层 `utils/`

#### `src/cellmates/utils/__init__.py`
- 空文件。

#### `src/cellmates/utils/hmm.py`
- **HMM 算法工具核心文件**。
- 包括：
  - `_forward_backward_pomegranate` / `_forward_backward_broadcast`
  - `_forward_likelihood_*`, `_backward_pass_*`
  - `viterbi_decode_pomegranate`
- 功能：在三链隐状态 HMM 上计算后验统计（供 EM E-step）和解码路径（供 CN 预测）。

#### `src/cellmates/utils/math_utils.py`
- 数学辅助：
  - `l_from_p`, `p_from_l`：长度与变化概率转换；
  - `compute_cn_changes`：统计 CN 序列变化；
  - `viterbi_matrix_K6`, `viterbi_optimized_K5`：自实现 Viterbi。

#### `src/cellmates/utils/tree_utils.py`
- 树结构/格式转换工具：
  - `nxtree_to_newick`, `newick_to_nx`
  - `write_cells_to_tree`
  - `get_ctr_table`
  - 各类 relabel、DendroPy/NetworkX 转换

#### `src/cellmates/utils/testing.py`
- 测试辅助函数。

#### `src/cellmates/utils/visual.py`
- 结果可视化辅助。

---

### 1.6 数据与仿真

#### `src/cellmates/simulation/__init__.py`
- 空文件。

#### `src/cellmates/simulation/datagen.py`
- 合成数据生成：
  - 随机树生成
  - CN 演化模拟
  - 观测生成（Normal/Poisson）
  - 生成 AnnData 结构

#### `src/cellmates/common_helpers/cnasim_data.py`
- CNAsim 数据桥接：
  - CNAsim 输出文件解析
  - 转 AnnData
  - tree 信息、clone 信息整合
  - readcount 校正

#### `src/cellmates/common_helpers/README.md`
- helper 说明文档。

---

### 1.7 其他方法适配

#### `src/cellmates/other_methods/dice_api.py`
- DICE 方法接口。

#### `src/cellmates/other_methods/medicc2_api.py`
- MEDICC2 方法接口。

---

## 2. 文件之间的逻辑关系 / 调用关系（以默认命令执行一次为例）

命令：

```bash
cellmates --input reproducibility/demo.h5ad --num-processors 4 --predict-cn --n-states 8
```

---

### 2.1 入口层

1. Shell 执行 `cellmates`
2. 命中 package entry point：`cellmates.bin.core:main`
3. `parse_args()` 拿到参数：
   - `input = reproducibility/demo.h5ad`
   - `num_processors = 4`
   - `predict_cn = True`
   - `n_states = 8`
   - 其余默认值（如 `max_iter=30`, `rtol=1e-3`）
4. 因为 `num_processors > 1`，设置多进程启动方式为 `spawn`
5. 调 `run_inference_pipeline(**vars(args))`

---

### 2.2 Pipeline 层（`inference/pipeline.py`）

#### 步骤 A：读入与预处理
- `load_and_prepare_adata(...)`
  - `anndata.read_h5ad(...)`
  - 若有 `cnasim-params`，调用 `correct_readcounts`
  - 校验需要的 layer 存在

#### 步骤 B：观测矩阵与观测模型
- `prepare_observations(...)`
  - 默认 `use_copynumbers=False`：
    - 从 `copy` layer 抽数据
    - `obs` shape 为 `(n_bins, n_cells)`（注意转置）
    - 观测模型为 `NormalModel`
  - 若 `use_copynumbers=True`：
    - 用 `state` layer
    - 观测模型变为 `JitterCopy`

#### 步骤 C：EM 主推断
- `run_em_inference(...)`
  - 初始化 `JCBModel(n_states=8, alpha=..., chromosome_ends=...)`
  - 构建 `EM(...)`
  - 调 `em.fit(...)`

#### 步骤 D：建树
- `build_tree(em.distances)`
  - `em.distances` 即 pairwise 三元距离张量
  - 输出 `networkx.DiGraph`

#### 步骤 E：结果保存
- `save_results(...)`
  - `distance_matrix.npy`
  - `tree.nwk`
  - `cell_names.txt`

#### 步骤 F（可选）：内部节点 CN 预测
- `predict_cn_profiles(...)`
  - 后序遍历内部节点
  - 每个内部节点构局部发射并做 Viterbi
- `save_cn_profiles(...)`
  - 保存 `predicted_copy_numbers.npz`

---

### 2.3 EM 内部调用链（再细一层）

`EM.fit(obs, ...)` 的核心流程：

1. 枚举所有细胞对 `(s,t)`，数量是 \({n\choose2}\)
2. 每个 pair 进入 `fit_quadruplet(s,t, obs[:,[s,t]], ...)`
3. `fit_quadruplet` 循环迭代直到收敛：

#### E-step
- `quad_model.multi_chr_expected_changes(obs_vw, obs_model)`
  - 分染色体（如果有 `chromosome_ends`）
  - 每段调用 `_expected_changes(...)`
  - `_expected_changes` 调 `forward_backward(...)`
  - `forward_backward` 最终进入 `utils/hmm.py` 执行前向后向
  - 输出：
    - `d`: 三条边期望变化次数
    - `dp`: 三条边期望不变化次数
    - `loglik`: 当前参数下对数似然

#### M-step
- `quad_model.update(d, dp)`：更新 JCB 边长 \(l_{ru},l_{uv},l_{uw}\)
- `obs_model.update(...)`：若观测参数可训练则更新（例如 Normal 的 `mu/tau`）

#### 收敛判据
- 相邻两次 log-likelihood 相对提升小于 `rtol`
- 且迭代次数超过 `min_iter`

4. 所有 pair 结果汇总到 `l_hat[s,t,:]`
5. `em.distances = l_hat`

---

## 3. JCB + NJ + HMM-EM：核心文件、关键函数、逐段解释与数学推导

下面按“模型定义 → E-step → M-step → 全局建树”给出最关键逻辑。

---

### 3.1 JCB 模型的概率定义（`models/evo/basefunc.py`）

#### 3.1.1 `p_delta_change(n_states, l, change, alpha)`

这函数编码了 JCB 的单边变化概率。记状态总数 \(K=n\_states\)：

- 不变（`change=False`）概率：

\[
P_{same}(l)=\frac{1}{K}+\frac{K-1}{K}e^{-K\alpha l}
\]

- 变到“某个其他差分态”（`change=True`）概率：

\[
P_{diff}(l)=\frac{1}{K}-\frac{1}{K}e^{-K\alpha l}
\]

这和 Jukes-Cantor 风格非常一致：
- 边长越大，指数项衰减，状态越趋向均匀；
- 边长越小，保持原状态概率更高。

---

#### 3.1.2 `p_delta_trans_mat(...)` 与 `p_delta_start_prob(...)`

- `p_delta_trans_mat` 构造四维局部转移核（索引 `[j', j, i', i]`）
- `p_delta_start_prob` 构造首位点概率（索引 `[j, i]`）

在 `JCBModel` 中会把三条边（ru, uv, uw）的局部核通过 `einsum` 组合成 6 维总转移：

\[
P(i',j',k'\mid i,j,k)=P_{ru}(i'\mid i)\,P_{uv}(j'\mid j,i',i)\,P_{uw}(k'\mid k,i',i)
\]

（在代码中通过张量广播与下标映射实现）

---

### 3.2 HMM 的隐变量结构（`models/evo/__init__.py` + `utils/hmm.py`）

每个位点 \(m\) 的隐状态可以看成三元组：

\[
Z_m=(C_m^u, C_m^v, C_m^w)
\]

其中根 \(r\) 一般固定 diploid（CN=2），作为约束注入 ru 分支。

观测层（以两叶子为观测）对应：

\[
\log p(y_m^v, y_m^w\mid C_m^v=i, C_m^w=j)
\]

`ObsModel.log_emission(obs_vw)` 返回 shape `(M,K,K)` 的 log 发射矩阵。

---

### 3.3 E-step 细化：forward-backward 与期望计数

#### 3.3.1 `EvoModel.forward_backward(...)`
- 输入：`obs_vw` + `obs_model` + 当前 `theta`
- 生成发射 `log_emissions`
- 调 `_forward_backward_pomegranate` 或 `_forward_backward_broadcast`
- 输出：
  - `expected_counts`（两切片后验累计，6维）
  - `log_gamma`（单切片后验，4维，按位点）
  - `log_p`（log likelihood）

---

#### 3.3.2 `EvoModel._expected_changes(...)`

这是 E-step 的“统计汇总”核心。

它把 `expected_counts` 与 `log_gamma` 通过 mask 分解为：

- \(d_e\)：边 e 的期望变化次数
- \(d'_e\)：边 e 的期望不变化次数

其中：
- `e=0` 对应 ru；
- `e=1` 对应 uv；
- `e=2` 对应 uw。

并且首位点（m=1）也补入对应变化/不变化贡献（因为链首没有前位点，需要单独处理）。

数学上就是：

\[
d_e=\sum_m \sum_{x_{m-1},x_m} \xi_m(x_{m-1},x_m)\,\mathbf{1}(\text{edge }e\text{ changed})
\]

\[
d'_e=\sum_m \sum_{x_{m-1},x_m} \xi_m(x_{m-1},x_m)\,\mathbf{1}(\text{edge }e\text{ unchanged})
\]

---

#### 3.3.3 多染色体处理 `multi_chr_expected_changes(...)`

由于真实数据是多染色体，跨染色体不应强制马尔可夫连续。实现方式：

1. 用 `chromosome_ends` 把观测切成多个子序列；
2. 每个子序列独立跑 `_expected_changes`；
3. 把各染色体的 `d`, `dp`, `loglik` 累加。

这是非常关键的工程细节，避免把 chr 末端到下一 chr 起始当作连续转移。

---

### 3.4 M-step 细化：JCB 闭式更新（`JCBModel.update`）

代码中：

```python
log_arg = 1 - K/(K-1) * d/(dp+d)
l = -(1/(alpha*K)) * log(log_arg)
```

#### 推导还原

定义：

\[
\hat p_e = \frac{d_e}{d_e+d'_e}
\]

模型理论变化概率（每条边）：

\[
p_e(l_e)=\frac{K-1}{K}(1-e^{-K\alpha l_e})
\]

EM 的 M-step 等价于把经验变化率匹配模型变化率：

\[
\hat p_e=p_e(l_e)
\]

解出：

\[
\hat p_e=\frac{K-1}{K}(1-e^{-K\alpha l_e})
\]
\[
e^{-K\alpha l_e}=1-\frac{K}{K-1}\hat p_e
\]
\[
l_e=-\frac{1}{K\alpha}\log\left(1-\frac{K}{K-1}\hat p_e\right)
\]

把 
\hat p_e=d_e/(d_e+d'_e)
代回即可得到代码更新式。

这就是 JCB 在当前框架里最核心的闭式推断点。

---

### 3.5 `fit_quadruplet(...)` 的 EM 主循环（逐段语义）

`inference/em.py` 里的 `fit_quadruplet` 可以概括为：

1. 拷贝初始化参数 `theta_init`，实例化本 pair 的 `quad_model` 和 `obs_model`；
2. while 循环直到收敛/达到 `max_iter`：
   - E-step：`d, dp, new_loglik = quad_model.multi_chr_expected_changes(...)`
   - 检查似然是否收敛（相对改变量 < `rtol`）
   - M-step：
     - `quad_model.update(exp_changes=d, exp_no_changes=dp)`
     - `obs_model.update(...)`
   - 记录 diagnostics（可选）
3. 返回 `(v,w), theta, loglik, it, obs_model, diagnostic_data`

这段函数对应论文里最核心的“每个 cell pair 的局部 EM 优化器”。

---

### 3.6 HMM 实现细节（`utils/hmm.py`）

#### 3.6.1 `_forward_backward_pomegranate(...)`

主要操作：

1. 将 `(M,K,K)` 发射扩展/重排到 pomegranate 期望的 `(1,M,K^3)`；
2. 将 6 维转移 `(K,K,K,K,K,K)` reshape 成 2 维 `(K^3,K^3)`；
3. 调 `DenseHMM.forward_backward(...)`；
4. 将结果 reshape 回：
   - expected_counts: `(K,K,K,K,K,K)`
   - marginal/log_gamma: `(M,K,K,K)`

这就是“数学上三链 HMM，工程上压平为单链大状态 HMM”的标准做法。

---

#### 3.6.2 `_forward_backward_broadcast(...)`

纯 numpy 对数域实现：
- 前向：log-sum-exp 递推
- 后向：反向 log-sum-exp
- 两切片后验 \(\xi\) 与单切片后验 \(\gamma\) 归一化

优点：可控、可 debug；
代价：可能内存更重，速度视场景而定。

---

### 3.7 邻接重建（`inference/neighbor_joining.py`）

#### 3.7.1 输入是什么？

`ctr_table[v,w,:] = [l_ru, l_uv, l_uw]`（仅上三角有效）。

这不是普通两两距离矩阵，而是每个 pair 的“共同祖先到根 + 两侧分支”三元信息。

#### 3.7.2 `build_tree(...)` 的数据结构

- `otus`：当前活跃 OTU 集合
- `ctr`：每对 OTU 的 centroid-to-root 距离
- `ntc`：node-to-centroid 距离（方向敏感）
- `ntr`：node-to-root 估计

#### 3.7.3 `_build_tree_rec(...)` 递归逻辑

每轮：
1. 选一对最优 pair（代码用 `max(ctr.items(), key=...)`）
2. 合并成新 OTU
3. 更新与其余 OTU 的距离关系
4. 递归直到只剩 2 个 OTU
5. 回溯拼装边，必要时将负边长截断到 0（代码有 warning）

最终得到 `nx.DiGraph`。

---

### 3.8 预测内部节点 CN（`predict_cn_profiles`）

当 `--predict-cn` 开启时，pipeline 在建树后进一步执行：

1. 将树节点重标成整数（便于矩阵索引）；
2. 根节点 CN 固定为 2；
3. 后序遍历内部节点：
   - 取该节点两个孩子（叶子或已预测内部节点）
   - 按边长设置 `evo_model.theta=[root->u, u->child1, u->child2]`
   - 组装发射概率
   - 调 `viterbi_decode_pomegranate` 求最优三链路径
   - 路径中父状态作为内部节点 CN，子状态用于必要时覆盖/补全
4. 保存所有节点（叶+内部）的 CN 矩阵。

这一步是“在已知树上做状态重建”。

---

## 4. 对你关心的“文章中的核心算法”给出最短索引

如果你要做论文复现/答辩，建议最先盯这 8 处：

1. `bin/core.py::main`（命令入口）
2. `inference/pipeline.py::run_inference_pipeline`（总流程）
3. `inference/em.py::EM.fit`（pairwise 批量调度）
4. `inference/em.py::fit_quadruplet`（单 pair EM）
5. `models/evo/__init__.py::EvoModel._expected_changes`（E-step统计核心）
6. `models/evo/__init__.py::JCBModel.update`（M-step闭式公式）
7. `utils/hmm.py::_forward_backward_pomegranate`（后验统计来源）
8. `inference/neighbor_joining.py::build_tree`（从 pair 到整树）

---

## 5. 公式与代码变量的一一对应表（便于读源码）

- \(K\) ↔ `n_states`
- \(\theta=(l_{ru}, l_{uv}, l_{uw})\) ↔ `quad_model.theta`
- \(Y\)（观测） ↔ `obs_vw`
- \(\log p(Y|C)\) ↔ `obs_model.log_emission(...)`
- \(\xi\)（两切片后验） ↔ `expected_counts`
- \(\gamma\)（单切片后验） ↔ `log_gamma`
- \(d,d'\) ↔ `exp_changes`, `exp_no_changes`
- \(\ell=\log p(Y|\theta)\) ↔ `loglik`

---

## 6. 一次执行的“时间线式”心智模型（建议你记住）

1. 读入数据，抽出每 bin 每细胞观测；
2. 对每对细胞做局部三边长度 EM；
3. 得到上三角 `n x n x 3` 长度张量；
4. 递归聚合得到全局树；
5. 可选地在树上回填内部 CN；
6. 输出矩阵 + Newick +（可选）内部 CN。

---

## 7. 你下一步最值得做的两件事

1. **先读懂 `fit_quadruplet` + `JCBModel.update`**：这是论文 EM 的落地核心。  
2. **再读 `neighbor_joining.build_tree`**：这是从局部 pair 参数到全局树结构的关键桥梁。

如果你愿意，我下一步可以再补一份：
- `fit_quadruplet` 逐行注释版（每一行做什么、输入输出维度是什么）
- `JCBModel.update` 的完整极大化推导（从 Q 函数写到闭式解）
- `utils/hmm.py` 中 forward/backward 的张量维度流图（非常适合排查 bug）

