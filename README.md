# 《A quantum computing concept for 1-D elastic wave simulation》完整复现实验方案

## 0. 目标、边界与成功标准

论文：arXiv:2312.14747v2。

本方案复现以下内容：

1. 正文 Figure 1：128 维量子状态（作者称 128 grid-point problem）、ODE、20 shots 理想模拟器、1000 shots 理想模拟器。
2. 正文 Figure 2：7 个内部网格点、ODE、1000 shots 理想模拟器、作者归档的 IBM Brisbane 1000 shots 真机数据。
3. 附录 Figure A1：Figure 2 算法对应的四阶段量子线路图。
4. 一个较小的 7-grid 冒烟实验，以及 100/1000/10000 shots 收敛实验。

注意：仓库 README 把线路图叫作“Figure 2”，但论文 PDF 中线路图实际是 **Figure A1**。论文真正的 Figure 2 是 IBM Brisbane 对比图。

成功标准：

- Figure 1 精确运行生成 `data_0.pkl`、`data_1.pkl`、`data_2.pkl` 和 PNG。
- Figure 1 的 1000-shots 平均时间相对 L2 误差应约为 5%。
- Figure 2 重绘的时间点应为 0、0.0002、0.0011、0.0015、0.0017 秒。
- Figure A1 线路应为 7 qubits、7 classical bits，包含 StatePreparation、PauliEvolution、Observable、Measurement 四段。

## 1. 已验证硬件和软件

已验证平台：NVIDIA DGX Spark，Ubuntu 24.04，aarch64，GB10，约 121 GiB RAM。

固定环境：

```text
Python 3.12
qiskit 1.0.0
qiskit-aer 0.14.2
qiskit-ibm-runtime 0.19.0
qiskit-experiments 0.7.0
numpy 1.26.4
cvxpy 1.6.7
setuptools 80.10.2
```

Qiskit Aer 在本机使用 CPU。官方 `qiskit-aer-gpu` wheel 面向 x86_64，不能直接装到 Spark 的 ARM64。不要为了本实验修改驱动、CUDA 或系统 Python，也不要在未验证的情况下编译 Aer GPU。

## 2. 安全规则

- 全部内容放在 `~/reproduction/Quantum-Wave-Equation-Solver-f64503a`。
- 使用独立 Conda 环境 `qcws-paper-f64503a`。
- 不使用 sudo，不修改系统 CUDA/驱动，不删除其他目录。
- SSH 密码只在交互登录或本地凭据文件中使用，绝不能写进脚本、日志或 Git。
- 精确 Figure 1 会运行约 2 小时 22 分钟。必须使用 `nohup`，避免 SSH 断开导致任务终止。

## 3. 获取固定版本代码

仓库：`https://github.com/malteschade/Quantum-Wave-Equation-Solver`

固定提交：

```text
f64503aa6135057b4a61647b704d87bea248004a
```

网络正常时：

```bash
mkdir -p ~/reproduction
cd ~/reproduction
git clone https://github.com/malteschade/Quantum-Wave-Equation-Solver.git Quantum-Wave-Equation-Solver-f64503a
cd Quantum-Wave-Equation-Solver-f64503a
git checkout f64503aa6135057b4a61647b704d87bea248004a
git rev-parse HEAD
```

最后一条必须输出上述提交号。Spark 无法访问 GitHub 时，在联网电脑检出该提交，再用 SFTP/SCP 上传完整目录。

## 4. 创建环境

推荐先安装 Miniconda ARM64，然后：

```bash
~/miniconda3/bin/conda create -n qcws-paper-f64503a python=3.12 numpy=1.26.4 scipy matplotlib cvxpy=1.6.7 -y
~/miniconda3/bin/conda run -n qcws-paper-f64503a python -m pip install \
  qiskit==1.0.0 \
  qiskit-aer==0.14.2 \
  qiskit-ibm-runtime==0.19.0 \
  qiskit-experiments==0.7.0 \
  setuptools==80.10.2
```

验证：

```bash
~/miniconda3/bin/conda run -n qcws-paper-f64503a python - <<'PY'
import qiskit, qiskit_aer, qiskit_ibm_runtime, qiskit_experiments, numpy, cvxpy
print('qiskit', qiskit.__version__)
print('aer', qiskit_aer.__version__)
print('runtime', qiskit_ibm_runtime.__version__)
print('experiments', qiskit_experiments.__version__)
print('numpy', numpy.__version__)
print('cvxpy', cvxpy.__version__)
from qiskit_aer import AerSimulator
print('devices', AerSimulator().available_devices())
PY
```

输出中 `devices` 只有 `CPU` 在 DGX Spark ARM64 上是预期行为。

## 5. 必须应用的 ARM64 层析补丁

原代码在 `utility/tomography.py` 中用默认 `ThreadPoolExecutor()` 同时执行 18 个 CVXPY 密度矩阵拟合。Spark 上会在 Aer 完成后无 Python traceback 直接退出，且不生成 `data_1.pkl`。

把：

```python
with ThreadPoolExecutor() as executor:
```

改为：

```python
# CVXPY native solvers are unstable when 18 reconstructions run concurrently
# on ARM64. Serial execution is slower but reproducible and stays within RAM.
with ThreadPoolExecutor(max_workers=1) as executor:
```

验证：

```bash
grep -n 'ThreadPoolExecutor' utility/tomography.py
git diff -- utility/tomography.py
```

不要把 worker 数恢复为 18。单线程每个时间点约 2 分 40 秒，内存约 12–13 GiB。

## 6. Figure 1 精确复现

作者配置和基准数据位于：

```text
examples/20240403T212351/configs.json
```

核心参数：19 个时间点，`dt=0.001`，内部 `nx=63`；变换后的量子态长度为 128，因此论文称 128-grid-point problem。求解器是 ODE、20 shots 本地理想模拟器、1000 shots 本地理想模拟器。

使用本交付脚本：

```text
.codex_tools/reproduce_paper_figure1.py
```

将脚本复制到仓库的某个结果根目录。脚本通过 `RESULT.parents[1]` 定位仓库，因此推荐严格使用：

```bash
cd ~/reproduction/Quantum-Wave-Equation-Solver-f64503a
mkdir -p data/figure1_exact
cp /path/to/reproduce_paper_figure1.py data/figure1_exact/
cd data/figure1_exact
nohup env PYTHONPATH=~/reproduction/Quantum-Wave-Equation-Solver-f64503a \
  ~/miniconda3/envs/qcws-paper-f64503a/bin/python reproduce_paper_figure1.py \
  > figure1_exact_stdout.log 2>&1 < /dev/null &
echo $!
```

脚本把本地模拟器的 `batch_size` 改为 `100000`，使每个 shots 配置的 2304 条线路进入一个 Aer job。否则默认分成 24 个 job，会严重过度并行。

监控：

```bash
tail -f figure1_exact_stdout.log
ps -p PID -o pid,etime,stat,%cpu,%mem,rss,cmd
```

正常阶段顺序：

```text
ODE -> 20-shot 2304 circuits -> tomography 1..18
    -> 1000-shot 2304 circuits -> tomography 1..18
    -> save data_2.pkl -> render PNG -> write summary
```

已验证结果：

```text
elapsed_seconds=8549.274180
shots=20,mean_time_rl2=0.27468514,max_time_rl2=0.57080071
shots=1000,mean_time_rl2=0.049174923,max_time_rl2=0.11207713
```

本机已生成的完整脚本：[reproduce_paper_figure1.py](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_tools\reproduce_paper_figure1.py)

本机结果：[Figure 1](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_cache\paper_figure1_exact\paper_figure1_reproduced.png)；[摘要](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_cache\paper_figure1_exact\paper_figure1_summary.txt)。

## 7. 正文 Figure 2：IBM Brisbane 真机数据

正确数据集是：

```text
examples/20231203T101131/
```

不是 `20231202T182229`。判断依据是论文图中的时间点为 `0、0.0002、0.0011、0.0015、0.0017 s`，与 `20231203T101131` 的 `dt=0.0001` 和索引 `[0,2,11,15,17]` 精确对应。

三个文件含义：

```text
data_0.pkl  ODE
data_1.pkl  理想量子模拟器，1000 shots
data_2.pkl  IBM Brisbane 真机，1000 shots
```

使用：[reproduce_paper_true_figure2.py](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_tools\reproduce_paper_true_figure2.py)

运行：

```bash
cd ~/reproduction/Quantum-Wave-Equation-Solver-f64503a
mkdir -p data/figure2_true
cp /path/to/reproduce_paper_true_figure2.py data/figure2_true/
cd data/figure2_true
env PYTHONPATH=~/reproduction/Quantum-Wave-Equation-Solver-f64503a \
  ~/miniconda3/envs/qcws-paper-f64503a/bin/python reproduce_paper_true_figure2.py
```

这一步读取作者保存的真实 IBM 测量数据并重算/重绘，不会重新访问 IBM 云。重新跑真机需要 IBM Quantum 凭据，而且今日硬件校准与论文时不同，不能期待逐点一致。

已验证数值：

```text
ideal_1000 mean_all=0.037338588, max_all=0.067098833
ibm_brisbane_1000 mean_all=0.81338344, max_all=1.4838982
```

论文称真机误差“up to 60%”是概略描述；对仓库归档数据按论文公式逐时间点重算，某些低范数时刻的相对误差会超过 100%。不要修改或裁剪数据来强行得到 60%。

本机结果：[真正的 Figure 2](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_cache\paper_true_figure2\paper_true_figure2_reproduced.png)；[摘要](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_cache\paper_true_figure2\paper_true_figure2_summary.txt)。

## 8. 附录 Figure A1（线路图）

使用：[reproduce_paper_figure2.py](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_tools\reproduce_paper_figure2.py)

脚本名称沿用早期命名，但产物对应论文 **Figure A1**。脚本只生成线路，不执行 shots 或 tomography。

关键代码：

```python
import utility.circuits as circuits
circuits.SIMPLE_CIRCUITS = True
generator = CircuitGen1DA(experiment.logger, backend=None)
groups = generator.tomography_circuits(
    solver.st.get_state(0), solver.tf.h, solver.times[1:2],
    config['backend']['synthesis'], 100000,
    config['backend']['optimization'], config['backend']['seed'], False,
)
circuit = groups[0][0]
figure = circuit.draw(output='mpl', fold=-1)
```

`SIMPLE_CIRCUITS=True` 只影响绘图表现，使四个算法阶段清晰显示，不改变 Figure 1 的数值运行。

预期：

```text
qubits=7
classical_bits=7
depth=4
operations={'measure': 7, 'barrier': 3, 'state_preparation': 1,
            'PauliEvolution': 1, 'Observable\n(ZZZZZZZ)': 1}
time=0.001
```

本机结果：[Figure A1 简化线路](D:\学习资料\实习\酉术-ai算法实习生\文件\.codex_cache\paper_figure2_exact\paper_figure2_reproduced.png)。

## 9. 冒烟测试和 shots 收敛

在启动 2 小时精确实验前，必须先跑 7-grid 冒烟测试。已验证一次运行：288 circuits、3 jobs、1000 shots，Aer 约 10.9 秒。

已验证误差：

```text
ODE vs matrix exponential: u=3.0786454e-05, v=4.8820135e-05
local Aer vs matrix exponential: u=0.03618076, v=0.020212226
```

shots 收敛：

```text
100 shots:   u=0.15426746, v=0.07386616
1000 shots:  u=0.04124716, v=0.02059964
10000 shots: u=0.00970627, v=0.00614706
```

误差随 shots 增加下降，趋势约为 `1/sqrt(shots)`。若误差反而持续增大，先检查 seed、数据顺序、边界补零和归一化，而不是直接增加 shots。

## 10. 相对 L2 误差定义

每个时间点：

```python
error = np.linalg.norm(u_test - u_reference) / max(
    np.linalg.norm(u_reference), 1e-30
)
```

全时间平均是对每个时间点的相对误差取平均，而不是先把整个时空数组展平再算一次。`t=0` 两者相同，误差为 0。

## 11. 常见故障和唯一正确处理

### Aer 完成后 Python 无 traceback 退出

原因：18 个 CVXPY tomography 并发。应用 `max_workers=1` 补丁后重跑。原始 measurement result 没有 checkpoint，因此退出后通常必须重做对应 Aer batch。

### 日志长时间显示 `Jobs completed: 0 | 1`

只要 CPU 时间持续增长、RSS 稳定，就不是卡死。精确单批在本机约 21–23 分钟。

### `ModuleNotFoundError: simulation`

设置：

```bash
export PYTHONPATH=~/reproduction/Quantum-Wave-Equation-Solver-f64503a
```

### 找不到 `examples/.../configs.json`

脚本被放错层级。按本方案把脚本放到仓库的 `data/<run-name>/`，或把脚本内 `REPO` 改成明确绝对路径。

### `Times New Roman not found`

只影响字体，不影响数据。不要用 sudo 安装字体。Matplotlib 回退到 DejaVu Sans 即可。

### OOM 风险

单线程 tomography RSS 约 12–13 GiB，Spark 121 GiB 足够。若 RSS 持续接近总内存，停止新任务，但不要杀死无关进程。确认 PID 后只终止本次实验 PID。

## 12. 最终交付清单

```text
代码提交号
Python/Qiskit/Aer/Numpy/CVXPY 版本
tomography.py 单线程补丁 diff
Figure 1 configs.json、data_0/1/2.pkl、完整日志、PNG、摘要
Figure 2 作者归档 data_0/1/2.pkl、PNG、摘要
Figure A1 PNG、线路参数摘要
硬件信息和 Aer available_devices 输出
任何 warning 和失败尝试的日志
```

不得把 SSH 密码、IBM token 或公司密钥加入交付包。

## 13. 已验证产物总目录

```text
.codex_cache/paper_figure1_exact/
.codex_cache/paper_true_figure2/
.codex_cache/paper_figure2_exact/   # 实际对应论文 Figure A1
.codex_tools/reproduce_paper_figure1.py
.codex_tools/reproduce_paper_true_figure2.py
.codex_tools/reproduce_paper_figure2.py
```

执行者应先完成第 3–5 节，再按 Figure 1、Figure 2、Figure A1 顺序执行。任何一步的实际参数、文件名或误差定义与本文不一致，都应视为尚未完成复现。
