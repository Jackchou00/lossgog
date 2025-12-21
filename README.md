# Loss GOG

基于感知色差损失函数的显示器色彩特性化模型研究。

本项目实现了一种显示器色彩特性化方法：**GOG 模型**，并研究了不同损失函数（XYZ MSE、CIEDE2000、sUCS）对模型精度的影响。

## 项目结构

```
lossgog/
├── src/
│   └── lossgog/                # 核心模块
│       ├── __init__.py
│       ├── data_io.py          # 数据读写
│       ├── plotting.py         # 可视化工具
│       ├── gog/                # GOG 模型
│       │   ├── __init__.py
│       │   ├── model.py        # 模型构建与转换
│       │   └── evaluation.py   # 模型评估
│       ├── lut/                # 3D LUT
│       │   ├── __init__.py
│       │   ├── model.py        # LUT 构建
│       │   └── evaluation.py   # LUT 评估
│       ├── ucs/                # 色差计算
│       │   ├── __init__.py
│       │   ├── de2000.py       # CIEDE2000 色差
│       │   └── sucs.py         # sUCS 色差
│       └── exp/                # 实验模块
│           └── sp_gog.py       
│
├── scripts/                    # 可执行脚本
│   ├── train_gog.py            # 训练 GOG 模型
│   ├── train_sp_gog.py         # 训练稀疏 GOG
│   ├── xgimi_train.py          # XGIMI 投影仪训练
│   ├── build_lut.py            # 构建 3D LUT
│   ├── train_size_experiment.py # 训练样本量实验
│   ├── compare_gog_formulas.py # GOG 公式对比实验
│   ├── gog_roundtrip_test.py   # GOG 往返测试
│   ├── apply_gog.py            # 应用 GOG 到图像
│   ├── apply_lut.py            # 应用 LUT 到图像
│   └── plot_gamut.py           # 绘制色域图
│
├── pyproject.toml              # 项目配置
└── README.md                   # 本文档
```

## 安装

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 克隆项目
git clone <repo-url>
cd lossgog

# 创建虚拟环境并安装依赖
uv sync
```

## 快速开始

```bash
# 训练 GOG 模型
uv run scripts/train_gog.py

# 构建 3D LUT
uv run scripts/build_lut.py

# 运行训练样本量实验
uv run scripts/train_size_experiment.py
```

---

## 核心模块详解

### 1. GOG 模型 (`gog/model.py`)

GOG (Gain-Offset-Gamma) 模型是一种参数化的显示器色彩特性化模型，共 18 个参数。

#### 数学公式

**前向转换 (RGB → XYZ)：**

$$
L_c = (\text{gain}_c \cdot \text{RGB}_c + \text{offset}_c)^{\gamma_c}, \quad c \in \{R, G, B\}
$$

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \mathbf{M} \cdot \begin{bmatrix} L_R \\ L_G \\ L_B \end{bmatrix}
$$

**逆向转换 (XYZ → RGB)：**

$$
\begin{bmatrix} L_R \\ L_G \\ L_B \end{bmatrix} = \mathbf{M}^{-1} \cdot \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

$$
\text{RGB}_c = \frac{L_c^{1/\gamma_c} - \text{offset}_c}{\text{gain}_c}
$$

#### 参数物理意义

| 参数       | 含义                                      |
| ---------- | ----------------------------------------- |
| **gain**   | 输入信号线性放大系数（在非线性变换之前）  |
| **offset** | 黑电平偏移（黑场补偿）                    |
| **gamma**  | 非线性幂指数（电光传递函数）              |
| **matrix** | 3×3 色彩转换矩阵（RGB 基色到 XYZ 的映射） |

> **注：** 本项目使用的是标准 GOG 公式 `L = (gain·RGB + offset)^γ`，其中 gain 在幂函数内部。
> 另一种形式 `L = gain·(RGB + offset)^γ` 中 gain 在外部，会与矩阵 M 产生冗余。
> 详见 `scripts/compare_gog_formulas.py` 的对比实验。

#### 两种 GOG 构建方法

本项目提供两种 GOG 模型构建方法：

##### 1. 经典 GOG (`classic_gog`)

从单通道原色 ramp 测量数据逐通道拟合，适用于有 R/G/B 单独灰阶测量的场景。

**原理：**
1. 对每个通道 c ∈ {R, G, B}，使用纯色 ramp 数据拟合 tone response：
   $$L_c = (\text{gain}_c \cdot \text{input} + \text{offset}_c)^{\gamma_c}$$
   其中 $L_c$ 归一化到 [0, 1]（基于亮度 Y 减去黑点后归一化）
2. 矩阵 M 的列直接取各原色最大亮度时的 XYZ（减去黑点贡献）

**优点：** 直观、物理意义明确、计算快速  
**缺点：** 无法处理通道间串扰，对投影仪等复杂设备精度有限

##### 2. 优化 GOG (`make_gog`)

使用所有测量数据联合优化 18 个参数，支持多种损失函数。

**原理：**
1. 初始化参数（gain=1, offset=0.001, gamma=2.2, matrix=缩放单位阵）
2. 使用 L-BFGS-B 优化器最小化预测 XYZ 与测量 XYZ 的误差
3. 支持三种损失函数：XYZ MSE、CIEDE2000、sUCS

**优点：** 精度高、能补偿通道间串扰  
**缺点：** 需要更多测量数据、计算较慢

#### 关键函数

##### `classic_gog(rgb, xyz, ramp_indices=None, verbose=True)`

从单通道原色 ramp 数据构建经典 GOG 模型。

**参数：**
- `rgb`: `(N, 3)` 数组，RGB 值，范围 [0, 1]
- `xyz`: `(N, 3)` 数组，对应的 XYZ 三刺激值
- `ramp_indices`: 可选，指定每个通道的数据索引范围
  - 如 `[(0, 18), (18, 36), (36, 54)]` 表示 R/G/B 各 18 个样本
  - 若为 None，自动检测单通道数据
- `verbose`: 是否打印拟合过程

**返回：**
```python
{
    "gain": np.array([g_R, g_G, g_B]),
    "offset": np.array([o_R, o_G, o_B]),
    "gamma": np.array([γ_R, γ_G, γ_B]),
    "matrix": np.array([[...], [...], [...]])  # 列为 R/G/B 原色 XYZ
}
```

**示例：**
```python
from lossgog.gog import classic_gog, rgb_to_xyz_gog

# 从原色 ramp 构建经典 GOG
ramp_indices = [(0, 18), (18, 36), (36, 54)]  # R/G/B 各 18 点
gog_model = classic_gog(rgb_train, xyz_train, ramp_indices=ramp_indices)

# 使用模型
xyz_pred = rgb_to_xyz_gog(rgb_test, gog_model)
```

##### `make_gog(rgb, xyz, mode="xyz", verbose=True)`

从 RGB-XYZ 测量数据拟合 GOG 模型参数。

**参数：**
- `rgb`: `(N, 3)` 数组，RGB 值，范围 [0, 1]
- `xyz`: `(N, 3)` 数组，对应的 XYZ 三刺激值
- `mode`: 损失函数类型
  - `"xyz"`: XYZ 空间均方误差（最快）
  - `"de2000"`: CIEDE2000 色差（感知均匀）
  - `"sucs"`: sUCS 色差（计算快于 DE2000）
- `verbose`: 是否打印优化过程

**返回：**
```python
{
    "gain": np.array([g_R, g_G, g_B]),      # 增益
    "offset": np.array([o_R, o_G, o_B]),   # 偏移
    "gamma": np.array([γ_R, γ_G, γ_B]),    # 伽马
    "matrix": np.array([[...], [...], [...]]) # 3×3 矩阵
}
```

**示例：**
```python
from lossgog.gog import make_gog, rgb_to_xyz_gog

# 训练模型
gog_model = make_gog(rgb_train, xyz_train, mode="de2000")

# 使用模型
xyz_pred = rgb_to_xyz_gog(rgb_test, gog_model)
```

##### `rgb_to_xyz_gog(rgb, gog_model)`

使用 GOG 模型将 RGB 转换为 XYZ。

##### `xyz_to_rgb_gog(xyz, gog_model)`

使用 GOG 模型的逆变换将 XYZ 转换为 RGB。

---

### 2. 3D LUT (`lut/model.py`)

3D LUT 是一种基于查表和插值的色彩转换方法，适用于任意非线性映射。

#### Mid Space（中间色彩空间）

由于测量数据在 XYZ 空间中分布不均匀，直接插值效果不佳。本项目引入 **Mid Space** 作为中间空间：

$$
\text{Mid Space} = \text{EOTF}^{-1}_{\text{BT.1886}} \left( \text{XYZ\_to\_RGB}_{\text{BT.2020}} \left( \frac{\text{XYZ}}{Y_{\text{white}}} \right) \right)
$$

**设计原理：**
1. **白点归一化**：除以白点亮度 $Y_{\text{white}}$，消除绝对亮度的影响
2. **BT.2020 RGB**：宽色域空间，减少色域裁剪
3. **逆 EOTF**：类似伽马编码，使数值分布更均匀

#### 关键函数

##### `build_forward_lut(source_rgb, target_xyz, source_grid_size, output_size=17)`

从规则网格测量数据构建前向 LUT (RGB → XYZ)。

**参数：**
- `source_rgb`: `(N, 3)` RGB 测量点，必须形成规则网格
- `target_xyz`: `(N, 3)` 对应的 XYZ 值
- `source_grid_size`: 源数据每通道采样数（如 15 表示 15³=3375 点）
- `output_size`: 输出 LUT 尺寸（如 17, 33, 65）

**返回：**
- `(output_size, output_size, output_size, 3)` 数组

**示例：**
```python
from lossgog.lut import build_forward_lut
import colour

# 从 15³ 测量点构建 17³ LUT
lut_data = build_forward_lut(rgb, xyz, source_grid_size=15, output_size=17)
lut = colour.LUT3D(table=lut_data, size=17)

# 保存为 .cube 文件
colour.write_LUT(lut, "output.cube")
```

##### `build_inverse_lut(measured_xyz, measured_rgb, white_point_luminance, output_size=17, method="linear")`

从散点测量数据构建逆向 LUT (Mid Space → RGB)。

**参数：**
- `measured_xyz`: `(N, 3)` XYZ 测量值
- `measured_rgb`: `(N, 3)` 对应的 RGB 值
- `white_point_luminance`: 白点亮度（Y 值）
- `output_size`: 输出 LUT 尺寸
- `method`: 插值方法
  - `"linear"`: 线性插值（快速，可能有空洞）
  - `"nearest"`: 最近邻（最快）
  - `"rbf"`: 径向基函数（平滑，较慢）

**使用逆向 LUT：**
```python
from lossgog.lut import build_inverse_lut, xyz_to_mid_space

# 构建 LUT
lut_data = build_inverse_lut(xyz, rgb, white_Y, method="rbf")
inverse_lut = colour.LUT3D(table=lut_data)

# 使用 LUT：XYZ → Mid Space → RGB
mid = xyz_to_mid_space(target_xyz, white_Y)
display_rgb = inverse_lut.apply(mid)
```

##### `xyz_to_mid_space(xyz, white_point_luminance)`

将 XYZ 转换为 Mid Space。

##### `mid_space_to_xyz(mid, white_point_luminance)`

将 Mid Space 转换回 XYZ。

---

### 3. 色差计算 (`ucs/`)

#### `calculate_de2000(xyz_pred, xyz_target)`

计算 CIEDE2000 色差。

**参数：**
- `xyz_pred`, `xyz_target`: `(N, 3)` XYZ 值，范围 [0, 1]

**返回：**
- `(N,)` 色差值数组

#### `calculate_de_sucs(xyz_pred, xyz_target)`

计算 sUCS (Simplified Uniform Color Space) 色差，比 CIEDE2000 计算更快。

---

### 4. 数据读写 (`data_io.py`)

#### `read_cs2000_csv(file_path, spectral_length=401)`

读取 CS2000 色度计的测量数据 CSV 文件。

**CSV 格式：**
```
R, G, B, X, Y, Z, 380nm, 381nm, ..., 780nm
```

**返回：**
```python
{
    "RGB": np.array(..., dtype=np.uint8),    # (N, 3)
    "XYZ": np.array(..., dtype=np.float32),  # (N, 3)
    "spectral": np.array(..., dtype=np.float32)  # (N, 401)
}
```

---

## 实验脚本

### 训练样本量实验 (`train_size_experiment.py`)

研究训练样本数量对模型精度的影响：

- 从 10 到 1000 点对数间隔采样
- 对比 XYZ MSE、CIEDE2000、sUCS 三种损失函数
- 每个样本量重复 3 次取平均
- 输出平均色差随样本量变化的曲线

### GOG 公式对比实验 (`compare_gog_formulas.py`)

对比两种 GOG 公式的拟合效果：

| 公式         | 形式                        | 说明                        |
| ------------ | --------------------------- | --------------------------- |
| **标准形式** | `L = (gain·RGB + offset)^γ` | gain 在幂函数内部，参数独立 |
| **旧形式**   | `L = gain·(RGB + offset)^γ` | gain 在外部，与矩阵 M 冗余  |

**实验结论：**

在 4 个数据集 × 3 种损失函数 = 12 组实验中：
- 两种公式拟合精度几乎相同（差异 < 0.001 ΔE）
- 原因：offset 值很小（≈0.001），此时两公式数学上等价
- **推荐使用标准形式**：参数物理意义更清晰，与显示器信号链模型一致

---

## 许可证

MIT License

---

## 作者

Miaosen Zhou
