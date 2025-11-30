# Loss-GOG

基于感知色差损失函数的显示器色彩特性化模型研究。

本项目实现了两种显示器色彩特性化方法：**GOG 模型**和 **3D LUT**，并研究了不同损失函数（XYZ MSE、CIEDE2000、sUCS）对模型精度的影响。

## 项目结构

```
loss-gog/
├── src/                        # 核心模块
│   ├── gog/                    # GOG 模型
│   │   ├── model.py            # 模型构建与转换
│   │   └── evaluation.py       # 模型评估
│   ├── lut/                    # 3D LUT
│   │   ├── model.py            # LUT 构建
│   │   └── evaluation.py       # LUT 评估
│   ├── ucs/                    # 色差计算
│   │   ├── de2000.py           # CIEDE2000 色差
│   │   └── sucs.py             # sUCS 色差
│   ├── data_io.py              # 数据读写
│   └── plotting.py             # 可视化工具
│
├── scripts/                    # 可执行脚本
│   ├── train_gog.py            # 训练 GOG 模型
│   ├── build_lut.py            # 构建 3D LUT
│   ├── train_size_experiment.py # 训练样本量实验
│   ├── apply_gog.py            # 应用 GOG 到图像
│   ├── apply_lut.py            # 应用 LUT 到图像
│   └── plot_gamut.py           # 绘制色域图
│
├── measured_data/              # CS2000 测量数据
└── results/                    # 输出结果
```

## 安装

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 克隆项目
git clone <repo-url>
cd loss-gog

# 创建虚拟环境并安装依赖
uv venv
uv pip install -e .
```

## 快速开始

```bash
# 训练 GOG 模型
python scripts/train_gog.py

# 构建 3D LUT
python scripts/build_lut.py

# 运行训练样本量实验
python scripts/train_size_experiment.py
```

---

## 核心模块详解

### 1. GOG 模型 (`gog/model.py`)

GOG (Gain-Offset-Gamma) 模型是一种参数化的显示器色彩特性化模型，共 18 个参数。

#### 数学公式

**前向转换 (RGB → XYZ)：**

$$
L_c = \text{gain}_c \cdot (\text{RGB}_c + \text{offset}_c)^{\gamma_c}, \quad c \in \{R, G, B\}
$$

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \mathbf{M} \cdot \begin{bmatrix} L_R \\ L_G \\ L_B \end{bmatrix}
$$

**逆向转换 (XYZ → RGB)：**

$$
\begin{bmatrix} L_R \\ L_G \\ L_B \end{bmatrix} = \mathbf{M}^{-1} \cdot \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

$$
\text{RGB}_c = \left( \frac{L_c}{\text{gain}_c} \right)^{1/\gamma_c} - \text{offset}_c
$$

#### 关键函数

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
from gog import make_gog, rgb_to_xyz_gog

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
from lut import build_forward_lut
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
from lut import build_inverse_lut, xyz_to_mid_space

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

---

## 依赖

- Python ≥ 3.13
- numpy
- scipy
- pandas
- matplotlib
- colour-science

---

## 许可证

MIT License

---

## 作者

Jack Chou

## 参考文献

- IEC 61966-2-1: sRGB 标准
- ITU-R BT.2020: 超高清电视色域
- ITU-R BT.1886: 显示器 EOTF
- CIE 142-2001: CIEDE2000 色差公式
