# 反粒子识别指南

## 🎯 问题背景

之前的代码无法识别反粒子，导致像 `e+e->mu+mu` 这样的命令被错误解析为：
- 初态：$e^- + e^-$ （两个电子）
- 末态：$\mu^- + \mu^-$ （两个缪子）

这在物理上是**不可能的**，因为：
1. 违反电荷守恒（总电荷 = -4）
2. 违反轻子味数守恒
3. 标准模型中没有可以实现此反应的相互作用

## ✅ 解决方案

现在代码支持使用 `_bar` 后缀来表示反粒子！

## 📝 使用方法

### 基本语法

```bash
python train.py --reaction "particle1+particle2->particle3+particle4"
```

### 反粒子表示

使用 `_bar` 后缀表示反粒子：

| 粒子符号 | 物理含义 | 电荷 |
|---------|---------|-----|
| `e` | 电子 ($e^-$) | -1 |
| `e_bar` | 正电子 ($e^+$) | +1 |
| `mu` | 缪子 ($\mu^-$) | -1 |
| `mu_bar` | 反缪子 ($\mu^+$) | +1 |
| `u` | 上夸克 ($u$) | +2/3 |
| `u_bar` | 反上夸克 ($\bar{u}$) | -2/3 |
| `d` | 下夸克 ($d$) | -1/3 |
| `d_bar` | 反下夸克 ($\bar{d}$) | +1/3 |

## 🧪 常见反应示例

### 1. 正负电子湮灭产生缪子对（树图级别最简单）

```bash
python train.py --reaction "e+e_bar->mu+mu_bar"
```

物理过程：
- 初态：$e^- + e^+$ （电荷 = 0）
- 通过光子 $\gamma$ 或 $Z$ 玻色子
- 末态：$\mu^- + \mu^+$ （电荷 = 0）

这是**物理上可行**的标准模型过程！

### 2. 正负电子湮灭产生光子

```bash
python train.py --reaction "e+e_bar->photon+photon"
```

### 3. 夸克反应（如果支持）

```bash
python train.py --reaction "u+u_bar->d+d_bar"
```

## 🔧 技术细节

### 代码修改点

1. **`feynman_env.py` 的 `reset()` 方法**：
   - 添加了 `parse_particle_id()` 辅助函数
   - 自动识别 `_bar` 后缀并设置 `is_anti=True`

2. **`train.py` 的验证逻辑**：
   - 允许带 `_bar` 后缀的粒子通过验证
   - 提供友好的错误提示

### 示例解析

输入：`"e+e_bar->mu+mu_bar"`

解析结果：
```python
initial_state = ['e', 'e_bar']
final_state = ['mu', 'mu_bar']

# 在环境中会被解析为：
# e: particle_id='e', is_anti=False  (电子, 电荷=-1)
# e_bar: particle_id='e', is_anti=True  (正电子, 电荷=+1)
# mu: particle_id='mu', is_anti=False  (缪子, 电荷=-1)
# mu_bar: particle_id='mu', is_anti=True  (反缪子, 电荷=+1)
```

## ⚠️ 常见错误

### 错误 1：忘记使用 `_bar`

```bash
# ❌ 错误
python train.py --reaction "e+e->mu+mu"
# 这会被解析为 e^- + e^- -> mu^- + mu^-，物理上不可能！
```

```bash
# ✅ 正确
python train.py --reaction "e+e_bar->mu+mu_bar"
```

### 错误 2：使用 `+` 号表示正电荷

```bash
# ❌ 错误（会导致解析混乱）
python train.py --reaction "e+->mu+"

# ✅ 正确
python train.py --reaction "e_bar->mu_bar"
```

## 🎓 物理守恒定律提醒

训练时模型会检查以下守恒定律：

1. **电荷守恒**：$\sum Q_{in} = \sum Q_{out}$
2. **轻子数守恒**：$\sum L_{in} = \sum L_{out}$
3. **重子数守恒**：$\sum B_{in} = \sum B_{out}$
4. **颜色守恒**（对于夸克）

确保您的反应在物理上是允许的！

## 📊 验证反应是否可行

运行训练后观察：
- ✅ 如果模型能够获得正奖励 → 反应可行
- ❌ 如果奖励一直是 0 或负数 → 检查守恒定律

## 🚀 完整训练示例

```bash
# 1. 激活 Python 环境（如果需要）
# conda activate your_env

# 2. 运行训练（正负电子湮灭）
python train.py --reaction "e+e_bar->mu+mu_bar" --timesteps 100000

# 3. 在另一个终端监控性能
python monitor_performance.py

# 4. 打开可视化（在浏览器中）
# 打开 training_viz.html
```

## 💡 提示

- 使用 `--help` 查看所有可用参数
- 检查 `PhysicsConstants` 类了解所有可用粒子
- 树图级别的简单反应更容易训练成功
- QED 反应（只涉及轻子和光子）通常最容易

祝训练成功！🎉
