#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
SOH 预测曲线对比:真实 SOH vs 预测 SOH(严格跨电池)
用两个代表方法:M2(PCT+随机森林,性价比冠军)与 M4(弛豫+高斯过程,失败案例)
输出:4 个电池的预测曲线 + 拟合指标
'''
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

fp = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
if os.path.exists(fp):
    font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 相对路径:code/ 的上一级是包根,数据在 ../data
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
BENCH = _os.path.join(_ROOT, "data")

IMG = _os.path.join(_ROOT, "figures")
_os.makedirs(IMG, exist_ok=True)
NOM = 2.0
df = pd.read_csv(f"{BENCH}/features.csv")
PCT = ["pct1", "pct2", "pct3", "pct4"]
RV = [f"rv{k}" for k in range(14)]
VSTAT = ["v_mean", "v_std", "v_max", "v_min", "t_total", "t_mean", "i_mean",
         "temp_mean", "temp_std", "temp_max"]

METHODS = {
    "M2 PCT+随机森林": (PCT, lambda: RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=2)),
    "M6 全特征+梯度提升": (PCT + RV + VSTAT, lambda: GradientBoostingRegressor(random_state=42)),
    "M4 弛豫+GPR": (RV, None),   # 单独处理
}
cells = sorted(df.cell.unique())
print(f"电池 {cells} | 样本 {len(df)}", flush=True)


def run(feats, mk):
    X = df[feats].values; y = df["soh"].values; cs = df["cell"].values
    pred = np.full(len(y), np.nan)
    for c in cells:
        te = cs == c; tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = mk().fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
METHODS["M4 弛豫+GPR"] = (RV, lambda: GaussianProcessRegressor(
    kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01), normalize_y=True,
    optimizer=None, random_state=42))

preds = {}
for name, (feats, mk) in METHODS.items():
    t0 = pd.Timestamp.now()
    preds[name] = run(feats, mk)
    m = np.isfinite(preds[name])
    r = float(np.sqrt(np.mean((df.soh.values[m] - preds[name][m]) ** 2)) / NOM * 100)
    r2 = r2_score(df.soh.values[m], preds[name][m])
    print(f"  {name}: RMSE {r:.3f}% | R² {r2:.4f} | MAE {mean_absolute_error(df.soh.values[m], preds[name][m])/NOM*100:.3f}%", flush=True)

# 画图:4 电池 × 3 方法
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
colors = {"M2 PCT+随机森林": "#2b6cb0", "M6 全特征+梯度提升": "#2f855a", "M4 弛豫+GPR": "#c53030"}
for i, c in enumerate(cells):
    ax = axes[i]
    m = df.cell.values == c
    x = df.cycle.values[m]
    yt = df.soh.values[m] * 100
    ax.plot(x, yt, "o-", color="#1a1a1a", ms=3, lw=1.2, label="真实 SOH", zorder=5)
    for name in METHODS:
        yp = preds[name][m] * 100
        ax.plot(x, yp, "--", color=colors[name], lw=1.5, label=f"{name}")
    ax.set_title(f"{c}(测试电池)", fontweight="bold")
    ax.set_xlabel("循环数"); ax.set_ylabel("SOH (%)")
    ax.legend(fontsize=7.5); ax.grid(alpha=.3)
plt.suptitle("SOH 预测曲线 vs 真实曲线(严格跨电池:每个电池都是没见过的)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = f"{IMG}/soh_curve_compare.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n已保存 {out}", flush=True)

# 各电池分别的拟合指标
print(f"\n{'电池':>7} {'真实SOH范围':>14} " + " ".join(f"{n[:12]:>14}" for n in METHODS), flush=True)
for c in cells:
    m = df.cell.values == c
    yt = df.soh.values[m]
    line = f"{c:>7} {yt.min()*100:>6.1f}-{yt.max()*100:<6.1f} "
    for n in METHODS:
        e = np.sqrt(np.mean((yt - preds[n][m]) ** 2)) / NOM * 100
        line += f"{e:>13.2f}%"
    print(line, flush=True)
