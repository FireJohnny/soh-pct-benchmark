#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''SOH 同台竞技排行榜图(4 子图)'''
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

fp = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
if os.path.exists(fp):
    font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 相对路径:code/ 的上一级是包根,数据在 ../data
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(_ROOT, "data")

IMG = _os.path.join(_ROOT, "figures")
_os.makedirs(IMG, exist_ok=True)
d = json.load(open(f"{OUT}/benchmark_results.json"))

names = [k for k in d if not k.startswith("M7")]
short = {"M1 PCT+SVR": "PCT+SVR", "M2 PCT+随机森林": "PCT+随机森林",
         "M3 弛豫14点+随机森林": "弛豫14点+RF", "M4 弛豫14点+高斯过程": "弛豫14点+GPR",
         "M5 端电压统计+线性回归": "端电压+线性", "M6 全特征+梯度提升": "全特征+GBDT"}
labs = [short.get(n, n) for n in names]
rand = [d[n]["random"]["RMSE_pct"] for n in names]
lobo = [d[n]["lobo"]["RMSE_pct"] for n in names]
r2 = [d[n]["lobo"]["R2"] for n in names]
nfeat = [d[n]["n_features"] for n in names]
mean_l = d["M7 猜均值"]["lobo"]["RMSE_pct"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ① 跨电池 RMSE 排行
ax = axes[0, 0]
order = np.argsort(lobo)
cols = ["#2f855a" if lobo[i] < 2 else ("#d69e2e" if lobo[i] < 4 else "#c53030") for i in order]
bars = ax.barh([labs[i] for i in order], [lobo[i] for i in order], color=cols)
ax.axvline(mean_l, ls="--", c="#718096", lw=1.2, label=f"猜均值基准 {mean_l:.2f}%")
ax.set_xlabel("跨电池 RMSE (%)  越低越好")
ax.set_title("SOH 方法排行(严格跨电池,同一数据)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")
for b, i in zip(bars, order):
    ax.text(b.get_width() + .08, b.get_y() + b.get_height() / 2, f"{lobo[i]:.2f}%", va="center", fontsize=9)

# ② 随机 vs 跨电池(划分陷阱)
ax = axes[0, 1]
x = np.arange(len(names)); w = .38
ax.bar(x - w / 2, rand, w, label="随机划分", color="#90cdf4")
ax.bar(x + w / 2, lobo, w, label="严格跨电池", color="#2b6cb0")
ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("RMSE (%)"); ax.set_title("随机划分 vs 跨电池:谁在\"裸泳\"?", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
for i, (a, b) in enumerate(zip(rand, lobo)):
    if b / a > 1.8:
        ax.annotate(f"×{b/a:.1f}", (i + w / 2, b), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color="#c53030", fontweight="bold")

# ③ 跨电池 R²
ax = axes[1, 0]
order2 = np.argsort(r2)[::-1]
cols2 = ["#2f855a" if r2[i] > .7 else ("#d69e2e" if r2[i] > .3 else "#c53030") for i in order2]
ax.barh([labs[i] for i in order2], [r2[i] for i in order2], color=cols2)
ax.axvline(0, c="k", lw=.8)
ax.set_xlabel("跨电池 R²")
ax.set_title("跨电池泛化能力(R²,越高越好)", fontweight="bold")
ax.grid(alpha=.3, axis="x")
for i, v in zip(order2, [r2[i] for i in order2]):
    ax.text(v + .01, list(order2).index(i), f"{v:.3f}", va="center", fontsize=9)

# ④ 特征数 vs 跨电池精度
ax = axes[1, 1]
ax.scatter(nfeat, lobo, s=180, c="#2b6cb0", zorder=3)
for n, x_, y_ in zip(labs, nfeat, lobo):
    ax.annotate(n, (x_, y_), textcoords="offset points", xytext=(7, 5), fontsize=9)
ax.axhline(mean_l, ls="--", c="#718096", lw=1, label="猜均值")
ax.set_xlabel("特征数量"); ax.set_ylabel("跨电池 RMSE (%)")
ax.set_title("特征数量 vs 跨电池精度", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
out = f"{IMG}/soh_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("已保存", out, flush=True)
