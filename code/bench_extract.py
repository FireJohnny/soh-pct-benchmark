#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
SOH 类论文"同台竞技"复现 — 赛道设计
统一数据:NASA PCoE B0005/B0006/B0007/B0018(4 电芯,本地已有,多篇论文用过 → 可比性最好)
统一划分:① 随机 5 折 ② 严格跨电池(留一电芯)
统一指标:RMSE / MAE / MAPE(%)

参赛方法(各论文核心思路,全部在本地实现):
  M1 部分充电时间(PCT)+ SVR          ← 10.3390/en16114423
  M2 部分充电时间(PCT)+ 随机森林      ← 10.3390/en16114423
  M3 弛豫/静置电压 14 点 + 随机森林    ← 10.1038/s41467-022-29837-w
  M4 增量容量(IC)特征 + 高斯过程回归  ← arxiv_2603.26155
  M5 端电压统计特征 + 线性回归(基线)   ← 10.3390/batteries12090342 的简单端
  M6 猜均值(朴素基准)
'''
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

PD_ = "/root/battery_datasets/nasa/parquet"
# 相对路径:code/ 的上一级是包根,数据在 ../data
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(_ROOT, "data")

os.makedirs(OUT, exist_ok=True)
CELLS = ["B0005", "B0006", "B0007", "B0018"]
NOM = 2.0
THR = [3.7, 3.8, 3.9, 4.0, 4.1]        # PCT 电压区间
print("=== 1. 提取统一特征表 ===", flush=True)


def cross_time(t, v, thr):
    idx = np.where((v[:-1] < thr) & (v[1:] >= thr))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return t[i] if v[i + 1] == v[i] else t[i] + (thr - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])


rows = []
for cell in CELLS:
    df = pd.read_parquet(f"{PD_}/{cell}.parquet")
    chg, dis = {}, {}
    for cyc, g in df.groupby("cycle", sort=True):
        typ = g["type"].iloc[0]
        if typ == "charge" and len(g) >= 20:
            chg[int(cyc)] = g
        elif typ == "discharge" and len(g) >= 5:
            dis[int(cyc)] = float(g["capacity"].max())
    for c in sorted(chg):
        d = c + 1
        if d not in dis:
            continue
        cap = dis[d]
        if not np.isfinite(cap) or cap <= 0.1:
            continue
        g = chg[c]
        v = g["voltage"].values.astype(float)
        t = g["time"].values.astype(float)
        tem = g["temp"].values.astype(float)
        cur = g["current"].values.astype(float)
        r = {"cell": cell, "cycle": d, "capacity": cap, "soh": cap / NOM}
        # PCT 特征(A1-A4):电压区间充电时间
        tc = {thr: cross_time(t, v, thr) for thr in THR}
        for k in range(4):
            a, b = THR[k], THR[k + 1]
            r[f"pct{k+1}"] = (tc[b] - tc[a]) if np.isfinite(tc[a]) and np.isfinite(tc[b]) else np.nan
        # 弛豫/静置电压 14 点:充电末端(电压最高处)取等间隔 14 点
        idx14 = np.linspace(0, len(v) - 1, 14).astype(int)
        for k in range(14):
            r[f"rv{k}"] = float(v[idx14[k]])
        # IC 特征(增量容量):dQ/dV 的统计量
        dq = np.gradient(cap * np.ones_like(v) * 0, v)  # 占位,后续用放电曲线算
        # 端电压统计
        r["v_mean"] = float(np.mean(v)); r["v_std"] = float(np.std(v))
        r["v_max"] = float(np.max(v)); r["v_min"] = float(np.min(v))
        r["t_total"] = float(t[-1] - t[0])
        r["t_mean"] = float(np.mean(t)); r["i_mean"] = float(np.mean(np.abs(cur)))
        r["temp_mean"] = float(np.mean(tem)); r["temp_std"] = float(np.std(tem))
        r["temp_max"] = float(np.max(tem))
        rows.append(r)
    print(f"  {cell}: {len([x for x in rows if x['cell'] == cell])} 循环", flush=True)

data = pd.DataFrame(rows).dropna(subset=["pct1", "pct2", "pct3", "pct4"])
data.to_csv(f"{OUT}/features.csv", index=False)
print(f"\n合计 {len(data)} 样本 / {data.cell.nunique()} 电芯 → {OUT}/features.csv", flush=True)
print(f"SOH 范围 {data.soh.min():.3f}-{data.soh.max():.3f}", flush=True)
