#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
SOH 类论文"同台竞技" — 主脚本
统一数据(NASA 4 电芯)、统一划分、统一指标,让各论文的方法同场比拼
'''
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# 相对路径:code/ 的上一级是包根,数据在 ../data
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(_ROOT, "data")

NOM = 2.0
df = pd.read_csv(f"{OUT}/features.csv")
PCT = ["pct1", "pct2", "pct3", "pct4"]
RV = [f"rv{k}" for k in range(14)]
VSTAT = ["v_mean", "v_std", "v_max", "v_min", "t_total", "t_mean", "i_mean",
         "temp_mean", "temp_std", "temp_max"]

# 参赛方法(名称 → 特征集 + 模型)
METHODS = {
    "M1 PCT+SVR": (PCT, lambda: SVR(C=10.0, gamma="scale")),
    "M2 PCT+随机森林": (PCT, lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=2)),
    "M3 弛豫14点+随机森林": (RV, lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=2)),
    "M4 弛豫14点+高斯过程": (RV, lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.01), normalize_y=True,
        optimizer=None, random_state=42)),   # 固定核:防超参过度优化→恒定预测
    "M5 端电压统计+线性回归": (VSTAT, lambda: LinearRegression()),
    "M6 全特征+梯度提升": (PCT + RV + VSTAT, lambda: GradientBoostingRegressor(random_state=42)),
}

X_all = df[PCT + RV + VSTAT].values
y = df["soh"].values
cells = df["cell"].values
print(f"样本 {len(df)} | 电芯 {df.cell.nunique()} | SOH {y.min():.3f}-{y.max():.3f}", flush=True)


def fp(make_model, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = make_model().fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def metrics(yt, yp):
    return {
        "RMSE_pct": float(np.sqrt(np.mean((yt - yp) ** 2)) / NOM * 100),
        "MAE_pct": float(np.mean(np.abs(yt - yp)) / NOM * 100),
        "MAPE_pct": float(np.mean(np.abs((yt - yp) / yt)) * 100),
        "R2": float(r2_score(yt, yp)),
    }


res = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 92, flush=True)
print("SOH 同台竞技(统一数据:NASA B0005/6/7/18;统一划分与指标)", flush=True)
print("=" * 92, flush=True)
print(f"{'方法':<26} {'随机RMSE':>9} {'随机MAE':>8} {'跨电池RMSE':>10} {'跨电池MAE':>9} {'跨R²':>7}", flush=True)
print("-" * 92, flush=True)

idx_map = {c: np.where(np.isin(df.columns, PCT + RV + VSTAT))[0] for c in []}
for name, (feats, mk) in METHODS.items():
    cols = [df.columns.get_loc(f) for f in feats]
    X = df.iloc[:, cols].values
    # 随机 5 折
    yt, yp = [], []
    for tr, te in kf.split(X):
        yp += list(fp(mk, X[tr], y[tr], X[te])); yt += list(y[te])
    r_rand = metrics(np.array(yt), np.array(yp))
    # 严格跨电池
    yt, yp = [], []
    for c in sorted(set(cells)):
        te = cells == c; tr = ~te
        yp += list(fp(mk, X[tr], y[tr], X[te])); yt += list(y[te])
    r_lobo = metrics(np.array(yt), np.array(yp))
    res[name] = {"random": r_rand, "lobo": r_lobo, "n_features": len(feats)}
    print(f"{name:<26} {r_rand['RMSE_pct']:>8.3f}% {r_rand['MAE_pct']:>7.3f}% "
          f"{r_lobo['RMSE_pct']:>9.3f}% {r_lobo['MAE_pct']:>8.3f}% {r_lobo['R2']:>7.4f}", flush=True)

# 猜均值基准
errs_r, errs_l = [], []
for tr, te in kf.split(X_all):
    errs_r.append(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)) / NOM * 100)
for c in sorted(set(cells)):
    te = cells == c
    errs_l.append(np.sqrt(np.mean((y[te] - y[~te].mean()) ** 2)) / NOM * 100)
print("-" * 92, flush=True)
print(f"{'M7 猜均值(基准)':<26} {np.mean(errs_r):>8.3f}% {'-':>8} {np.mean(errs_l):>9.3f}% {'-':>9} {'-':>7}", flush=True)
res["M7 猜均值"] = {"random": {"RMSE_pct": float(np.mean(errs_r))}, "lobo": {"RMSE_pct": float(np.mean(errs_l))}}

json.dump(res, open(f"{OUT}/benchmark_results.json", "w"), ensure_ascii=False, indent=2)
print(f"\n保存 → {OUT}/benchmark_results.json", flush=True)
