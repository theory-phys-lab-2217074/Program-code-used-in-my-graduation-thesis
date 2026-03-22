import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

# --- 物理定数（SI 単位系）-------------------------------
c_SI  = 2.99792458e8
G_SI  = 6.67430e-11
M_sun = 1.98847e30
Rg_sun = 2 * G_SI * M_sun / c_SI**2   # 太陽質量のシュワルツシルト半径

# --- パラメータ（幾何化単位 G = c = 1）----------------
M_SI = 2.0 * M_sun          # 2 M⊙ ブラックホール
Rg_SI = 2 * G_SI * M_SI / c_SI**2     # 事象の地平面

# --- 無次元化 ------------------------------------------------
Rg = Rg_SI / Rg_sun         # → Rg = 2 になる
c  = 1.                    # 単位系で c = 1
v0 = 0.2                    # 初期 3‑速度の大きさ（0 < v0 < 1）

# --- インパクトパラメータ生成 & フィルタ -------------------
b_all  = np.linspace(10, 30, 20)     # [-30 M, …, +30 M]
b_list = b_all[np.abs(b_all) > Rg]    # |b| ≤ Rg (=2) を除外

# --- カラーマップ ------------------------------------------
colors = cm.plasma(np.linspace(0, 1, len(b_list)))

# ============================================================
# 1.  測地線方程式  d²r/dτ², dφ/dτ（τ = 固有時,  Rg = 2M）
# ============================================================
def geodesic(tau, y, L):
    """y = (r, dr/dτ, φ)"""
    r, dr, phi = y
    # M = Rg / 2
    d2r  = -0.5*Rg/r**2 + L**2/r**3 - 1.5*Rg*L**2/r**4
    dphi = L / r**2
    return [dr, d2r, dphi]

def horizon_event(tau, y, *args):
    return y[0] - 1.01*Rg          # r = 2.01 M で停止
horizon_event.terminal  = True
horizon_event.direction = -1       # r が減少方向でのみ計算停止

r_max = np.max(np.abs(b_list)) * 1.   # グラフ範囲を少し広めに設定
def event_escape(tau, y, *args):
    r = y[0]
    return r - r_max
event_escape.terminal  = True
event_escape.direction = 1

# ============================================================
# 2.  軌道計算 & プロット
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

for start_axis in ("x", ):#("y", "x"):           # (0,b)→±x   /  (b,0)→±y
    for b, col in zip(b_list, colors):
        for sign in (+1,): # -1):           # 発射方向の符号

            # --- 起点と速度ベクトル（座標速度） ---------------
            if start_axis == "y":       # (x, y) = (0, b)  → ±x
                x0, y0 = 0.0, b
                vx0, vy0 = sign * v0, 0.0
            else:                       # (x, y) = (b, 0)  → ±y
                x0, y0 = b, -10.0
                vx0, vy0 = 0.0, sign * v0

            # --- 初期極座標 -----------------------------------
            r0   = np.hypot(x0, y0)
            phi0 = np.arctan2(y0, x0)

            # --- dr/dτ, dφ/dτ (τ パラメータ化) ----------------
            dr0   = np.cos(phi0)*vx0 + np.sin(phi0)*vy0
            dphi0 = (-np.sin(phi0)*vx0 + np.cos(phi0)*vy0) / r0

            # --- 角運動量 L = b * γ v  ------------------------
            # 近似：γ ≈ 1/√(1-v0²). 速度が同じなので比例定数は共通。
            L = b * v0 / np.sqrt(1 - v0**2) * sign

            # --- 数値積分 -------------------------------------
            y_init = [r0, dr0, phi0]
            sol = solve_ivp(
                geodesic, (0.0, 500.0), y_init,
                args=(L,),
                events=(horizon_event, event_escape),
                rtol=1e-11, atol=1e-13
            )

            # --- 軌跡をデカルト座標へ -------------------------
            r_vals, phi_vals = sol.y[0], sol.y[2]
            x_vals = r_vals * np.cos(phi_vals)
            y_vals = r_vals * np.sin(phi_vals)

            # --- 描画 -----------------------------------------
            lbl = (f"{'(0,b)' if start_axis=='y' else '(b,0)'} "
                   f"b={b:+.1f}, "
                   f"{'+' if sign>0 else '-'}"
                   f"{'x' if start_axis=='y' else 'y'}")
            ax.plot(x_vals, y_vals, color=col, lw=0.9, label=lbl)

# --- 地平面 --------------------------------------------------
ax.add_patch(plt.Circle((0, 0), Rg, color='k', alpha=0.35, label='horizon'))

lim = 1.1 * r_max
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
#ax.set_title("Massive-particle geodesics in Schwarzschild spacetime")
#ax.grid(True, ls=':')
#ax.legend(fontsize=7, ncol=2, loc='upper right')
plt.tight_layout(); 

# --- 保存手続き ---
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Particle_Scattering_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- 保存 ---
fig.savefig(OUTDIR/"Multiparticle_scattering_in Schwarzschild_spacetime.pdf",dpi=300)
print("保存が完了しました。")

# --- プレビュ ---
plt.show()
