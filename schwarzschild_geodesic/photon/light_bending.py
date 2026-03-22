import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

# --- 物理定数（SI単位） ---
c_SI = 2.99792458e8
G_SI = 6.67430e-11
M_sun_SI = 1.98847e30
h_SI = 6.626e-34
Rg_sun = 2 * G_SI * M_sun_SI / c_SI**2  # 太陽のシュワルツシルト半径

# --- パラメータ（幾何単位 G=c=1） ---
M_SI  = 2.0 * M_sun_SI
Rg_SI = 2.0 * (G_SI*M_SI/c_SI**2)
lamda_SI = 550e-9
E_SI = h_SI*c_SI/lamda_SI

# --- 無次元化 ---
c  = 1.0
M  = M_SI / M_sun_SI
Rg = Rg_SI / Rg_sun
E  = 1.0
b = np.linspace(-30, 30, 30)      # [-20 M, +20 M]
b_list = b[np.abs(b) > Rg]       # Rg = 2 → |b| ≤ 2 を排除

# ---- カラーマップ（b の値で共通） ----

colors = cm.cool(np.linspace(0.5, 0.5, len(b_list)))

# =========================================
# r 形式の測地線方程式
# =========================================
def geodesic(s, y, L):
    r, dr, phi = y
    d2r  = (L**2/r**3)*(1 - Rg/r) - 0.5*L**2*Rg/r**4
    dphi = L / r**2
    return [dr, d2r, dphi]

def horizon_event(s, y, *args):
    return y[0] - 1.01*Rg          # r = 2.01 M で停止
horizon_event.terminal, horizon_event.direction = True, -1

r_max = np.max(np.abs(b_list))
def event_escape(tau, y):
    r = 1/y[0]
    return r - r_max  # 無次元 r の上限
event_escape.terminal = True
event_escape.direction = 1

# =========================================
# 軌道計算
# =========================================
fig, ax = plt.subplots(figsize=(7, 7))

for start_axis in ("y", "x"):              # 起点 (0,b) or (b,0)
    for b, col in zip(b_list, colors):

        # --- 発射方向は ±1 で表現（+1: +軸方向, -1: -軸方向） ---
        for sign in (+1, -1):
            # 起点と速度ベクトル
            if start_axis == "y":          # (x, y) = (0, b) → ±x
                x0, y0 = 10.0, b
                vx0,  vy0 = sign*c, 0.0
            else:                          # (x, y) = (b, 0) → ±y
                x0, y0 = b, 10.0
                vx0,  vy0 = 0.0, sign*c

            # --- 初期極座標 ---
            r0   = np.hypot(x0, y0)
            phi0 = np.arctan2(y0, x0)

            # --- (dr/ds, dphi/ds) 初期値 ---
            dr0   = np.cos(phi0)*vx0 + np.sin(phi0)*vy0
            dphi0 = (-np.sin(phi0)*vx0 + np.cos(phi0)*vy0)/r0

            # --- 角運動量 L = b*E*sign ---
            L = b * E * sign

            # --- 積分 ---
            y_init = [r0, dr0, phi0]
            sol = solve_ivp(
                geodesic, (0.0, 1000.0), y_init,
                args=(L,), events=horizon_event,
                rtol=1e-10, atol=1e-12
            )

            # --- デカルト座標へ変換 ---
            r_vals, phi_vals = sol.y[0], sol.y[2]
            x_vals = r_vals*np.cos(phi_vals)
            y_vals = r_vals*np.sin(phi_vals)

            label = (f"start {'(0,b)' if start_axis=='y' else '(b,0)'}  "
                     f"b={b:+.1f}, "
                     f"{'+' if sign>0 else '-'}"
                     f"{'x' if start_axis=='y' else 'y'}")
            ax.plot(x_vals, y_vals, color=col, lw=1, label=label)

# 地平面
ax.add_patch(plt.Circle((0, 0), Rg, color='k', alpha=0.3, label='horizon'))

lim = 1.1 * max(np.abs(b_list))
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
ax.set_title("Photon geodesics")
plt.tight_layout()

# --- 保存手続き ---
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Lens_Shape_Photon_Geodesics_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- 保存 ---
fig.savefig(OUTDIR/"Gravitational_lensing.pdf", dpi=300)
print("保存が完了しました。")

# --- プレビュ ---
plt.show()
