import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

# ------------------------------------------
# 0. 定数（幾何化単位 G=c=1）
# ------------------------------------------
M   = 1.0
Rg  = 2.0*M
E   = 1.0                    # エネルギーを 1 に固定

# 発射点
x0, y0   = 0., -20.0
r0       = np.hypot(x0, y0)
phi0     = np.arctan2(y0, x0)
f0       = 1.0 - 2.0*M/r0    # 1 - 2M/r0

# 角度リスト（0〜2π まで N 分割）
N_theta  = 1000
theta_list = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
colors   = cm.plasma( np.concatenate([np.linspace(0, 1, int(N_theta/2)), np.linspace(1, 0, int(N_theta/2)) ]) )

# ------------------------------------------
# 1. 測地線方程式（一次系）
# ------------------------------------------
def geodesic(lmbd, y, L):
    r, pr, phi, t = y
    f   = 1.0 - 2.0*M/r
    dpr = (L**2/r**3)*(1-2*M/r) - M*pr**2/r**2
    dphi= L / r**2
    dt  = E / f
    return [pr, dpr, dphi, dt]

def horizon(lmbd, y, *a):
    return y[0] - 2.01*M
horizon.terminal, horizon.direction = True, -1

r_max = np.max(np.abs(r0))*2
def event_escape(tau, y,*a):
    r = y[0] 
    return r - r_max  # 無次元 r の上限
event_escape.terminal = True
event_escape.direction = 1

# ------------------------------------------
# 2. 積分 & 描画
# ------------------------------------------
fig, ax = plt.subplots(figsize=(8,8))
max_r_seen = r0

for th, col in zip(theta_list, colors):

    # ---- (a) 単位方向ベクトル ----
    vx_dir, vy_dir = np.cos(th), np.sin(th)

    # ---- (b) 正規化係数 v_scale ----
    #   a = (v·hat_r)^2 + f0 (v·hat_phi)^2
    a = (vx_dir*np.cos(phi0) + vy_dir*np.sin(phi0))**2 \
      + f0 * (-vx_dir*np.sin(phi0) + vy_dir*np.cos(phi0))**2
    v_scale = E / np.sqrt(a)

    # ---- (c) デカルト速度 ----
    vx0, vy0 = v_scale*vx_dir, v_scale*vy_dir

    # ---- (d) 極座標速度 -> 保存量 ---
    pr0   = vx0*np.cos(phi0) + vy0*np.sin(phi0)          # dr/dλ
    dphi0 = (-vx0*np.sin(phi0) + vy0*np.cos(phi0)) / r0  # dφ/dλ
    L     = r0**2 * dphi0

    # ---- (e) 初期ベクトル y=[r,pr,φ,t] ---
    y_init = [r0, pr0, phi0, 0.0]

    # ---- (f) 積分 ----
    sol = solve_ivp(
        geodesic, (0, 100), y_init,
        args=(L,), events=[horizon,event_escape],
        rtol=1e-10, atol=1e-12, method='RK45'
    )
    
    if sol.t_events[0].size == 0:          # ← 落ち込んでいない軌跡のみ可視化
        r, phi = sol.y[0], sol.y[2]
        max_phi = max(th, 0)
        max_r_seen = max(max_r_seen, r.max())
        x, y   = r*np.cos(phi), r*np.sin(phi)
        ax.plot(x, y, color=col, lw=1.0)

print("落ちないギリギリの入射角(垂直を0) =",th)

# ------------------------------------------
# 3. グラフ仕上げ
# ------------------------------------------
ax.add_patch(plt.Circle((0,0), Rg, color='k', alpha=0.35, label='horizon'))
lim = 2*r0 #1.05*max_r_seen
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal'); ax.grid(True)
ax.set_xlabel(r"$x\,[M]$"); ax.set_ylabel(r"$y\,[M]$")
ax.set_title(f"Does not fall Photon geodesics from (x0={x0}, y0={y0})")
plt.tight_layout()

# --- 保存手続き ---
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Fan_Shape_Photon_Geodesics_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- 保存 ---
fig.savefig(OUTDIR/"Does not fall Photon geodesics.pdf", dpi=300)
print("保存が完了しました。")

# --- プレビュ ---
plt.show()