import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# ----------------------------------------
# 物理・数値パラメータ
# ----------------------------------------
Rg = 2.0                  # 無次元シュワルツシルト半径 (2M)
v0 = 0.4                  # 初期 3-速度の大きさ (0<v0<1)
b  = 10.0                 # インパクトパラメータ
tau_max = 5000.0
rtol, atol = 1e-11, 1e-13
Nframes = 500             ## テスト用にフレーム数を少し減らしています -> ここの数値を増やしてください ##

# ----------------------------------------
# 初期条件と保存量の計算
# ----------------------------------------
x0, y0 = 0.0, b
vx0, vy0 = v0, 0.0

r0   = np.hypot(x0, y0)
phi0 = np.arctan2(y0, x0)
f0   = 1.0 - Rg / r0

dr0   = np.cos(phi0)*vx0 + np.sin(phi0)*vy0
dphi0 = (-np.sin(phi0)*vx0 + np.cos(phi0)*vy0) / r0
L     = r0**2 * dphi0

# 正規化条件から dt/dτ を導く
dt_dtau0 = np.sqrt( (1 + dr0**2/f0 + r0**2*dphi0**2) / f0 )
E        = f0 * dt_dtau0

# ----------------------------------------
# ODE (測地線方程式)
# ----------------------------------------
def geodesic(tau, y, L, E):
    r, dr, phi, t = y
    f = 1.0 - Rg / r
    # 有質量粒子の動径方向の方程式
    d2r  = -0.5*Rg/r**2 + L**2/r**3 - 1.5*Rg*L**2/r**4
    dphi = L / r**2
    dt   = E / f
    return [dr, d2r, dphi, dt]

# イベント検知（地平面到達と脱出）
def horizon_event(tau, y, *args):
    return y[0] - 1.000001*Rg
horizon_event.terminal = True
horizon_event.direction = -1

def event_escape(tau, y, *args):
    return y[0] - 50.0  # 十分遠方へ逃げた場合
event_escape.terminal = True

# ----------------------------------------
# 数値積分
# ----------------------------------------
y_init = [r0, dr0, phi0, 0.0]
sol = solve_ivp(
    geodesic, (0.0, tau_max), y_init,
    args=(L, E),
    events=[horizon_event, event_escape],
    rtol=rtol, atol=atol
)

# 観測者時間 t に基づく補間
t_raw = sol.y[3]
r_raw = sol.y[0]
phi_raw = sol.y[2]

t_obs = np.linspace(0, t_raw[-1], Nframes)

# 3次スプライン補間モデルを作成（点と点の間を滑らかな曲線でつなぐ準備）
spline_r = CubicSpline(t_raw, r_raw)
spline_phi = CubicSpline(t_raw, phi_raw)

# 作成したモデルに t_obs を渡して、高精度な r と phi を計算
r_obs = spline_r(t_obs)
phi_obs = spline_phi(t_obs)

x_obs = r_obs * np.cos(phi_obs)
y_obs = r_obs * np.sin(phi_obs)

# ----------------------------------------
# アニメーション設定
# ----------------------------------------
fig, ax = plt.subplots(figsize=(6,6))
lim = 1.5 * b
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid(ls=':')
ax.set_xlabel("$x\,[M]$"); ax.set_ylabel("$y\,[M]$")

# ブラックホール（シュヴァルツシルト半径）
ax.add_patch(plt.Circle((0,0), Rg, color='black', alpha=0.5, label="Horizon"))

orbit_line, = ax.plot([], [], lw=1.5, color='tab:blue')
particle,   = ax.plot([], [], 'o', color='crimson', markersize=6)

def init():
    orbit_line.set_data([], [])
    particle.set_data([], [])
    return orbit_line, particle

def update(frame):
    # 軌跡（これまでの全データ）
    orbit_line.set_data(x_obs[:frame], y_obs[:frame])
    # 現在の粒子位置（スカラーを [ ] で囲んでリストにするのがポイント！）
    particle.set_data([x_obs[frame]], [y_obs[frame]])
    return orbit_line, particle

ani = FuncAnimation(fig, update, frames=Nframes,
                    init_func=init, blit=True, interval=20)

# --- 保存手続き ---
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Particle_Animation_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- 保存 ---
ani.save(OUTDIR/"animation_of_particle_orbits.mp4", dpi=250)
print("保存が完了しました。")

# --- プレビュ ---
plt.show()