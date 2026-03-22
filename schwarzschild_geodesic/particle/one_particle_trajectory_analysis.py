import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =========================================
# 0. 基本定数（SI）とスケール
# =========================================
c_SI = 2.99792458e8
G_SI = 6.67430e-11
M_sun_SI = 1.98847e30
Rg_sun_SI = 2 * G_SI * M_sun_SI / c_SI**2   # ⇒ 長さスケール

# ── 質量・シュワルツシルト半径（今回は 1 M⊙）──
M_SI  = M_sun_SI
Rg_SI = 2 * G_SI * M_sun_SI / c_SI**2   # ⇒ 長さスケール

# =========================================
# 1. 無次元パラメータ
# =========================================
Rg = Rg_SI/Rg_sun_SI       # = 1
M  = M_SI/M_sun_SI  #Rg / 2.0          # = 0.5
c  = 1.0               # 幾何単位

# 初期位置・速度（無次元）
x0 = 10.0              # = 10 Rg
y0 = -10.0
vx0 = 0.0
vy0 = 0.2 #            # 0 < v < 1 (数周回：0.4039 )

# =========================================
# 2. 補助関数
# =========================================
def to_r_phi(x, y):
    r   = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return r, phi

def to_dr_dphi(vx, vy, r, phi):
    dr   = np.cos(phi)*vx + np.sin(phi)*vy
    dphi = (-np.sin(phi)*vx + np.cos(phi)*vy) / r
    return dr, dphi

# 初期値
r0,  phi0  = to_r_phi(x0, y0)
dr0, dphi0 = to_dr_dphi(vx0, vy0, r0, phi0)

# 保存量
L = r0**2 * dphi0
# timelike 条件  -1 = g_{μν}u^μu^ν  →  (dr/dτ)² + r²(dφ/dτ)² + f (dt/dτ)² = 1
f0        = 1 - Rg / r0
dt_dtau0  = np.sqrt((1 + dr0**2 + (r0*dphi0)**2) / f0)
E_Einstein = f0 * dt_dtau0          # エネルギー

# =========================================
# 4. 有効ポテンシャル V_eff(r) の描画
# =========================================
def ax_with_Effective_potential_foundation(ax, L, Rg, grid_lim, resolution=500, cmap='plasma'):

    # x–y 平面上の格子生成
    x = np.linspace(-grid_lim, grid_lim, resolution)
    y = np.linspace(-grid_lim, grid_lim, resolution)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # シュワルツシルト半径より内側には描画しない（Veffが虚になる領域）
    epsilon = 1e-3
    R[R < Rg + epsilon] = Rg + epsilon  # 発散防止のため最低値制限

    # 有効ポテンシャル V_eff(r)
    f = 1 - Rg / R
    Veff = ( -(Rg/R) + L**2*(Rg/R)**2 - L**2*(Rg/R)**3 )*0.5*c

    # カラーマップ表示
    im = ax.pcolormesh(X, Y, Veff, shading='auto', cmap=cmap)
    # ax.set_aspect('equal')
    # ax.set_xlim(-grid_lim, grid_lim)
    # ax.set_ylim(-grid_lim, grid_lim)
    # ax.set_title(r"$V_{\rm eff}(r)$ in $xy$ plane")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    # カラーバーを追加（必要なら）
    plt.colorbar(im, ax=ax, label=r"$V_{\rm eff}(r)$")

# =========================================
# 3. 測地線 ODE  y = (r, dr/dτ, φ, t)
# =========================================
def geodesic(tau, y):
    r, dr, phi, t = y
    f = 1 - Rg / r
    d2r  = -M/r**2 + L**2/r**3 - 3*M*L**2/r**4      # 正式形
    dphi = L / r**2
    dt   = E_Einstein / f
    return [dr, d2r, dphi, dt]

# =========================================
# 4. イベント
# =========================================
def event_horizon(tau, y):
    return y[0] - 1.000001*Rg
event_horizon.terminal, event_horizon.direction = True, -1

r_max =  r0
def event_escape(tau, y):
    return y[0] - r_max
event_escape.terminal, event_escape.direction = True, 1

# =========================================
# 5. 数値積分（τ）
# =========================================
tau_span = (0.0, 500.0)
y_init   = [r0, dr0, phi0, 0.0]

sol = solve_ivp(
    geodesic, tau_span, y_init, dense_output=True,
    events=(event_horizon, event_escape), rtol=1e-11, atol=1e-13
)

tau_vals  = sol.t
r_vals    = sol.y[0]
phi_vals  = sol.y[2]
t_vals    = sol.y[3]

# =========================================
# 6. 観測者時間で再サンプリング
# =========================================
t_obs = np.linspace(t_vals[0], t_vals[-1], 1500)
tau_of_t = interp1d(t_vals, tau_vals, kind='cubic',bounds_error=False, fill_value=(tau_vals[0],tau_vals[-2]))
tau_obs  = tau_of_t(t_obs)

r_obs   = np.interp(tau_obs, tau_vals, r_vals)
phi_obs = np.interp(tau_obs, tau_vals, phi_vals)
x_obs   = r_obs * np.cos(phi_obs)
y_obs   = r_obs * np.sin(phi_obs)

# =========================================
# 7. プロット
# =========================================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0,0].plot(t_vals, tau_vals)
axs[0,0].set_xlabel("t"); axs[0,0].set_ylabel("τ")
axs[0,0].set_title("Observer time and proper time"); axs[0,0].grid()

axs[0,1].plot(tau_vals, r_vals)
axs[0,1].set_xlabel("τ"); axs[0,1].set_ylabel("r")
axs[0,1].set_title("Proper Time and Radial Distanc"); axs[0,1].grid()

axs[1,0].plot(t_obs, r_obs)
axs[1,0].set_xlabel("t"); axs[1,0].set_ylabel("r")
axs[1,0].set_title("Observer Time and Radial Distance"); axs[1,0].grid()

r_Einstein_lim = 1.1*r_max
axs[1,1].plot(x_obs, y_obs)
axs[1,1].add_patch(plt.Circle((0,0), Rg, color='k', alpha=0.4))
ax_with_Effective_potential_foundation(axs[1, 1], L, Rg, grid_lim=r_Einstein_lim)
axs[1,1].set_xlim(-r_Einstein_lim, r_Einstein_lim)
axs[1,1].set_ylim(-r_Einstein_lim, r_Einstein_lim)
axs[1,1].set_aspect('equal'); axs[1,1].grid()
axs[1,1].set_xlabel("x"); axs[1,1].set_ylabel("y")
axs[1,1].set_title("Trajectory of a Particle ")

fig.tight_layout(); #plt.show()


# --- 保存 ---
import os
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Particle_Trajectory_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- パラメータの記録 ---
sim_params = {
    "日時"          : datetime.now().isoformat(timespec="seconds"), #now_str,
    "重力源の質量M_SI ="  : f"{M_SI:.3e}  [kg]",
    "Rg_SI =": f"{Rg_SI:.3e} [m]",
    "x0 = "            : f"{x0:.3e} [m]",
    "y0 = "            : f"{y0:.3e} [m]",
    "vx0 = "           : f"{vx0:.3e} [m/s]",
    "vy0 = "           : f"{vy0:.3e} [m/s]",
    "粒子の(相対論的)エネルギー ="       : f"{E_Einstein:.6f}",
    "粒子の角運動量L ="       : f"{L:.6f}",
}

# ---  個々のグラフ保存の関数 ---
def copy_subplot_to_new_figure(ax_original, title, add_blackhole=False):
    fig_new, ax_new = plt.subplots()
    for line in ax_original.get_lines():
        ax_new.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
    ax_new.set_xlabel(ax_original.get_xlabel())
    ax_new.set_ylabel(ax_original.get_ylabel())
    #ax_new.set_title(ax_original.get_title())
    ax_new.grid(True)

    if add_blackhole:
        ax_new.add_patch(plt.Circle((0, 0), Rg, color='black', alpha=0.5))
        ax_new.set_xlim(-r_Einstein_lim, r_Einstein_lim)
        ax_new.set_ylim(-r_Einstein_lim, r_Einstein_lim)
        ax_new.set_aspect("equal")

    plt.tight_layout()
    # ファイルを output_dir に保存
    fig_new.savefig(OUTDIR /  f"{title}.pdf")
    plt.close(fig_new)  # メモリ節約のため閉じる

# ------------------------ 3) CSV 保存ユーティリティ ------------------------
def save_params(data: dict, fname: str = "parameters") -> Path:
    """data を <OUTDIR>/<fname>.text に key,value 形式で保存して Path を返す"""
    path = OUTDIR / f"{fname}.text"
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(data.items())
    return path

# --- パラメータ保存 ---
save_params(sim_params, OUTDIR / "parameters")

# --- 4枚まとめた図を保存 ---
fig.tight_layout()
fig.savefig(OUTDIR / "graphs_of_a_particle_of_Schwarzschild_spacetime.pdf")

# --- 各図を個々に保存 ---
copy_subplot_to_new_figure(axs[0, 0],title="tau_t_of_a_particle_of_Schwarzschild_spacetime")  # tau vs t
copy_subplot_to_new_figure(axs[0, 1],title="r_tau_of_a_particle_of_Schwarzschild_spacetime")  # r vs tau
copy_subplot_to_new_figure(axs[1, 0],title="r_t_of_a_particle_of_Schwarzschild_spacetime")  # r vs t
copy_subplot_to_new_figure(axs[1, 1],title="x_y_of_a_particle_of_Schwarzschild_spacetime",add_blackhole=True)  # 軌道

# --- プレビュ ---
plt.show()