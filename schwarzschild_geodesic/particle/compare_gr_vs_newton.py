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
y0 = 0.0
vx0 = 0.0
vy0 = 0.27             # 0 < v < 1

# =========================================
# 2. 座標変換・不変量計算
# =========================================
def Coordinate_r_phi(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

def Coordinate_dr_dphi(vx, vy, r, phi):
    dr = np.cos(phi)*vx + np.sin(phi)*vy
    dphi = (-np.sin(phi)*vx + np.cos(phi)*vy) / r
    return dr, dphi

r0, phi0 = Coordinate_r_phi(x0, y0)
dr0, dphi0 = Coordinate_dr_dphi(vx0, vy0, r0, phi0)

f0        = 1 - Rg / r0
dt_dtau0  = np.sqrt((1 + dr0**2 + (r0*dphi0)**2) / f0)
# 保存エネルギーと角運動量（無次元）
L = r0**2 * dphi0            # L / (m Rg c)
E_Einstein = f0 * dt_dtau0  
E_Newton = (dr0**2 + r0**2*dphi0**2)/(2*c**2) - Rg/r0 # Hamiltonian / (mc^2)

print("L =",L)
print("E_Einstein =",E_Einstein )
print("E_Newton", E_Newton)

# =========================================
# 3. 数値計算設定(共通)
# =========================================

# --- 計算終了条件 ---
r_max = r0*1.5

# --- イベント関数 ---
def event_horizon(tau, y):
    return y[0] - Rg*1.001
event_horizon.terminal = True
event_horizon.direction = -1

def event_escape(tau, y):
    return y[0] - r_max  # 無次元 r の上限
event_escape.terminal = True
event_escape.direction = 1

# --- 積分範囲と初期条件まとめ ---
tau_span = (0, 5000)  # 無次元時間
tau_eval = np.arange(*tau_span, 0.01)
y0_vec = [r0, dr0, phi0, 0.0]

# =========================================
# 4. 数値計算
# =========================================
#################### 一般相対論 ####################
# --- 測地線方程式（無次元） ---
def Einstein_equation(tau, y):
    r, dr, phi, t = y
    f = 1 - Rg / r
    dphi = L / r**2
    dt = E_Einstein / f
    d2r = (Rg / (2*f*r**2)) * (dr**2 - c**2*E_Einstein**2) + f * L**2 / r**3
    return [dr, d2r, dphi, dt]

# --- Einstein解の取得 ---
sol_Einstein = solve_ivp(Einstein_equation, tau_span, y0_vec, t_eval=tau_eval,
                events=[event_horizon, event_escape])
tau_Einstein_vals = sol_Einstein.t
r_Einstein_vals = sol_Einstein.y[0]
phi_Einstein_vals = sol_Einstein.y[2]
t_Einstein_vals = sol_Einstein.y[3]

# --- 観測者のEinstein時間で再サンプリング ---
tau_of_t_Einstein = interp1d(t_Einstein_vals, tau_Einstein_vals, kind='cubic', fill_value="extrapolate", bounds_error=False)
t_Einstein_list = np.linspace(t_Einstein_vals[0], t_Einstein_vals[-1], 3000)
tau_Einstein_list = tau_of_t_Einstein(t_Einstein_list)
r_Einstein_list = np.interp(tau_Einstein_list, tau_Einstein_vals, r_Einstein_vals)
phi_Einstein_list = np.interp(tau_Einstein_list, tau_Einstein_vals, phi_Einstein_vals)

x_Einstein_list = r_Einstein_list * np.cos(phi_Einstein_list)
y_Einstein_list = r_Einstein_list * np.sin(phi_Einstein_list)


#################### ニュートン理論 ####################
def Neton_equation(tau, y):
    r, dr, phi, t = y
    dphi = L / r**2
    dt = 1
    d2r = -(Rg*c**2)/(2*r**2) + L**2/r**3
    return [dr, d2r, dphi, dt]

# --- Newton解の取得 ---
sol_Newton = solve_ivp(Neton_equation, tau_span, y0_vec, t_eval=tau_eval,
                events=[event_horizon, event_escape])
tau_Newton_vals = sol_Newton.t
r_Newton_vals = sol_Newton.y[0]
phi_Newton_vals = sol_Newton.y[2]
t_Newton_vals = sol_Newton.y[3]

# --- 観測者のNewton時間で再サンプリング ---
tau_of_t_Newton = interp1d(t_Newton_vals, tau_Newton_vals, kind='cubic', fill_value="extrapolate", bounds_error=False)
t_Newton_list = np.linspace(t_Newton_vals[0], t_Newton_vals[-1], 3000)
tau_Newton_list = tau_of_t_Newton(t_Newton_list)
r_Newton_list = np.interp(tau_Newton_list, tau_Newton_vals, r_Newton_vals)
phi_Newton_list = np.interp(tau_Newton_list, tau_Newton_vals, phi_Newton_vals)

x_Newton_list = r_Newton_list * np.cos(phi_Newton_list)
y_Newton_list = r_Newton_list * np.sin(phi_Newton_list)

##################################################
# --- プロット数の削減 ---
def downsample(arrays, N=300):
    """
    arrays : 同じ長さを持つ 1D NumPy 配列 or list を並べたリスト
    N      : 抽出したい点数
    return : 各配列を同じインデックスで間引いた新しい list
    """
    length = len(arrays[0])
    if N >= length:                       # もとの点数以下なら触らない
        idx = np.arange(length)
    else:
        idx = np.linspace(0, length-1, N, dtype=int)
    return [arr[idx] for arr in arrays]

Nomber_of_plotpoint = 300  # ← ここで一括管理

# ----------------------------------------------------------
# Einstein 系列
(t_Einstein_vals_plot,
 tau_Einstein_vals_plot,
 r_Einstein_vals_plot) = downsample(
        [t_Einstein_vals, tau_Einstein_vals, r_Einstein_vals],
        Nomber_of_plotpoint)

# Newton 系列
(t_Newton_vals_plot,
 tau_Newton_vals_plot,
 r_Newton_vals_plot) = downsample(
        [t_Newton_vals, tau_Newton_vals, r_Newton_vals],
        Nomber_of_plotpoint)

# Observer‑time 列（r(t) 用）
(t_Einstein_list_plot,
 r_Einstein_list_plot) = downsample(
        [t_Einstein_list, r_Einstein_list],
        Nomber_of_plotpoint)

(t_Newton_list_plot,
 r_Newton_list_plot) = downsample(
        [t_Newton_list, r_Newton_list],
        Nomber_of_plotpoint)

# 軌跡 (x,y) 列
(x_Einstein_list_plot,
 y_Einstein_list_plot) = downsample(
        [x_Einstein_list, y_Einstein_list],
        Nomber_of_plotpoint)

(x_Newton_list_plot,
 y_Newton_list_plot) = downsample(
        [x_Newton_list, y_Newton_list],
        Nomber_of_plotpoint)


# =========================================
# 5. 可視化
# =========================================
# --- プロット ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(t_Newton_list, tau_Newton_list, c='red', marker='*', markersize=2 ,label="Newton")
axs[0, 0].plot(t_Einstein_list, tau_Einstein_list, c='blue',label="Einstein")
axs[0, 0].set_xlabel(r"Observer Time $[t]$")
axs[0, 0].set_ylabel(r"Proper Time $[\tau]$")
axs[0, 0].grid(True)
axs[0, 0].legend()
axs[0, 0].set_aspect('equal')
axs[0, 0].set_title("Observer time and proper time")

axs[0, 1].plot(tau_Newton_list, r_Newton_list, c='red', marker='*', markersize=2 ,label="Newton")
axs[0, 1].plot(tau_Einstein_list, r_Einstein_list, c='blue',label="Einstein")
axs[0, 1].set_xlabel(r"Proper Time $[\tau]$")
axs[0, 1].set_ylabel(r"Radial Distance $[r]$")
axs[0, 1].grid(True)
axs[0, 1].legend()
axs[0, 1].set_title("Proper Time and Radial Distance")

axs[1, 0].plot(t_Newton_list, r_Newton_list, c='red', marker='*', markersize=2 ,label="Newton")
axs[1, 0].plot(t_Einstein_list, r_Einstein_list, c='blue',label="Einstein")
axs[1, 0].set_xlabel(r"Observer Time $[t]$")
axs[1, 0].set_ylabel(r"Radial Distance $[r]$")
axs[1, 0].grid(True)
axs[1, 0].legend()
axs[1, 0].set_title("Observer Time and Radial Distance")

r_lim = np.max( [max(r_Einstein_list)*1.1,max(r_Newton_list)*1.1])
axs[1, 1].plot(x_Newton_list, y_Newton_list, c='red', marker='*', markersize=2 ,label="Newton")
axs[1, 1].plot(x_Einstein_list, y_Einstein_list, c='blue',label="Einstein")
axs[1, 1].add_patch(plt.Circle((0, 0), Rg, color='black', alpha=0.5)) 
axs[1, 1].set_xlim(-r_lim, r_lim)
axs[1, 1].set_ylim(-r_lim, r_lim)
axs[1, 1].set_xlabel(r"$x$")
axs[1, 1].set_ylabel(r"$y$")
axs[1, 1].set_title("Trajectory of a Particle ")
axs[1, 1].grid(True)
axs[1, 1].legend()
axs[1, 1].set_aspect('equal')
fig.tight_layout()

# =========================================
# 6. 保存手続き
# =========================================
# --- 保存 ---
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Comparison_Schwarzschild_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- パラメータの記録 ---
sim_params = {
    "日時"          : datetime.now().isoformat(timespec="seconds"), #now_str,
    "重力源の質量M_SI ="  : f"{M_SI:.3e}  [kg]",
    "Rg_SI =": f"{Rg_SI:.3e} [m]",
    "x0_SI = "            : f"{x0:.3e} [m]",
    "y0_SI = "            : f"{y0:.3e} [m]",
    "vx0_SI = "           : f"{vx0:.3e} [m/s]",
    "vy0_SI = "           : f"{vy0:.3e} [m/s]",
    "粒子の(非相対論的)エネルギー ="       : f"{E_Newton:.6f}",
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
    ax_new.legend()

    if add_blackhole:
        ax_new.add_patch(plt.Circle((0, 0), Rg, color='black', alpha=0.5))
        ax_new.set_xlim(-r_lim, r_lim)
        ax_new.set_ylim(-r_lim, r_lim)
        ax_new.set_aspect("equal")

    plt.tight_layout()
    # ファイルを output_dir に保存
    fig_new.savefig(OUTDIR /  f"{title}.pdf", dpi=300)
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
fig.savefig(OUTDIR / "Comparison_graphs_of_a_particle_of_Schwarzschild_spacetime.pdf")

# --- 各図を個々に保存 ---
copy_subplot_to_new_figure(axs[0, 0],title="Comparison_tau_t_of_a_particle_of_Schwarzschild_spacetime")  # tau vs t
copy_subplot_to_new_figure(axs[0, 1],title="Comparison_r_tau_of_a_particle_of_Schwarzschild_spacetime")  # r vs tau
copy_subplot_to_new_figure(axs[1, 0],title="Comparison_r_t_of_a_particle_of_Schwarzschild_spacetime")  # r vs t
copy_subplot_to_new_figure(axs[1, 1],title="Comparison_x_y_of_a_particle_of_Schwarzschild_spacetime",add_blackhole=True)  # 軌道

# --- プレビュ ---
plt.show()