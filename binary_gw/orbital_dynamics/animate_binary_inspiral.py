import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
from scipy.interpolate import interp1d  # ★追加

# 同じフォルダに計算コードのファイルが存在することを確認
try:
    import Calculating_gravitational_waves_from_binary_star_systems as gw_phys
except ImportError:
    # ファイル名が異なる場合のフォールバック
    print("Error: 指定された物理計算モジュールが見つかりません。ファイル名を確認してください。")
    exit()

# ==========================================
# 物理定数 (SI単位系)
# ==========================================
G_SI = 6.67430e-11       # [m^3 kg^-1 s^-2]
C_SI = 2.99792458e8      # [m s^-1]
M_SUN_SI = 1.9891e30     # [kg]
PC_SI = 3.0857e16        # [m]
AU = 1.496e11            # [m]
YEAR_SI = 31557600.0     # [s] (365.25日)
"""
計算コード内部ではc=M=Rg=1を計算単位として桁落ち等を防ぐ。
"""

# 連星系の初期条件 (GW150914のようなブラックホール連星を想定しつつ、可視化用にアレンジ)
m1 = 30 * M_SUN_SI
m2 = 30 * M_SUN_SI
Rg_SI = (2 * G_SI * (m1+m2)) / (C_SI**2)
a0 = 10*Rg_SI   # 初期長半径 (非常に近接させて計算時間を短縮)
e0 = 0.6         # 離心率の変化を見るためにあえて高い値を設定
r_obs = 1e22     # 観測距離 (波形計算時に使用,今回は軌道描画のみなのでダミー)

# 計算クラスのインスタンス化
binary = gw_phys.BinaryRadiatingGravitationalWave(
    m1_SI=m1, m2_SI=m2, a0_SI=a0, e0=e0, r_SI=r_obs, theta=0, iota=0
)

print("物理計算を実行中...")
t_data, state_data = binary.Archive_for_state_of_binary_star_systems()
print(f"計算完了: 合体までの時間 = {t_data[-1]:.4f} s")

# 生データの取得
raw_a = state_data[0]
raw_e = state_data[1]
raw_phi = state_data[2]
raw_t = t_data

# ==========================================
# 等時間間隔リサンプリング
# ==========================================
# 1. 補間関数の作成
#    phiはunwrapして連続値にしておく（2piで飛ばないように）
raw_phi_unwrapped = np.unwrap(raw_phi)
func_a = interp1d(raw_t, raw_a, kind='linear', fill_value="extrapolate")
func_e = interp1d(raw_t, raw_e, kind='linear', fill_value="extrapolate")
func_phi = interp1d(raw_t, raw_phi_unwrapped, kind='linear', fill_value="extrapolate")

# 2. アニメーション用の「完全に均一な」時間配列を作成
num_frames = 2500  # 総フレーム数
t_merger = raw_t[-1]
t_anim = np.linspace(0, t_merger, num_frames)

# 3. 均一な時間軸 t_anim に基づいて物理量を再取得
a_anim = func_a(t_anim)
e_anim = func_e(t_anim)
phi_anim = func_phi(t_anim)

# ==========================================
# 座標変換: 軌道要素 -> デカルト座標
# ==========================================
# 4. 座標変換
r_anim = a_anim * (1 - e_anim**2) / (1 + e_anim * np.cos(phi_anim))
x_rel_anim = r_anim * np.cos(phi_anim)
y_rel_anim = r_anim * np.sin(phi_anim)

# 重心系における各天体の位置
# M = m1 + m2
M_total = m1 + m2
x1 = (m2 / M_total) * x_rel_anim
y1 = (m2 / M_total) * y_rel_anim
x2 = -(m1 / M_total) * x_rel_anim
y2 = -(m1 / M_total) * y_rel_anim

# ==========================================
# アニメーション設定
# ==========================================
# 描画をスムーズにするため、データを間引く (フレームレート調整)
# 全データ点数が多すぎる場合、アニメーション生成が重くなるため
'''
num_frames = 10000  # 作成する総フレーム数
step = max(1, len(t_data) // num_frames)

# スライスを使用したデータの間引き
t_anim = t_data[::step]
x1_anim = x1[::step]
y1_anim = y1[::step]
x2_anim = x2[::step]
y2_anim = y2[::step]
a_anim = a_anim[::step]
e_anim = e_anim[::step]
'''
# プロットのセットアップ
#plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 8))
ax.grid(True, alpha=0.3)

# 軸の範囲設定 (初期軌道の1.2倍程度で固定)
limit = np.max(np.abs(np.concatenate([x1, x2, y1, y2]))) * 1.2
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
#ax.set_title(f"Binary System Inspiral (Mass: {m1/M_SUN_SI:.1f}+{m2/M_SUN_SI:.1f} Msun)")

# プロット要素の初期化
trail_len = 200 # 軌跡の長さ
line1, = ax.plot([], [], color='tab:blue', linestyle='--', alpha=0.3) # 天体1の軌跡
line2, = ax.plot([], [], color='tab:red', linestyle='--', alpha=0.3) # 天体2の軌跡
star1, = ax.plot([], [], 'o', color='tab:blue', markersize=30, label='M1')
star2, = ax.plot([], [], 'o', color='tab:red', markersize=30, label='M2')

# 情報表示用テキスト
info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white',
                    verticalalignment='top', fontfamily='monospace')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    star1.set_data([], [])
    star2.set_data([], [])
    info_text.set_text('')
    return  star1, star2, info_text # line1, line2,

def update(frame):
    # 現在のインデックス
    idx = frame
    
    # 軌跡のデータ範囲
    start_idx = max(0, idx - trail_len)
    
    # 天体1の更新
    line1.set_data(x1[start_idx:idx], y1[start_idx:idx])
    star1.set_data([x1[idx]], [y1[idx]]) # 配列として渡す
    
    # 天体2の更新
    line2.set_data(x2[start_idx:idx], y2[start_idx:idx])
    star2.set_data([x2[idx]], [y2[idx]])
    
    # テキスト情報の更新
    # 合体までの残り時間
    time_left = t_anim[-1] - t_anim[idx]
    
    info = (
        f"Time: {t_anim[idx]:.2f} s\n"
        f"Time to Merger: {time_left:.2f} s\n"
        f"Semi-major axis (a): {a_anim[idx]:.3e} m\n"
        f"Eccentricity (e): {e_anim[idx]:.4f}"
    )
    info_text.set_text(info)
    
    return star1, star2, info_text, line1, line2, 

# アニメーション作成
ani = FuncAnimation(fig, update, frames=len(t_anim), init_func=init, 
                    interval=20, blit=True)
plt.tight_layout()

# 保存する場合 (ffmpegが必要)
#ani.save('00_binary_inspiral.mp4', dpi=250, fps=30)

plt.show()