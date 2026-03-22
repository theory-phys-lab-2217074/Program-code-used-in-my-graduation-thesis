import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 物理計算モジュールのインポート
try:
    from Calculating_gravitational_waves_from_binary_star_systems import BinaryRadiatingGravitationalWave
except ImportError:
    print("Error: 物理計算モジュールが見つかりません。")
    exit()

# ==========================================
# 物理定数 & パラメータ
# ==========================================
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN_SI = 1.9891e30

# --- 連星系パラメータ  ---
m1_SI     = 30 * M_SUN_SI
m2_SI     = 30 * M_SUN_SI
Rg_SI     = (2 * G_SI * (m1_SI + m2_SI)) / (C_SI**2)
a0_SI     = 20 * Rg_SI
e0        = 0.6            
theta_inc = 0.0   
iota_azi  = 0.0

# --- アニメーション設定 ---
FPS = 30
DURATION_SEC = 10.0     

# ==========================================
# 物理計算 & グリッド生成
# ==========================================
print("物理計算を開始します...")
binary_system = BinaryRadiatingGravitationalWave(
    m1_SI, m2_SI, a0_SI, e0, r_SI=1.0, theta=theta_inc, iota=iota_azi
)
t_data, state_data = binary_system.Archive_for_state_of_binary_star_systems()
t_merger = t_data[-1]

# 空間グリッド
# 光速 * (合体までの時間 + 余韻)
spatial_width = C_SI * t_merger * 1.5 
R_MIN = Rg_SI * 200     # 中心付近の除外範囲
R_MAX = spatial_width
num_r_points = 5000   

r_array = np.linspace(R_MIN, R_MAX, num_r_points)
r_grid  = r_array[np.newaxis, :]

# 時間グリッド
# 波が空間の端まで到達する時間を考慮
total_phys_time = R_MAX / C_SI * 1.1 
total_frames = int(FPS * DURATION_SEC)
t_anim_array = np.linspace(0, total_phys_time, total_frames)
t_grid       = t_anim_array[:, np.newaxis]

print("時空波形行列を一括計算中...")
h_plus_matrix, h_cross_matrix = binary_system.Calculation_of_waveforms_at_distant_observation_locations(
    t_obs_SI=t_grid, r_obs_SI=r_grid, theta_inc=theta_inc, iota_azi=iota_azi
)

# ==========================================
# 可視化 (Linear Scale / White Background)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))

# --- デザイン修正: 白背景・黒文字 (参照元と統一) ---
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 軸ラベル等は黒に
ax.set_xlabel('Distance from Source $r$ [km]', color='black', fontsize=12)
ax.set_ylabel('Strain $h$ (Linear Scale)', color='black', fontsize=12)
ax.set_title(f'Gravitational Wave Propagation ($e_0={e0}, \\theta=\pi/2$)', color='black', fontsize=14)

ax.tick_params(colors='black', which='both')
for spine in ax.spines.values(): spine.set_color('black')

# プロットデータ準備 (km単位)
r_km = r_array / 1000.0

# 線の色: 背景が白なので、少し濃い目の色が見やすい
line_plus, = ax.plot([], [], color='tab:orange', lw=1.5, label='$h_+$')
line_cross, = ax.plot([], [], color='tab:blue', lw=1.5, alpha=0.7, label='$h_{\\times}$') # シアンから青へ変更

# 凡例
legend = ax.legend(loc='upper right', frameon=False, fontsize=12)
plt.setp(legend.get_texts(), color='black')

# 軸範囲設定
global_max = np.max(np.abs(h_plus_matrix))
print(f"Max Strain: {global_max}")
ax.set_ylim(-global_max * 1.1, global_max * 1.1)
ax.set_xlim(r_km[0], r_km[-1])

# グリッド
ax.grid(True, which="major", ls="--", alpha=0.3, color='gray')

# 情報テキスト (黒文字)
info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color='black', va='top', fontsize=12, family='monospace')

def update(frame):
    current_time = t_anim_array[frame]
    
    # 行列から値を取得
    h_p = h_plus_matrix[frame, :]
    h_c = h_cross_matrix[frame, :]
    
    line_plus.set_data(r_km, h_p)
    line_cross.set_data(r_km, h_c)
    
    # 進行状況等の表示
    info_text.set_text(f"Time : {current_time:.4f} s\nFrame: {frame}/{total_frames}")
    
    return line_plus, line_cross, info_text

print(f"アニメーション生成中 (全 {total_frames} フレーム)...")
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS, blit=True)
# ani.save('GW_Propagation_Unified.mp4', writer='ffmpeg', fps=FPS, dpi=150)

plt.show()