## 連星系から放射される重力波により遠方の観測者の固定座標系が変化する様子を示すアニメーション ##
## Animation of the cartesian coordinate system of a distant observer changing due to gravitational waves emitted from a binary star system ##

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

# 物理計算モジュールのインポート
# 同ディレクトリに 'Calculating_gravitational_waves_from_binary_star_systems.py' を置く
try:
    from Calculating_gravitational_waves_from_binary_star_systems import BinaryRadiatingGravitationalWave
except ImportError:
    print("Error: 物理計算モジュールが見つかりません。ファイル名を確認してください。")
    exit()

# ==========================================
# 物理定数 (SI単位系)
# ==========================================
G_SI = 6.67430e-11       # [m^3 kg^-1 s^-2]
C_SI = 2.99792458e8      # [m s^-1]
M_SUN_SI = 1.9891e30     # [kg]
PC_SI = 3.0857e16        # [m]
YEAR_SI = 31557600.0     # [s] (365.25日)

# ==========================================
# 1. パラメータ設定
# ==========================================
# --- 連星系パラメータ ---
m1_SI     = 30 * M_SUN_SI   # 星1の質量[kg]
m2_SI     = 30 * M_SUN_SI   # 星2の質量[kg]
Rg_SI = (2 * G_SI * (m1_SI+m2_SI)) / (C_SI**2)
a0_SI = 30*Rg_SI   # 初期長半径[m]
e0        = 0.3    # 初期離心率

# --- 観測パラメータ ---
r_obs_SI  = 100 * 1e6 * PC_SI  # 100Mpc
theta_inc = np.pi/2            
iota_azi  = 0.0            

# --- 可視化・アニメーション設定 ---
VISUAL_SCALE = 0.3         # 可視化時の最大変形率
T_anime   = 5.0           # 作成する動画の長さ [秒]
FPS       = 50             # フレームレート
GRID_SIZE = 10             # 表示空間のサイズ
NUM_POINTS = 16            # グリッドの分割数

# ==========================================
# 2. 物理計算の実行
# ==========================================
print("物理計算を開始します")

# 連星系インスタンスの作成
binary_system = BinaryRadiatingGravitationalWave(
    m1_SI, m2_SI, a0_SI, e0, r_obs_SI, theta_inc, iota_azi
)

print("連星系から放射される重力波の計算を外部ファイルに依頼中...")
t_data, state_data = binary_system.Archive_for_state_of_binary_star_systems()
t_merger_SI = t_data[-1]  # 数値計算上の合体時刻 (正規化単位 -> 物理単位)

# データ取得 
raw_a = state_data[0]   # [m]
raw_e = state_data[1]   # [dimensionless]
raw_phi = state_data[2] # 今回は使わない
raw_t = t_data          # [s]

# 時間軸作成 (物理時間: 0 -> T_life)
# ※アニメーションの最後が合体直前になるように調整
# ※アニメーションの最後が合体直前/16になるように調整
T_orb = np.sqrt( 4*np.pi**2 *a0_SI**3/(G_SI*(m1_SI+m2_SI)) )
total_frames = int(T_anime * FPS)
t_physics_array_SI = np.linspace(0, T_orb*1.2, total_frames)

# 桁落ち回避として近距離で波形を計算
r_calc_SI = binary_system.unit_L * 100 
t_obs_calc = t_physics_array_SI + (r_calc_SI / C_SI)

print("波形計算中")
h_plus_calc, h_cross_calc = binary_system.Calculation_of_waveforms_at_distant_observation_locations(
    t_obs_calc, r_obs_SI=r_calc_SI, theta_inc=theta_inc, iota_azi=iota_azi
)

# 計算上(r_calc_SI)の最大歪みを取得
max_strain_calc = np.max(np.sqrt(h_plus_calc**2 + h_cross_calc**2))

# 実際の観測点(r_obs_SI)の最大歪み（表示用）: 距離に反比例して減衰
correction_factor = r_calc_SI / r_obs_SI
max_strain_true = max_strain_calc * correction_factor

# 可視化スケール用の倍率
vis_factor = 1.0*1e20 #VISUAL_SCALE / max_strain_true

# 3. 可視化用データの作成（直接 h_calc から正規化可能）
if max_strain_calc != 0:
    vis_norm_factor = VISUAL_SCALE / max_strain_calc
else:
    vis_norm_factor = 0

h_plus_visual = h_plus_calc * vis_norm_factor
h_cross_visual = h_cross_calc * vis_norm_factor

print("計算完了。")
print(f"合体時刻(t_merger) = {t_merger_SI:.4f} s")
print(f"最大歪み(物理値): {max_strain_true:.2e} -> 可視化スケール倍率: {vis_factor:.2e}")

# ==========================================
# 3. 可視化 (Visualization Setup)
# ==========================================
print("得られた計算結果を可視化用に処理中")

# 連星系の時間変化情報を取得
a_anim = np.interp(t_physics_array_SI, raw_t, raw_a) 
e_anim = np.interp(t_physics_array_SI, raw_t, raw_e)

# 格子点配列定義
X_grid, Y_grid = np.meshgrid(
    np.linspace(-GRID_SIZE, GRID_SIZE, NUM_POINTS),
    np.linspace(-GRID_SIZE, GRID_SIZE, NUM_POINTS)
)

# 軸定義
AXIS_LEN = GRID_SIZE * 1.1
axis_x_origin = np.array([-AXIS_LEN, 0.0])
axis_x_end_init = np.array([AXIS_LEN, 0.0])
axis_y_origin = np.array([0.0, -AXIS_LEN])
axis_y_end_init = np.array([0.0, AXIS_LEN])

# 図の準備
print("グラフ作成中")
fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
ax.set_aspect('equal')
LIMIT = GRID_SIZE * 1.5
ax.set_xlim(-LIMIT, LIMIT)
ax.set_ylim(-LIMIT, LIMIT)
ax.axis('off')

# 原点
ax.plot(0, 0, marker='o', markersize=25, color='black', zorder=10) # 中心点
ax.plot(0, 0, marker='o', markersize=20, color='white', zorder=20) # 中心点
ax.plot(0, 0, marker='o', markersize=10, color='black', zorder=30) # 中心点

# 座標軸オブジェクト
lines_x = [ax.plot([], [], color='#cccccc', lw=1.0)[0] for _ in range(NUM_POINTS)]
lines_y = [ax.plot([], [], color='#cccccc', lw=1.0)[0] for _ in range(NUM_POINTS)]

ax_x_line, = ax.plot([], [], color='black', lw=2.5, zorder=15)
ax_x_head, = ax.plot([], [], marker='>', markersize=14, color='black', linestyle='None', zorder=15)
label_x = ax.text(0, 0, "x", fontsize=18, fontweight='bold', ha='left', va='center')

ax_y_line, = ax.plot([], [], color='black', lw=2.5, zorder=15)
ax_y_head, = ax.plot([], [], marker='>', markersize=14, color='black', linestyle='None', zorder=15)
label_y = ax.text(0, 0, "y", fontsize=18, fontweight='bold', ha='center', va='bottom')

# 情報テキスト (初期値)
'''
time_text = ax.text(-LIMIT*0.9, LIMIT*0.9, "", fontsize=14, fontweight='bold', va='top')
param_text = ax.text(-LIMIT*0.9, -LIMIT*0.9, 
                     f"$r={r_obs_SI/Rg_SI:.3g} Rg$\n$\\theta={theta_inc/np.pi:.3f} \pi$\n$\\iota={iota_azi/np.pi:.3f} \pi$", 
                     fontsize=12, va='bottom')
# 注釈
note_text = ax.text(0, -LIMIT*0.95, 
    f"Note: Strain amplitude is exaggerated by factor {vis_factor:.1e} for visualization.", 
    fontsize=10, ha='center', color='gray')
'''

# ==========================================
# 4. アニメーション
# ==========================================
def transform_coords(x, y, hp, hx):
    """ TTゲージにおける座標変換 """
    x_new = (1 + 0.5 * hp) * x + (0.5 * hx) * y
    y_new = (0.5 * hx) * x + (1 - 0.5 * hp) * y
    return x_new, y_new

def update(frame):
    # 現在のフレームに対応する歪みを取得
    hp = h_plus_visual[frame]
    hx = h_cross_visual[frame]
    
    # 合体進行度 [%]
    progress_percent = (t_physics_array_SI[frame] / t_merger_SI) * 100
    
    # 情報表示テキストの更新
    info = (
        f"Merger Progress: {progress_percent:5.1f} % (Observer's View)\n"
        f"a : {a_anim[frame]/Rg_SI:.1f} Rg\n"
        f"e : {e_anim[frame]:.4f}"
    )
    #time_text.set_text(info)
    
    # --- 1. グリッドの更新 ---
    for i in range(NUM_POINTS):
        lx_x, lx_y = transform_coords(X_grid[i, :], Y_grid[i, :], hp, hx)
        lines_x[i].set_data(lx_x, lx_y)
        ly_x, ly_y = transform_coords(X_grid[:, i], Y_grid[:, i], hp, hx)
        lines_y[i].set_data(ly_x, ly_y)
    
    # --- 2. 座標軸と矢印の更新 ---
    def update_axis(ax_line, ax_head, ax_label, start_pos, end_pos_init):
        p_start = transform_coords(start_pos[0], start_pos[1], hp, hx)
        p_end   = transform_coords(end_pos_init[0], end_pos_init[1], hp, hx)
        
        ax_line.set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])
        ax_head.set_data([p_end[0]], [p_end[1]])
        
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        rot_deg = np.degrees(np.arctan2(dy, dx))
        
        trans = Affine2D().rotate_deg(rot_deg)
        m = MarkerStyle('>')
        m._transform = m.get_transform() + trans
        ax_head.set_marker(m)
        
        offset_len = 0.5
        offset_x = offset_len * np.cos(np.radians(rot_deg))
        offset_y = offset_len * np.sin(np.radians(rot_deg))
        ax_label.set_position((p_end[0] + offset_x, p_end[1] + offset_y))

    update_axis(ax_x_line, ax_x_head, label_x, axis_x_origin, axis_x_end_init)
    update_axis(ax_y_line, ax_y_head, label_y, axis_y_origin, axis_y_end_init)

    return lines_x + lines_y + [ax_x_line, ax_x_head, label_x, ax_y_line, ax_y_head, label_y, ]  #time_text]

# ==========================================
# 5. アニメーション作成と保存
# ==========================================
# テストで前半300だけ
print(f"アニメーション作成中 (全 500 フレーム)...")

ani = animation.FuncAnimation(
    fig, update, frames=range(total_frames), interval=1000/FPS, blit=False)

plt.tight_layout()

# 保存する場合
#ani.save(f"公転面_Animation_cartesian_coordinate_e{e0:.2f}_vis_factor{vis_factor:.2e}.mp4", writer="ffmpeg", fps=FPS, dpi=150)

plt.show()
print("完了")