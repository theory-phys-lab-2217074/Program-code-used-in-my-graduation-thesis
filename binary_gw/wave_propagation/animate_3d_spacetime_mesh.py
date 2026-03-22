import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as colors
from matplotlib import cm
import time
import Calculating_gravitational_waves_from_binary_star_systems as gw_phys

# ==========================================
# 1. パラメータ設定
# ==========================================
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN_SI = 1.9891e30

# --- 連星系パラメータ ---
m1_SI = 30 * M_SUN_SI
m2_SI = 30 * M_SUN_SI
M_total = m1_SI + m2_SI
Rg_SI = (2 * G_SI * M_total) / (C_SI**2)
e0 = 0.5

# --- 空間・格子密度調整 ---
# GRID_RES: 網目の細かさ。80-100が「網」として見た目と計算負荷のバランスが最適
GRID_RES = 50    
GRID_SIZE = 1000.0 * Rg_SI 
DECAY_P = 0.01     # 視覚的減衰(非物理的な表現的処置)

# --- 動画構成の設定 ---
T_TARGET_VIDEO_SEC = 30.0 
FPS = 40
TOTAL_FRAMES = int(T_TARGET_VIDEO_SEC * FPS)

# --- 初期長半径 a0 の逆算 ---
t_reach = 1.             
t_prop_z = GRID_SIZE / C_SI  
t_merger_target = t_prop_z * (T_TARGET_VIDEO_SEC / t_reach - 1) 
a0_SI = gw_phys.find_a0_for_lifetime(m1_SI, m2_SI, e0, t_merger_target)

# --- 初期段階の波数計算 ---
# 1. 公転周期と基本周波数の計算
# 楕円軌道でも基本となる重力波周波数は公転周波数の2倍 (f_gw = 2 * f_orb)
f_orb_0 = (1.0 / (2.0 * np.pi)) * np.sqrt(G_SI * (m1_SI + m2_SI) / a0_SI**3)
f_gw_0  = 2.0 * f_orb_0

# 2. t_reach (動画上の2秒) に対応する物理時間 t_prop_z の間に放出される波数
N0_wave = t_prop_z * f_gw_0 # N = (物理的な伝搬時間) * (重力波の周波数)

# ==========================================
# 2. 物理エンジンの初期化
# ==========================================
binary_system = gw_phys.BinaryRadiatingGravitationalWave(
    m1_SI=m1_SI, m2_SI=m2_SI, a0_SI=a0_SI, e0=e0, 
    r_SI=100*Rg_SI, theta=0.0, iota=0.0
)
t_phys_data, _ = binary_system.Archive_for_state_of_binary_star_systems()
t_merger_SI = t_phys_data[-1]
t_anim_phys = np.linspace(0, t_merger_SI+ 1.5*GRID_SIZE/C_SI, TOTAL_FRAMES)

# グリッド生成
x = np.linspace(-GRID_SIZE, GRID_SIZE, GRID_RES)
y = np.linspace(-GRID_SIZE, GRID_SIZE, GRID_RES)
X, Y = np.meshgrid(x, y)
R_grid = np.sqrt(X**2 + Y**2)
Phi_grid = np.arctan2(Y, X)

# ==========================================
# 3. 描画セットアップ (3D・透明メッシュ)
# ==========================================
fig = plt.figure(figsize=(16, 9), dpi=100)
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=-1.95, right=2.85, bottom=-2, top=2.9)

ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.grid(False)

# 最新のMatplotlib APIに準拠した軸の透明化
ax.xaxis.set_pane_color((0, 0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0, 0))
ax.zaxis.set_pane_color((0, 0, 0, 0))
ax.axis('off')

# 表示上の振幅倍率
A_ref_at_edge = (G_SI * M_total) / (C_SI**2 * GRID_SIZE)
Z_MAX_VISUAL = GRID_SIZE * 0.00005
Z_SCALE = Z_MAX_VISUAL / A_ref_at_edge

surf = None
star1, = ax.plot([], [], [], 'o', color='white', markersize=5, zorder=100)
star2, = ax.plot([], [], [], 'o', color='white', markersize=5, zorder=100)

cmap = cm.RdBu_r

start_time = time.time()
def update(frame_idx):
    global surf
    t_now = t_anim_phys[frame_idx]
    
    if surf is not None:
        surf.remove()
    
    # 物理フィールド計算
    h_p, _ = binary_system.Calculation_of_waveforms_at_distant_observation_locations(
        t_obs_SI=np.array([t_now]), r_obs_SI=R_grid.ravel(), theta_inc=0.0, iota_azi=Phi_grid.ravel()
    )
    field_raw = h_p.reshape(GRID_RES, GRID_RES)
    
    # 減衰率スケーリングとマスク処理
    field_scaled = field_raw * (R_grid / GRID_SIZE)**(1.0 - DECAY_P)
    mask_radius = 30.0 * Rg_SI
    field_scaled[R_grid < mask_radius] = 0
    
    # RGBAマッピング (透明度の動的制御)
    # COLOR_GAIN: 色彩の飽和速度。1より小さいほど色がはっきりつく
    COLOR_GAIN = 0.5 
    # ALPHA_EXP: 透明度の立ち上がり。小さいほど波が不透明に見える
    ALPHA_EXP = 0.4
    norm_field = np.clip(field_scaled / (A_ref_at_edge * COLOR_GAIN), -1, 1)
    color_dimension = (norm_field + 1) / 2.0
    face_colors = cmap(color_dimension)
    
    # 透明度を振幅の絶対値に連動 (0.7乗で中間調の透過を美しく)
    face_colors[:, :, 3] = np.abs(norm_field) ** ALPHA_EXP
    
    # 3Dサーフェス描画: linewidthを細くして「網」の繊細さを出す
    # rstride=1, cstride=1 を指定することで、GRID_RES=100 の密度が100%反映されます。
    surf = ax.plot_surface(X, Y, field_scaled * Z_SCALE, 
                           facecolors=face_colors,
                           edgecolor=(1, 1, 1, 0.6), # 白色の線を 40% の透過度で設定
                           linewidth=0.15, 
                           shade=False, 
                           antialiased=True,
                           rstride=1, 
                           cstride=1)
    
    ax.set_zlim(-Z_MAX_VISUAL * 2, Z_MAX_VISUAL * 2)
    
    # 天体位置
    t_norm = min(t_now / binary_system.unit_T, binary_system.t_archive[-1])
    a = binary_system.func_a(t_norm) * binary_system.unit_L
    e = binary_system.func_e(t_norm)
    phi = binary_system.func_phi(t_norm)
    r_sep = a * (1 - e**2) / (1 + e * np.cos(phi))
    
    if t_now < t_merger_SI:
        x1, y1 = 0.5 * r_sep * np.cos(phi), 0.5 * r_sep * np.sin(phi)
        star1.set_data([x1], [y1]); star1.set_3d_properties([0])
        star2.set_data([-x1], [-y1]); star2.set_3d_properties([0])
    else:
        star1.set_data([], []); star2.set_data([], [])

    if (frame_idx % 50 == 0 and frame_idx > 0) or frame_idx==1:
        elapsed = time.time() - start_time
        speed = elapsed / frame_idx
        print(f"現在処理済のフレーム {frame_idx}/{TOTAL_FRAMES} | 残り時間: {speed*(TOTAL_FRAMES-frame_idx)/60:.1f} 分 ( 処理スピード:{speed/60:.2f}[枚/分] )")

    # カメラのゆっくりとした回転
    ax.view_init(elev=30, azim=-45)# + frame_idx * 0.15)
    return surf, star1, star2

# ==========================================
# 4. レンダリング・保存
# ==========================================

print("\n" + "="*60)
print("連星系・重力場の情報")
print(f"離心率{e0}, 初期長半径 a0: {a0_SI/Rg_SI:.4f} [Rg], 寿命(解析解):{t_merger_SI:.4f} s")
#print(f"初期段階での重力波の基本周波数: {f_gw_0:.4f} [Hz]")
print(f"初期 {t_reach} 秒間（画面端到達まで）に放射される重力波の数: {N0_wave:.2f} 本")
print(f"動画上の{t_reach}s = 物理時間{t_prop_z:.4f}s")
print("\n" + "="*60)
print(f"作成動画の情報")
print(f"目標 動画時間 :{T_TARGET_VIDEO_SEC} s")
print(f"総フレーム数: {TOTAL_FRAMES}")
print("="*60 + "\n")
writer = FFMpegWriter(fps=FPS, bitrate=12000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)

#anim.save(f'SpacetimeMesh_Final_Adjusted_{a0_SI:.5f}.mp4', writer=writer)
plt.show()