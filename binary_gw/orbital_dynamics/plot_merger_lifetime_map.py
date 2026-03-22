import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
# ==========================================
# 論文用フォント設定 (Mac OS)
# ==========================================
plt.rcParams['font.family'] = 'Hiragino Sans'      # 日本語: ヒラギノ角ゴ
plt.rcParams['mathtext.fontset'] = 'cm'            # 数式: TeXフォント
plt.rcParams['axes.unicode_minus'] = False         # マイナス記号対策
plt.rcParams['font.size'] = 25                     # 基本フォントサイズ
plt.rcParams['pdf.fonttype'] = 42

# 物理計算モジュールをインポート
import Calculating_gravitational_waves_from_binary_star_systems as phys

def visualize_binary_lifetime_swapped():
    # ==========================================
    # 1. パラメータ設定
    # ==========================================
    # 連星中性子星をモデルに (36 M_sun + 29 M_sun)
    M1_SUN = 3.
    M2_SUN = 3.5
    
    m1_si = M1_SUN * phys.M_SUN_SI
    m2_si = M2_SUN * phys.M_SUN_SI
    
    # 連星系の総質量とシュバルツシルト半径(Rg)
    M_total = m1_si + m2_si
    Rg_si = (2 * phys.G_SI * M_total) / (phys.C_SI**2)
    
    # --- グリッドの設定 ---
    resolution = 200
    
    # 縦軸：離心率 e の範囲 (線形スケール)
    # 0.0 ~ 0.95
    e_list = np.linspace(0.0, 0.98, resolution)
    
    # 横軸：長半径 a の範囲 (Rg単位) - 対数スケール
    # 3.1 Rg (ISCO直上) ~ 50,000 Rg (宇宙年齢超)
    a_rg_min = 1 
    a_rg_max = 1e6
    a_rg_list = np.logspace(np.log10(a_rg_min), np.log10(a_rg_max), resolution)

    # メッシュグリッド作成
    E, A_Rg = np.meshgrid(e_list, a_rg_list)
    
    # 計算結果(寿命)を格納する配列
    Lifetime_years = np.zeros_like(E)
    
    print(f"計算開始: {resolution}x{resolution} グリッド (X:Semi-major Axis, Y:Eccentricity)")
    print(f"質量設定: {M1_SUN}M☉ + {M2_SUN}M☉ (Rg = {Rg_si/1000:.3f} km)")
    
    # ==========================================
    # 2. 計算ループ
    # ==========================================
    start_time = time.time()
    
    for i in range(resolution):
        for j in range(resolution):
            # メッシュグリッドの値を取得
            a_rg_val = A_Rg[i, j]
            e_val = E[i, j]
            
            a_si_val = a_rg_val * Rg_si
            
            # 物理クラスの呼び出し
            binary = phys.BinaryRadiatingGravitationalWave(
                m1_SI=m1_si, 
                m2_SI=m2_si, 
                a0_SI=a_si_val, 
                e0=e_val, 
                r_SI=1.0, theta=0.0, iota=0.0
            )
            
            Lifetime_years[i, j] = binary.T_life_SI / phys.YEAR_SI

    elapsed_time = time.time() - start_time
    print(f"計算完了: {elapsed_time:.2f}秒")

    # ==========================================
    # 3. 可視化
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # pcolormesh (X, Y, C) の順序で指定
    # X軸: A_Rg (長半径), Y軸: E (離心率)
    mesh = ax.pcolormesh(A_Rg, E, Lifetime_years, 
                         norm=LogNorm(vmin=1e-11, vmax=1e11), 
                         cmap='viridis', shading='auto', alpha=0.9, rasterized=True)
    
    # 横軸を対数スケールに設定
    ax.set_xscale('log')
    
    # カラーバー# 目盛りの設定: 10^-10 から 10^10 まで 10^2 刻み
    cbar_ticks = 10.0 ** np.arange(-10, 11, 2)
    cbar = fig.colorbar(mesh, ax=ax, ticks=cbar_ticks)
    cbar.set_label('合体までの時間[Year]', fontsize=30) # ラベルの文字サイズ設定
    cbar.ax.tick_params(labelsize=30)  # 目盛りの数字のサイズも変えたい場合
    
    # --- 等高線の追加 ---
    #levels = [1.0/365.25/24/60/60, 1.0/365.25/24/60, 1.0/365.25/24, 1.0, 1e3, 1e6, 1e9, 1.38e10] 
    #level_labels = ['1sec', '1min', '1 Hour', '1 Year', '1 kyr', '1 Myr', '1 Gyr', 'Age of Univ.']
    levels = [1.0/365.25/24/60/60, 1.0/365.25/24, 1.0, 1e3, 1e6, 1.38e10] 
    level_labels = [' 1 Second ',' 1 Hour ', ' 1 Year ', ' 1 kyr ', ' 1 Myr ',  ' Age of Univ. ']
    
    # データ範囲内の等高線のみ描画
    min_l, max_l = np.min(Lifetime_years), np.max(Lifetime_years)
    valid_idxs = [i for i, l in enumerate(levels) if min_l < l < max_l]
    valid_levels = [levels[i] for i in valid_idxs]
    valid_labels = [level_labels[i] for i in valid_idxs]
    
    if valid_levels:
        # contour も (X, Y, Z) の順序で指定
        CS = ax.contour(A_Rg, E, Lifetime_years, levels=valid_levels, 
                        colors='black', linewidths=4, linestyles='-')
        
        # === 中心位置の計算処理 ===
        label_positions = []
        for segs in CS.allsegs:
            # そのレベルの線分が存在しない場合はスキップ
            if not segs:
                continue
            
            # 線分が複数ある（途切れている）場合、最も長いものを選ぶ
            longest_seg = max(segs, key=lambda x: len(x))
            
            x_coords = longest_seg[:, 0]
            y_coords = longest_seg[:, 1]
            
            # 距離計算（X軸は対数スケールなので log10 をとって距離を測る）
            x_log = np.log10(x_coords)
            
            # 各点間の距離を計算
            dists = np.sqrt(np.diff(x_log)**2 + np.diff(y_coords)**2)
            cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
            total_dist = cum_dist[-1]
            
            # 全長の半分の地点を探す
            mid_dist = total_dist / 2.0
            idx = np.abs(cum_dist - mid_dist).argmin()
            
            # そのインデックスの座標を取得
            label_positions.append((x_coords[idx], y_coords[idx]))
        # ==========================
        
        fmt = {l: label for l, label in zip(valid_levels, valid_labels)}
        # manual 引数に計算した位置リストを渡す
        clabels = ax.clabel(CS, inline=True, fontsize=30, fmt=fmt, 
                             manual=label_positions)
        for txt in clabels:
            txt.set_fontweight('bold')

    # 必要ならタイトル
    #ax.set_title(f'重力波放射による連星系の合体までの時間: {M1_SUN}$M_\odot$ + {M2_SUN}$M_\odot$', fontsize=14)
    # 軸ラベルの設定
    ax.set_xlabel(f'初期長半径$a_0 / R_g$ ($R_g = {Rg_si/1000:.1f}\cdots$km)', weight='bold', fontsize=35)
    ax.set_ylabel('初期離心率$e_0$', weight='bold', fontsize=35)
    
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    ax.grid(True, which='major', linestyle='-', alpha=0.8, color='white')
    ax.grid(True, which='minor', linestyle=':', alpha=0.3, color='white')

    plt.tight_layout()
    #fig.savefig(rf"重力波放射連星系の寿命カラーマップ_{M1_SUN}M+{M2_SUN}M.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_binary_lifetime_swapped()