'''
このコードは以下の物理計算パートを担い、可視化・グラフ化は別ファイルに委ねる。
0, 連星系の初期設定、観測点の設定を受け取る
1, 連星系が合体するまでの時間(計算上限時間)を計算する
2, 合体時間までの長半径・離心率・回転角phiの微分方程式を解き、時間に対する関数として出力する
3, 遠方観測位置(r,iota,theta)における原点からの遅延時間(t_ret)を計算する
4, 遅延時間(t_ret)における連星系の長半径・離心率・回転角phiを受け取る
5, 遠方観測位置(r,iota,theta)における重力波の偏光モード(h_plus,h_cross)を計算する
a, 合体時間までの連星系を成す2天体の位置(x1,y1),(x2,y2)を相対ベクトルから計算し、時間に対する関数として出力する
b, 合体時間までのエネルギー・角運動量の微分方程式を解き、時間に対する関数として出力する
c, 観測位置(r,iota,theta)における重力波による潮汐力 doot{h_ij}/2 を計算し、時間に対する関数として出力する
'''
# ==========================================
# モジュール
# ==========================================
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline  #遅延時間に対する配列を得る為の3次補間アルゴリズム

# ==========================================
# 物理定数 (SI単位系)
# ==========================================
G_SI = 6.67430e-11       # [m^3 kg^-1 s^-2]
C_SI = 2.99792458e8      # [m s^-1]
M_SUN_SI = 1.9891e30     # [kg]
PC_SI = 3.0857e16        # [m]
YEAR_SI = 31557600.0     # [s] (365.25日)
"""
SI単位系の設定パラメータを受け取る。
このコード内部ではc=M=Rg=1を計算単位として桁落ち等を防ぐ。
出力時にはSI単位系に数値に戻す
"""

# ==========================================
# 計算クラス
# ==========================================
class BinaryRadiatingGravitationalWave:
    
    # ==========================================
    # 0, 連星系の初期設定、観測点の設定を受け取る
    # ==========================================
    def __init__(self, m1_SI, m2_SI, a0_SI, e0, r_SI, theta, iota):
        """
        params:
          m1_SI, m2_SI : 質量 [kg]
          a0_SI      : 初期軌道長半径 [m]
          e0      : 初期離心率
          r_SI      : 観測距離 [m] (デフォルト値。波形計算時に上書き可能)
          theta  : 公転軸からの天頂角 [rad] (Inclination)
          iota   : 長軸からの方位角 [rad] (Azimuth)
        """
        # ==========================================
        # このコードの計算単位 (c=M=Rg=1)
        # ==========================================
        # 正規化の基準
        M_SI = m1_SI + m2_SI                   # 物理的な総質量 [kg]
        Rg_SI = (2 * G_SI * M_SI) / (C_SI**2)  # 物理的なシュバルツシルト半径 [m]
        
        # 物理単位への変換係数をクラス変数として保持
        self.unit_L = Rg_SI             # 長さの単位 [m]
        self.unit_T = Rg_SI / C_SI      # 時間の単位 [s]
        self.unit_M = M_SI             # 質量の単位 [kg]
        
        # 計算用正規化定数の設定
        self.C = 1.0
        self.M = 1.0
        self.G = 0.5  # c=1, M=1, Rg=1 となるように定義すると、Gは0.5
        
        # ==========================================
        # 連星系の設定 (正規化単位へ変換)
        # ==========================================
        self.m1 = m1_SI / self.unit_M  # 質量は総質量に対する比率になる (例: 0.5, 0.5)
        self.m2 = m2_SI / self.unit_M
        self.a0 = a0_SI / self.unit_L  # 距離は「シュバルツシルト半径の何倍か」に変換
        self.e0 = e0
        self.Rg = 2 * self.G * self.M / self.C**2  # 計算結果は 1.0
        self.R_isco = 3.0 * self.Rg                # 最も内側の安定円軌道(Innter-most Stable Circular Orbit)
        
        ####### 観測点の設定(正規化単位へ変換) #######
        self.r_obs = r_SI / self.unit_L  # 観測距離も「シュバルツシルト半径の何倍か」に変換
        self.theta = theta
        self.iota  = iota
        
        ####### 内部変数の初期化 #######
        self.is_solved = False # 連星系の時間発展計算を行ったらTrueにする
        self.t_archive = None
        self.y_archive = None  # [a, e, Phi]
        self.func_a = None     # IM: Interpolation Method（補間法、内挿法）
        self.func_e = None
        self.func_phi = None
        
        # 合体までの時間を計算 (シミュレーション上限時間決定用)
        self.T_life_SI = self._analytical_lifetime_SI() 
        
    # ==========================================
    # 1, 連星系が合体するまでの時間(計算上限時間)を計算する
    # ==========================================
    def _analytical_lifetime_SI(self):
        """
        連星系が合体するまでの時間 T_BSM を解析解から計算する。
        """
        # 円軌道の場合の寿命 T_circ
        T_circ = (5 * self.C**5 * self.a0**4) / (256 * self.G**3 * self.m1 * self.m2 * self.M)
        if self.e0 < 1e-12: 
            # 物理単位 [s] に変換して返す
            return T_circ * self.unit_T #安全装置

        # 楕円軌道の場合の厳密寿命 T_BSM (Peters Integral)
        factor_num = 48 * (1 - self.e0**2)**4
        factor_den = 19 * (1 + (121/304)*self.e0**2)**(3480/2299)
        prefactor = T_circ * (factor_num / factor_den)
        def integrand(u):
            term1 = u**(29/19)
            term2 = (1 + (121/304) * self.e0**2 * u**2)**(1181/2299)
            term3 = (1 - self.e0**2 * u**2)**(1.5)
            return (term1 * term2) / term3
        
        integral_val, _ = quad(integrand, 0.0, 1.0)
        # 物理単位 [s] に変換して返す
        return prefactor * integral_val * self.unit_T
    
    # ==========================================
    # 2, 合体時間までの長半径・離心率・回転角の微分方程式を解き、時間に対する関数として出力する
    # ==========================================
    # 解くべき微分方程式
    def _Derivatives_for_state_of_binary_star_systems(self, t, y):
        """連星系の長半径・離心率・回転角の時間発展方程式 (ODE)"""
        a, e, phi = y
        # 安全装置: 合体後あるいは異常値
        if a <= 0 or e < 0: return [0.0, 0.0, 0.0] #変化量なしとする
        
        dPhi_dt = np.sqrt(self.G * self.M / (a**3 * (1 - e**2)**3)) * (1 + e * np.cos(phi))**2
        da_dt = -(64 * self.G**3 * self.m1 * self.m2 * self.M) / (5 * self.C**5 * a**3 * (1 - e**2)**(3.5)) * (1 + (73/24)*e**2 + (37/96)*e**4)
        de_dt = -(304 * self.G**3 * self.m1 * self.m2 * self.M) / (15 * self.C**5 * a**4 * (1 - e**2)**(2.5)) * e * (1 + (121/304)*e**2)
        return [da_dt, de_dt, dPhi_dt]
    # 微分方程式を解き、時間に対する関数として出力する
    def Archive_for_state_of_binary_star_systems(self, t_end_SI=None, rtol=1e-10, atol=1e-12):
        """時間発展計算を実行し、補間関数を作成する。"""
         
        # 計算終了時間の設定 (正規化単位へ変換)
        if t_end_SI is None: #指定の計算終了時間が入力されなかった場合は合体時間ちょいまでの計算を行う
            t_end = (self.T_life_SI / self.unit_T) * 1.05 
        else:
            t_end = t_end_SI / self.unit_T
        
        # 計算終了判定を合体直前(最小安定円軌道)に設定する
        def merger_event(t, y): return y[0] - self.R_isco*0.1
        merger_event.terminal = True
        
        # 微分方程式を解く
        y0 = [self.a0, self.e0, 0.0] # 初期条件
        sol = solve_ivp(
            self._Derivatives_for_state_of_binary_star_systems, 
            [0, t_end], y0, method='DOP853', events=merger_event, rtol=rtol, atol=atol )
        
        # 解の取得(正規化単位)
        self.t_archive = sol.t
        self.y_archive = sol.y
        self.is_solved = True  # 連星系の時間発展計算を行ったことを記録する
        
        # 遅延時間における値を渡せるように補間関数を用意(正規化単位)
        self.func_a = CubicSpline(self.t_archive, self.y_archive[0], extrapolate=False)  #3次補間で精度担保
        self.func_e = CubicSpline(self.t_archive, self.y_archive[1], extrapolate=False)
        self.func_phi = CubicSpline(self.t_archive, self.y_archive[2], extrapolate=False)
        
        # 返り値用に物理単位へ変換した配列を作成
        t_archive_SI = self.t_archive * self.unit_T
        y_archive_SI = self.y_archive.copy()
        y_archive_SI[0] *= self.unit_L # 長半径a を [m] に変換 (e, phiは無次元なのでそのまま)
        
        return t_archive_SI, y_archive_SI
        
    # ==========================================
    # 3, 遠方観測位置(r,iota,theta)における原点からの遅延時間(t_ret)を計算する
    # 4, 遅延時間(t_ret)における連星系の長半径・離心率・回転角phiを受け取る
    # 5, 遠方観測位置(r,iota,theta)における重力波の偏光モード(h_plus,h_cross)を計算する
    # ==========================================
    def Calculation_of_waveforms_at_distant_observation_locations(self, t_obs_SI, r_obs_SI=None, theta_inc=None, iota_azi=None):
        """厳密な遅延時刻処理を用いて波形を計算する。"""
        if not self.is_solved:  # 連星系の時間発展計算を既にしていればスルーされる
            t_max_norm = np.max(t_obs_SI) / self.unit_T
            sim_end_SI = max(np.max(t_obs_SI) * 1.1, self.T_life_SI * 1.05) # 結局,安全装置(計算終了判定)があるので長めに設定しておく
            print("連星系の時間発展の計算を行います。")
            self.Archive_for_state_of_binary_star_systems(t_end_SI=sim_end_SI)
            print("連星系の時間発展の計算が終わりました。放出される重力波の計算に移ります。")
        
        # 観測点の設定(正規化単位)
        # 引数がNoneならデフォルト値(正規化済み)を使用、あれば正規化して使用
        t_obs = t_obs_SI / self.unit_T
        r_val = (r_obs_SI / self.unit_L) if r_obs_SI is not None else self.r_obs
        theta_val = theta_inc if theta_inc is not None else self.theta
        iota_val = iota_azi if iota_azi is not None else self.iota
        
        # 1次元以上の配列化 (一瞬または一点に対しても1次元配列として対応させる)
        t_obs = np.atleast_1d(t_obs) 
        r_val = np.atleast_1d(r_val) 
        
        # 遅延時間の定義(正規化単位)
        t_ret = t_obs - r_val / self.C
        
        # 遅延時間(t_ret)における連星系の長半径・離心率・回転角phi
        a_ret = self.func_a(t_ret)
        e_ret = self.func_e(t_ret)
        phi_ret = self.func_phi(t_ret)
        
        # 合体時間を受け取り、描画可能時間(valid_mask)を論理式で定義
        t_merger = self.t_archive[-1]
        valid_mask = (t_ret >= 0) & (t_ret <= t_merger) & (~np.isnan(a_ret))
        
        # --- 異常値の置換 ---
        # 計算不可能な領域(合体後)は、初期値や安全な値を入れて計算エラーを防ぐ
        # (最終的にマスクで0にするので値は何でも良いが、NaN回避のため)
        a_safe = np.where(valid_mask, a_ret, self.a0)
        e_safe = np.where(valid_mask, e_ret, self.e0)
        phi_safe = np.where(valid_mask, phi_ret, 0.0)
        
        # 振幅係数・振動関数
        A_val = (self.m1 * self.m2 * self.G**2) / (self.C**4 * a_safe * (1 - e_safe**2))
        exp_i_phi  = np.exp(1j * phi_safe)
        f_val = 4*e_safe**2 + 2*e_safe*(exp_i_phi**3) + 8*(exp_i_phi**2) + 10*e_safe*(exp_i_phi)
        # 観測者方位角依存項
        term_angle = np.exp(-1j * 2 * iota_val)
        
        # --- h+, hx の計算 ---
        h_plus = - ((1 + np.cos(theta_val)**2) / 2) * (A_val / r_val) * np.real(f_val * term_angle) \
                 - 2 * np.sin(theta_val)**2 * (A_val / r_val) * (e_safe**2 + e_safe * np.cos(phi_safe))
        h_cross = - np.cos(theta_val) * (A_val / r_val) * np.imag(f_val * term_angle)
        
        # 描画可能時間(valid_mask)以外では振幅0とする
        h_plus[~valid_mask] = 0.0
        h_cross[~valid_mask] = 0.0
        return h_plus, h_cross  # 無次元量
    

# ==========================================
# 【追加】固定離心率で目標寿命から初期長半径を逆算する
# ==========================================
def find_a0_for_lifetime(m1_SI, m2_SI, e0, T_target_SI, a_min_Rg=1.0, a_max_Rg=1e8, n_samples=500):
    """
    目標寿命 T_target_SI [s] を実現する初期長半径 a0 [m] を算出する。
    """
    # 正規化基準の算出
    M_total = m1_SI + m2_SI
    Rg_SI = (2 * G_SI * M_total) / (C_SI**2)
    
    # サンプル生成 (対数スケール)
    a_samples_SI = np.logspace(np.log10(a_min_Rg * Rg_SI), np.log10(a_max_Rg * Rg_SI), n_samples)
    T_samples_SI = []
    
    for a_val in a_samples_SI:
        # 寿命計算用のテンポラリインスタンス (観測パラメータはダミー)
        tmp_sys = BinaryRadiatingGravitationalWave(m1_SI, m2_SI, a_val, e0, r_SI=1e10, theta=0.0, iota=0.0)
        T_samples_SI.append(tmp_sys.T_life_SI)
        
    T_samples_SI = np.array(T_samples_SI)
    
    # 対数空間での補間
    log_T = np.log10(T_samples_SI)
    log_a = np.log10(a_samples_SI)
    interp_log_a = CubicSpline(log_T, log_a, extrapolate=True)
    
    a0_res_SI = 10**interp_log_a(np.log10(T_target_SI))
    return a0_res_SI