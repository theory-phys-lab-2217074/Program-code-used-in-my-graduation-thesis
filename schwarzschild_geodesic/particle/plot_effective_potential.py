import numpy as np
import matplotlib.pyplot as plt

# -------------------- 有効ポテンシャル --------------------
def Effective_Potential(r, L):
    # 構文エラー修正 + 有効ポテンシャル公式に基づく記述
    U = -(Rg / (2*r)) + (Rg**2 * L**2) / (2 * r**2) - (Rg**3 * L**2) / (2 * r**3)
    return U

# -------------------- 定数定義 --------------------
M = 1.0                 # 質量（任意単位）
Rg = 2 * M              # シュワルツシルト半径
rtol, atol = 1e-9, 1e-9

tau_max = 200           # 固有時刻の最大値
rlim = 15
dr = 0.05

# -------------------- 粒子の設定 --------------------
N = 5
L_max = 3
L = np.array([1, 2, 3, 4, 5 ]) *1e2

# -------------------- r 配列の生成 --------------------
N_r = int(rlim / dr)
r = np.linspace(1e-5 , rlim, N_r) 

# -------------------- プロット --------------------
fig, ax = plt.subplots(figsize=(6, 6))

# 有効ポテンシャルプロット
U_max = 0
for i in range(len(L)) :
    
    U_eff = Effective_Potential(r, L[i])
    ax.plot(r/Rg, U_eff, label=f"L = {L[i]:.1f}")
    U_max = max(U_max,max(U_eff))

#ax.set_xlim(Rg, 15)
ax.set_ylim(-0.1, 2e4)

ax.set_xlabel(r"$r / r_g$")
ax.set_title("Angular momentum dependent effective potential")
ax.grid(True)
ax.legend()  # 必要に応じて

# --- 保存手続き ---
import csv
from datetime import datetime
from pathlib import Path

# ---------- 出力フォルダ作成 ----------
OUTDIR = Path(__file__).resolve().parent / datetime.now().strftime(
         "Effective_Potential_%Y-%m-%d_%H-%M-%S")
OUTDIR.mkdir(exist_ok=True)

# --- 保存 ---
fig.savefig(OUTDIR/"beta1_Angular_momentum_dependent_effective_potential.pdf", dpi=300)
print("保存が完了しました。")

# --- プレビュ ---
plt.show()
