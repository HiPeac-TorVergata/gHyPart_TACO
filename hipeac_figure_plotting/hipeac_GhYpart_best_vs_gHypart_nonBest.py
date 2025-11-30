import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Caricamento dati
df1 = pd.read_csv("../results/hipeac_ghypart_NVIDIA_GeForce_RTX_4090_t32_2025-09-14.csv")
df2 = pd.read_csv("../results/hipeac_ghypart_all_best_NVIDIA_GeForce_RTX_4090_t32_2025-11-29.csv")

df2["k=2"] = df2["k=2_perf"]
df2["k=3"] = df2["k=3_perf"]
df2["k=4"] = df2["k=4_perf"]

df1["speedup_k=2"] = df2["k=2"]/df1["k=2"]
df1["speedup_k=3"] = df2["k=3"]/df1["k=3"]
df1["speedup_k=4"] = df2["k=4"]/df1["k=4"]


cols = ["speedup_k=2", "speedup_k=3", "speedup_k=4"]

# Lista delle colonne speedup

# Crea i subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)


for i, c in enumerate(cols):
    #app = df1[np.isfinite(df1[c])]
    app = df1
    ax = axes[i]
    y = np.sort(app[c].values)  
    x = np.arange(len(y))
    # Linea principale
    ax.plot(x, y, color='blue', alpha=0.7)
     
    # Linea orizzontale della media
   
    # Conta quanti valori sono maggiori di 1
    count_gt1 = np.sum(y> 1)
    # Disegna linea verticale
    ax.axvline(len(y) - count_gt1, color='green', linestyle=':', 
               label=f"#element Speed up > 1:  {count_gt1}")

    # Titolo e scala logaritmica
    ax.set_title(f"{c}")
    ax.set_yscale('symlog')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("grafico3_normal_best.pdf")
plt.show()
