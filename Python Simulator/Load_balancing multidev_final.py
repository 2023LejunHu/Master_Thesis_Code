import matplotlib.pyplot as plt
import numpy as np


device_counts = [3, 5, 7, 9]

initial_tmax = [298.67, 128.0, 42.67, 32.00]

final_tmax = [94.67, 41.33, 33.33, 30.50]


reduction_percent = [(1 - f/i) * 100 for i, f in zip(initial_tmax, final_tmax)]


bar_width = 0.35
x = np.arange(len(device_counts))

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - bar_width/2, initial_tmax, bar_width, label='Initial Tmax')
bars2 = ax.bar(x + bar_width/2, final_tmax, bar_width, label='Final Tmax')


for i in range(len(device_counts)):
    height = max(initial_tmax[i], final_tmax[i])
    ax.text(x[i], height + 1, f"-{reduction_percent[i]:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')


plt.rcParams.update({'font.size': 14})  
ax.set_xlabel("Number of Devices", fontsize=14)
ax.set_ylabel("Max Execution Time (ms)", fontsize=14)
ax.set_title("Tmax Before and After Optimization", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in device_counts])
ax.legend()
# ax.grid(True, axis='y')
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tmax_initial_final_with_percent.pdf")
plt.show()
