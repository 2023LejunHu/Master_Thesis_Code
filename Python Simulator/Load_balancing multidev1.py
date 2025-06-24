import matplotlib.pyplot as plt

# data
device_counts = [3, 5, 7, 9]
initial_vals = [12.4, 10.8, 9.6, 8.3]
final_vals = [9.8, 7.4, 6.1, 5.9]

x = range(len(device_counts))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width/2 for i in x], initial_vals, width=bar_width, label='Initial Tmax')
plt.bar([i + bar_width/2 for i in x], final_vals, width=bar_width, label='Final Tmax')

plt.xticks(x, [str(n) for n in device_counts])
plt.xlabel("Number of Devices")
plt.ylabel("Tmax (ms)")
plt.title("Initial vs Final Tmax by Device Count")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("/mnt/data/tmax_initial_final_comparison.png")
plt.show()
