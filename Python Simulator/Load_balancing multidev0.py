import matplotlib.pyplot as plt
import csv
import os


TOTAL_CHANNELS = 768
CHANNEL_UNIT = 64
MAX_ITER = 1000
CONVERGENCE_THRESHOLD = 1.0
CHANNEL_MEMORY_MB = 1.0
GAMMA_FLOPS_PER_CHANNEL = 1e6
ALPHA_FWD = 0.609
ALPHA_BWD = 0.409
S = 1e6

class Device:
    def __init__(self, name, flops, memory_mb, bandwidth_mb_per_s, latency_ms):
        self.name = name
        self.flops = flops
        self.memory_mb = memory_mb
        self.bandwidth = bandwidth_mb_per_s
        self.latency = latency_ms
        self.local_channels = 0
        self.remote_channels = 0
        self.execution_time = 0.0

    @property
    def total_channels(self):
        return self.local_channels + self.remote_channels

def create_devices_with_count(num):
    presets = [
        ("A", 2.0e6, 256, 100, 10),
        ("B", 5.0e6, 128, 150, 10),
        ("C", 1.5e6, 512, 80, 10),
        ("D", 6.0e6, 192, 250, 10),
        ("E", 4.5e6, 384, 120, 10),
        ("F", 3.5e6, 320, 90, 10),
        ("G", 2.5e6, 400, 85, 10),
        ("H", 7.0e6, 256, 200, 10),
        ("I", 6.5e6, 300, 170, 10),
    ]
    return [Device(f"Device{name}", fl, mem, bw, lat) for name, fl, mem, bw, lat in presets[:num]]

def compute_execution_time(device):
    device.execution_time = device.total_channels * GAMMA_FLOPS_PER_CHANNEL / device.flops
    return device.execution_time

def compute_comm_time(src, dst, channels):
    if channels == 0:
        return 0.0
    min_bandwidth = min(src.bandwidth, dst.bandwidth)
    fwd = (ALPHA_FWD * S * channels) / (min_bandwidth * 1e6) * 1000
    bwd = (ALPHA_BWD * S * channels) / (min_bandwidth * 1e6) * 1000
    return src.latency + dst.latency + fwd + bwd

def initial_channel_assignment(devices, total_channels, unit, channel_mem_mb):
    total_perf = sum(d.flops for d in devices)
    max_channels = [int(d.memory_mb // channel_mem_mb) for d in devices]

    tentative_allocs = []
    for d in devices:
        share_ratio = d.flops / total_perf
        alloc = int(round((total_channels * share_ratio) / unit)) * unit
        tentative_allocs.append(alloc)

    overflow = 0
    for i, d in enumerate(devices):
        max_local = int(max_channels[i] // unit) * unit
        if tentative_allocs[i] <= max_local:
            d.local_channels = tentative_allocs[i]
        else:
            d.local_channels = max_local
            overflow += tentative_allocs[i] - max_local

    while overflow > 0:
        candidates = [d for d in devices if d.total_channels + unit <= max_channels[devices.index(d)]]
        if not candidates:
            break
        best = max(candidates, key=lambda d: max_channels[devices.index(d)] - d.total_channels)
        best.local_channels += unit
        overflow -= unit

def heuristic_iteration(devices, unit):
    for d in devices:
        compute_execution_time(d)
    src = max(devices, key=lambda d: d.execution_time)
    Tmax_before = src.execution_time

    candidates = []
    for dst in devices:
        if dst == src:
            continue
        compute_execution_time(dst)
        slack = Tmax_before - dst.execution_time
        if slack > 0:
            candidates.append((dst, slack))
    candidates.sort(key=lambda x: x[1], reverse=True)

    for dst, _ in candidates:
        new_src_time = (src.local_channels + src.remote_channels - unit) * GAMMA_FLOPS_PER_CHANNEL / src.flops
        projected_remote_total = dst.remote_channels + unit
        dst_compute_time = projected_remote_total * GAMMA_FLOPS_PER_CHANNEL / dst.flops
        comm_time = compute_comm_time(src, dst, unit)
        dst_total_future_time = (dst.local_channels + projected_remote_total) * GAMMA_FLOPS_PER_CHANNEL / dst.flops

        if (comm_time + dst_compute_time <= new_src_time) and (dst_total_future_time <= new_src_time):
            src.remote_channels -= unit
            dst.remote_channels += unit
            return True
    return False

def simulate(devices):
    initial_channel_assignment(devices, TOTAL_CHANNELS, CHANNEL_UNIT, CHANNEL_MEMORY_MB)
    tmax_history = []

    for _ in range(MAX_ITER):
        for d in devices:
            compute_execution_time(d)
        tmax = max(d.execution_time for d in devices)
        tmin = min(d.execution_time for d in devices)
        tmax_history.append(tmax)
        if tmax - tmin < CONVERGENCE_THRESHOLD:
            break
        if not heuristic_iteration(devices, 1):
            break
    return tmax_history

def run_all_experiments():
    device_counts = [3, 5, 7, 9]
    initial_tmax_values = []
    final_tmax_values = []

    for n in device_counts:
        print(f"\n=== Running for {n} devices ===")
        devices = create_devices_with_count(n)
        tmax_history = simulate(devices)
        initial_tmax_values.append(tmax_history[0])
        final_tmax_values.append(tmax_history[-1])
        print(f"Initial Tmax: {tmax_history[0]:.2f}, Final Tmax: {tmax_history[-1]:.2f}")

    return device_counts, initial_tmax_values, final_tmax_values

def plot_bar_chart(device_counts, initial, final):
    reductions = [(i - f) / i * 100 for i, f in zip(initial, final)]
    plt.figure(figsize=(8, 6))
    bars = plt.bar([str(n) for n in device_counts], reductions)
    plt.ylim(0, 100)
    plt.ylabel("Reduction in Tmax (%)")
    plt.xlabel("Number of Devices")
    plt.title("Tmax Reduction vs Device Count")
    for bar, val in zip(bars, reductions):
        plt.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%", ha='center')
    os.makedirs("./results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("./results/tmax_reduction_vs_device_count.png")
    plt.show()

if __name__ == "__main__":
    device_counts, initial_vals, final_vals = run_all_experiments()
    plot_bar_chart(device_counts, initial_vals, final_vals)
