# Re-execute required logic due to state reset
import matplotlib.pyplot as plt
import numpy as np
import os

# Global constants
TOTAL_CHANNELS = 768
CHANNEL_UNIT = 64
MAX_ITER = 1000
CONVERGENCE_THRESHOLD = 1.0
CHANNEL_MEMORY_MB = 1.0
GAMMA_FLOPS_PER_CHANNEL = 1e6
ALPHA_FWD = 0.609
ALPHA_BWD = 0.409
S = 1e6

# Base bandwidths for each device
base_bandwidths = [100, 150, 80, 250, 120]

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

def create_sample_devices(scaling_factor=1.0):
    scaled_bandwidths = [bw * scaling_factor for bw in base_bandwidths]
    return [
        Device("Device1", 2.0e6, 256, scaled_bandwidths[0], 10),
        Device("Device2", 5.0e6, 128, scaled_bandwidths[1], 10),
        Device("Device3", 1.5e6, 512, scaled_bandwidths[2], 10),
        Device("Device4", 6.0e6, 192, scaled_bandwidths[3], 10),
        Device("Device5", 4.5e6, 384, scaled_bandwidths[4], 10),
    ]

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

def initial_channel_assignment(devices):
    total_perf = sum(d.flops for d in devices)
    max_channels = [int(d.memory_mb // CHANNEL_MEMORY_MB) for d in devices]
    tentative_allocs = []
    for d in devices:
        share_ratio = d.flops / total_perf
        alloc = int(round((TOTAL_CHANNELS * share_ratio) / CHANNEL_UNIT)) * CHANNEL_UNIT
        tentative_allocs.append(alloc)
    overflow = 0
    for i, d in enumerate(devices):
        max_local = int(max_channels[i] // CHANNEL_UNIT) * CHANNEL_UNIT
        if tentative_allocs[i] <= max_local:
            d.local_channels = tentative_allocs[i]
        else:
            d.local_channels = max_local
            overflow += tentative_allocs[i] - max_local
    while overflow > 0:
        candidates = [d for d in devices if d.total_channels + CHANNEL_UNIT <= max_channels[devices.index(d)]]
        if not candidates:
            break
        best = max(candidates, key=lambda d: max_channels[devices.index(d)] - d.total_channels)
        best.local_channels += CHANNEL_UNIT
        overflow -= CHANNEL_UNIT

def heuristic_iteration(devices):
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

    # 修改迁移单位为 1
    transfer_unit = 1

    for dst, _ in candidates:
        if src.local_channels + src.remote_channels < transfer_unit:
            continue

        new_src_time = (src.local_channels + src.remote_channels - transfer_unit) * GAMMA_FLOPS_PER_CHANNEL / src.flops
        projected_remote_total = dst.remote_channels + transfer_unit
        dst_compute_time = projected_remote_total * GAMMA_FLOPS_PER_CHANNEL / dst.flops
        comm_time = compute_comm_time(src, dst, transfer_unit)
        total_remote_time = comm_time + dst_compute_time

        dst_total_future_time = (dst.local_channels + projected_remote_total) * GAMMA_FLOPS_PER_CHANNEL / dst.flops

        if total_remote_time <= new_src_time and dst_total_future_time <= new_src_time:
            src.remote_channels -= transfer_unit
            dst.remote_channels += transfer_unit
            return True

    return False


def simulate_bandwidth_scaled(scaling_factor):
    devices = create_sample_devices(scaling_factor)
    initial_channel_assignment(devices)
    tmax_history = []
    for _ in range(MAX_ITER):
        for d in devices:
            compute_execution_time(d)
        tmax = max(d.execution_time for d in devices)
        tmin = min(d.execution_time for d in devices)
        tmax_history.append(tmax)
        if tmax - tmin < CONVERGENCE_THRESHOLD:
            break
        if not heuristic_iteration(devices):
            break
    return tmax_history[0], tmax_history[-1]

# Run simulation across bandwidth scaling
scales = np.arange(0.1, 1.75, 0.02)
initial_tmax = []
final_tmax = []

for scale in scales:
    tmax_init, tmax_final = simulate_bandwidth_scaled(scale)
    initial_tmax.append(tmax_init)
    final_tmax.append(tmax_final)

plt.rcParams.update({'font.size': 14})  
# print("Initial Tmax:", initial_tmax)
# print("Final Tmax:", final_tmax)
# Plot result
plt.figure(figsize=(10, 6))
plt.plot(scales, initial_tmax, label="Initial Tmax", marker='.')
plt.plot(scales, final_tmax, label="Final Tmax", marker='.')
plt.xlabel("Bandwidth Scaling Factor")
plt.ylabel("Max Execution Time (ms)")
plt.title("Effect of Bandwidth Scaling on Execution Time")
plt.legend()
plt.grid(True)
plt.savefig("./results/bandwidth_scaling_effect.pdf")
plt.tight_layout()
plt.show()




