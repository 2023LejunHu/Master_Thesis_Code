import matplotlib.pyplot as plt
import csv
import os
# Global parameters
NUM_DEVICES = 5
TOTAL_CHANNELS = 768
CHANNEL_UNIT = 64
MAX_ITER = 1000
CONVERGENCE_THRESHOLD = 1.0
CHANNEL_MEMORY_MB = 1.0  # memory overhead per channel（MB）
GAMMA_FLOPS_PER_CHANNEL = 1e6  # workload per channel（FLOPs）
ALPHA_FWD = 0.609   # forward proportion
ALPHA_BWD = 0.409  # backward proportion
S = 1e6  #  data size per channel # (bytes)

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

def create_sample_devices():
    return [
        Device("Device1", 2.0e6, 256, 100, 10),
        Device("Device2", 5.0e6, 128, 150, 10),
        Device("Device3", 1.5e6, 512, 80, 10),
        Device("Device4", 6.0e6, 192, 250, 10),
        Device("Device5", 4.5e6, 384, 120, 10),
    ]



def compute_execution_time(device):
    device.execution_time = device.total_channels * GAMMA_FLOPS_PER_CHANNEL / device.flops
    return device.execution_time

def compute_comm_time(src, dst, channels):
    if channels == 0:
        return 0.0
    min_bandwidth = min(src.bandwidth, dst.bandwidth)
    fwd = (ALPHA_FWD * S * channels) / (min_bandwidth * 1e6) * 1000  # ms
    bwd = (ALPHA_BWD * S * channels) / (min_bandwidth * 1e6) * 1000  # ms
    return src.latency + dst.latency + fwd + bwd

def initial_channel_assignment(devices, total_channels, unit, channel_mem_mb):
    print("\n[Initialization] Tentative allocation (before applying memory constraints):")
    total_perf = sum(d.flops for d in devices)
    max_channels = [int(d.memory_mb // channel_mem_mb) for d in devices]

    tentative_allocs = []
    for d in devices:
        share_ratio = d.flops / total_perf
        alloc = int(round((total_channels * share_ratio) / unit)) * unit
        tentative_allocs.append(alloc)
        print(f"{d.name}: tentative_alloc = {alloc}")

    overflow = 0
    print("\n[Allocation] Applying memory constraints and detecting overflow:")
    for i, d in enumerate(devices):
        max_local = int(max_channels[i] // unit) * unit
        if tentative_allocs[i] <= max_local:
            d.local_channels = tentative_allocs[i]
        else:
            d.local_channels = max_local
            overflow += tentative_allocs[i] - max_local
        print(f"{d.name}: local={d.local_channels} (max={max_local}), overflow={max(0, tentative_allocs[i] - max_local)}")

    print(f"\nTotal overflow to redistribute: {overflow} channels")

    while overflow > 0:
        candidates = [d for d in devices if d.total_channels + unit <= max_channels[devices.index(d)]]
        if not candidates:
            print("No device has space for remaining workload.")
            break
        best = max(candidates, key=lambda d: max_channels[devices.index(d)] - d.total_channels)
        best.local_channels += unit
        overflow -= unit
        print(f"→ Assigning +{unit} local channel to {best.name}")

    print("\n[Final Initial Allocation]")
    for d in devices:
        print(f"{d.name}: local={d.local_channels}, remote={d.remote_channels}, total={d.total_channels}")

def heuristic_iteration(devices, unit):
    for d in devices:
        compute_execution_time(d)

    src = max(devices, key=lambda d: d.execution_time)
    Tmax_before = src.execution_time

    # if src.local_channels < unit:
    #     return False

    candidates = []
    for dst in devices:
        if dst == src:
            continue
        compute_execution_time(dst)
        slack = Tmax_before - dst.execution_time
        if slack > 0:
            candidates.append((dst, slack))
    candidates.sort(key=lambda x: x[1], reverse=True)
   # print(f"\n trying to offloading src: {src.name}")

    for dst, _ in candidates:
        new_src_time = (src.local_channels + src.remote_channels - unit) * GAMMA_FLOPS_PER_CHANNEL / src.flops
        projected_remote_total = dst.remote_channels + unit
        dst_compute_time = projected_remote_total * GAMMA_FLOPS_PER_CHANNEL / dst.flops
        comm_time = compute_comm_time(src, dst, unit)
        total_remote_time = comm_time + dst_compute_time
        
        #dst_total_future_time = (dst.local_channels + projected_remote_total) * GAMMA_FLOPS_PER_CHANNEL / dst.flops
        dst_total_future_time = (dst.local_channels + projected_remote_total) * GAMMA_FLOPS_PER_CHANNEL / dst.flops
        condition1 = (total_remote_time <= new_src_time)
        # print(f"→ total_remote_time: {total_remote_time}, comm_time: {comm_time}, dst_compute_time: {dst_compute_time}")
        # print(f"→ new_src_time: {new_src_time}")
        condition2 = (dst_total_future_time <= new_src_time)
        # print(f"→ src: {src.name}, dst: {dst.name}, unit: {unit}, condition1: {condition1}, condition2: {condition2}")
        if condition1 and condition2:
            src.remote_channels -= unit
            dst.remote_channels += unit
            #print(f"→ Iteration Offload: {unit} channel from {src.name} to {dst.name}")
            return True

    return False

def simulate(devices, total_channels, unit, max_iter, threshold, channel_mem_mb):
    initial_channel_assignment(devices, total_channels, unit, channel_mem_mb)
    tmax_history = []
    bottleneck_history = []
    allocation_history = []

    for _ in range(max_iter):
        for d in devices:
            compute_execution_time(d)
        tmax = max(d.execution_time for d in devices)
        tmin = min(d.execution_time for d in devices)
        # print(f"\n[Iteration] Tmax: {tmax:.2f} ms, Tmin: {tmin:.2f} ms")
        bottleneck = max(devices, key=lambda d: d.execution_time)

        tmax_history.append(tmax)
        bottleneck_history.append(bottleneck.name)
        allocation_history.append([(d.local_channels, d.remote_channels) for d in devices])

        if tmax - tmin < threshold:
            print(f"\n[Convergence] Stopping criteria met: Tmax - Tmin < {threshold}")
            break
        if not heuristic_iteration(devices, 1) :
            print("\n[Convergence] No further offloading possible.")
            break

    return tmax_history, bottleneck_history, allocation_history

def summarize_convergence(tmax_history, max_iter, threshold):
    final_iter = len(tmax_history)
    print(f"\n[Summary of Convergence]")
    print(f"Total iterations run: {final_iter}")
    if final_iter < max_iter:
        print(f"→ Converged early")
    else:
        print(f"→ Stopped at max iterations: {max_iter}")
    print(f"Initial Tmax: {tmax_history[0]:.2f} ms")
    print(f"Final Tmax: {tmax_history[-1]:.2f} ms")
    print(f"Reduction: {tmax_history[0] - tmax_history[-1]:.2f} ms")
    print(f"Percentage reduction: {(1 - tmax_history[-1] / tmax_history[0]) * 100:.2f}%")

def plot_tmax_curve(tmax_history, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(tmax_history)), tmax_history, marker='o')
    plt.title("Convergence of Max Execution Time (tmax)")
    plt.xlabel("Iteration")
    plt.ylabel("Max Execution Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.show()
    print(f"Saved Tmax convergence plot to: {save_path}")

def save_detailed_allocation_csv(devices, allocation_history, save_path="channel_allocation_detailed.csv"):
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "device", "local", "remote", "total"])
        for iteration, allocation in enumerate(allocation_history):
            for device, (local, remote) in zip(devices, allocation):
                total = local - remote
                writer.writerow([iteration, device.name, local, remote, total])
    print(f"Saved detailed channel allocation history to: {save_path}")

def plot_channel_trend_per_device(allocation_history, save_dir="./results/", interval=2):
    os.makedirs(save_dir, exist_ok=True)
    num_devices = len(allocation_history[0])
    sampled_iterations = list(range(0, len(allocation_history), interval))

   
    # collect total channels from each device over sampled iterations
    sampled_data = {
        f"Device{dev_id+1}": [
            allocation_history[it][dev_id][0] - allocation_history[it][dev_id][1]
            for it in sampled_iterations
        ]
        for dev_id in range(num_devices)
    }

    image_paths = []
    for dev_id, (device, values) in enumerate(sampled_data.items(), start=1):
        plt.figure()
        plt.plot(sampled_iterations, values, marker='.')
        plt.title(f"Total Channels Over Iterations - {device}")
        plt.xlabel("Iteration")
        plt.ylabel("Total Channels")
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(save_dir, f"device_{dev_id}_over_time.png")
        plt.savefig(filename)
        plt.close()
        image_paths.append(filename)

    print(f"Saved channel trend plots to: {save_dir}")
    return image_paths

def plot_all_devices_in_one(allocation_history, save_path="./results/channel_trend_all_devices.png", interval=2):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_devices = len(allocation_history[0])
    sampled_iterations = list(range(0, len(allocation_history), interval))

    
    sampled_data = {
        f"Device{dev_id+1}": [
            allocation_history[it][dev_id][0] - allocation_history[it][dev_id][1]
            for it in sampled_iterations
        ]
        for dev_id in range(num_devices)
    }

    # draw all devices in one plot
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    for device_name, values in sampled_data.items():
        plt.plot(sampled_iterations, values, marker='.', label=device_name)

    plt.title("Total Channels Over Iterations (All Devices)")
    plt.xlabel("Iteration")
    plt.ylabel("Total Channels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.show()

    save_path

if __name__ == "__main__":
    devices = create_sample_devices()
    tmax_history, bottleneck_devices, allocation_history = simulate(
        devices,
        TOTAL_CHANNELS,
        CHANNEL_UNIT,
        MAX_ITER,
        CONVERGENCE_THRESHOLD,
        CHANNEL_MEMORY_MB
    )

    print("\nFinal Allocation and Times:")
    for d in devices:
        print(f"{d.name}: local={d.local_channels}, remote={d.remote_channels}, exec_time={d.execution_time:.2f} ms")

    summarize_convergence(tmax_history, MAX_ITER, CONVERGENCE_THRESHOLD)
    plot_tmax_curve(tmax_history, save_path="tmax_convergence.pdf")
    save_detailed_allocation_csv(devices, allocation_history)
    # plot_channel_trend_per_device(allocation_history)
    plot_all_devices_in_one(allocation_history)

    
