import argparse
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def parse_benchmark_file(filepath):
    """
    Legge un file testuale e restituisce:
      - execution_data: dict {(version, size) -> time}
      - kernel_data: dict {(version, size, kernel) -> time}
    """
    execution_data = {}
    kernel_data = {}
    pattern = r'>\s*BENCHMARK_RECORD\s*:\s*([^,]+),\s*([0-9.]+),\s*([0-9]+)(?:,\s*([0-9]+))?'
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if not match:
                continue
            version = match.group(1).strip()
            time = float(match.group(2))
            size = int(match.group(3))
            kernel_str = match.group(4)
            kernel = int(kernel_str) if kernel_str is not None else 0

            # Normalizza kernel: 0 → 8
            if kernel == 0:
                kernel = 8

            execution_data[(version, size)] = time
            kernel_data[(version, size, kernel)] = time

    return execution_data, kernel_data

def plot_absolute_execution_time(filepath):
    exec_data, _ = parse_benchmark_file(filepath)
    versions = sorted(set(v for v, _ in exec_data.keys()))
    sizes = sorted(set(s for _, s in exec_data.keys()))

    plt.figure(figsize=(9, 6))
    for version in versions:
        times = []
        used_sizes = []
        for size in sizes:
            if (version, size) in exec_:
                times.append(exec_data[(version, size)])
                used_sizes.append(size)
        if times:
            label = version.replace('smatmulop_f32_', '').replace('rvv_', '')
            plt.plot(used_sizes, times, marker='o', label=label)

    plt.xlabel('Dimensione matrice (N x N)')
    plt.ylabel('Tempo di esecuzione [s]')
    plt.title('Absolute Execution Time')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(sizes)
    plt.tight_layout()
    plt.show()

def plot_speedup_vs_baseline(filepath):
    exec_data, _ = parse_benchmark_file(filepath)
    versions = sorted(set(v for v, _ in exec_data.keys()))
    sizes = sorted(set(s for _, s in exec_data.keys()))

    baseline_version = None
    for v in versions:
        if 'baseline' in v.lower():
            baseline_version = v
            break
    if not baseline_version:
        raise ValueError("Baseline non trovata.")

    plt.figure(figsize=(9, 6))
    for version in versions:
        if version == baseline_version:
            continue
        speedups = []
        used_sizes = []
        for size in sizes:
            key = (version, size)
            base_key = (baseline_version, size)
            if key in exec_data and base_key in exec_:
                speedup = exec_data[base_key] / exec_data[key]
                speedups.append(speedup)
                used_sizes.append(size)
        if speedups:
            label = version.replace('smatmulop_f32_', '').replace('rvv_', '')
            plt.plot(used_sizes, speedups, marker='o', label=label)

    plt.xlabel('Dimensione matrice (N x N)')
    plt.ylabel('Speedup rispetto alla baseline')
    plt.title('Speedup vs Baseline')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(sizes)
    plt.tight_layout()
    plt.show()

def plot_kernel_scaling(filepath, target_version, target_size=None):
    _, kernel_data = parse_benchmark_file(filepath)

    # Filtra per versione
    filtered = defaultdict(dict)  # size -> {kernel: time}
    for (ver, sz, k), t in kernel_data.items():
        if ver == target_version:
            filtered[sz][k] = t

    if not filtered:
        raise ValueError(f"Nessun dato trovato per la versione: {target_version}")

    plt.figure(figsize=(8, 5))

    if target_size is not None:
        # Caso singolo: una curva
        if target_size not in filtered:
            raise ValueError(f"Nessun dato per size={target_size} nella versione {target_version}")
        kernels = sorted(filtered[target_size].keys())
        times = [filtered[target_size][k] for k in kernels]
        plt.plot(kernels, times, marker='s')
        plt.title(f'Kernel Scaling: {target_version} @ {target_size}')
    else:
        # Caso multiplo: una curva per size
        all_sizes = sorted(filtered.keys())
        for size in all_sizes:
            kernels = sorted(filtered[size].keys())
            times = [filtered[size][k] for k in kernels]
            plt.plot(kernels, times, marker='o', label=f'{size}x{size}')

        plt.title(f'Kernel Scaling: {target_version} (tutte le dimensioni)')
        plt.legend()

    plt.xlabel('Dimensione kernel')
    plt.ylabel('Tempo di esecuzione [s]')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(sorted({k for d in filtered.values() for k in d.keys()}))
    plt.tight_layout()
    plt.show()

def plot_kernel_speedup(filepath, target_version, target_size=None):
    exec_data, kernel_data = parse_benchmark_file(filepath)

    # Estrai baseline per ogni size
    baseline_times = {}
    for (ver, sz), t in exec_data.items():
        if 'baseline' in ver.lower():
            baseline_times[sz] = t

    if not baseline_times:
        raise ValueError("Baseline non trovata nei dati.")

    # Filtra i dati della versione richiesta
    filtered = defaultdict(dict)  # size -> {kernel: time}
    for (ver, sz, k), t in kernel_data.items():
        if ver == target_version and sz in baseline_times:
            filtered[sz][k] = t

    if not filtered:
        raise ValueError(f"Nessun dato trovato per la versione: {target_version}")

    plt.figure(figsize=(8, 5))

    if target_size is not None:
        if target_size not in filtered:
            raise ValueError(f"Nessun dato per size={target_size} nella versione {target_version}")
        if target_size not in baseline_times:
            raise ValueError(f"Baseline mancante per size={target_size}")
        base_t = baseline_times[target_size]
        kernels = sorted(filtered[target_size].keys())
        speedups = [base_t / filtered[target_size][k] for k in kernels]
        plt.plot(kernels, speedups, marker='s')
        plt.title(f'Kernel Speedup vs Baseline: {target_version} @ {target_size}')
    else:
        all_sizes = sorted(filtered.keys())
        for size in all_sizes:
            if size not in baseline_times:
                continue
            base_t = baseline_times[size]
            kernels = sorted(filtered[size].keys())
            speedups = [base_t / filtered[size][k] for k in kernels]
            plt.plot(kernels, speedups, marker='o', label=f'{size}x{size}')

        plt.title(f'Kernel Speedup vs Baseline: {target_version} (tutte le dimensioni)')
        plt.legend()

    plt.xlabel('Dimensione kernel')
    plt.ylabel('Speedup rispetto alla baseline')
    plt.grid(True, linestyle='--', linewidth=0.5)
    all_kernels = sorted({k for d in filtered.values() for k in d.keys()})
    plt.xticks(all_kernels)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Genera grafici dai benchmark.")
    parser.add_argument(
        'chart_type',
        choices=['absolute_execution_time', 'speedup_vs_baseline', 'kernel_scaling', 'kernel_speedup'],
        help="Tipo di grafico da generare"
    )
    parser.add_argument('input_file', help="File di testo con i record BENCHMARK_RECORD")
    parser.add_argument('--version', help="Versione per kernel_scaling (obbligatorio)")
    parser.add_argument('--size', type=int, help="Dimensione input opzionale per kernel_scaling")

    args = parser.parse_args()

    if args.chart_type == 'absolute_execution_time':
        plot_absolute_execution_time(args.input_file)
    elif args.chart_type == 'speedup_vs_baseline':
        plot_speedup_vs_baseline(args.input_file)
    elif args.chart_type == 'kernel_scaling':
        if not args.version:
            raise ValueError("--version è obbligatorio per kernel_scaling")
        plot_kernel_scaling(args.input_file, target_version=args.version, target_size=args.size)
    elif args.chart_type == 'kernel_speedup':
        if not args.version:
            raise ValueError("--version è obbligatorio per kernel_speedup")
        plot_kernel_speedup(args.input_file, target_version=args.version, target_size=args.size)
    else:
        raise ValueError("Tipo di grafico non supportato.")

if __name__ == '__main__':
    main()