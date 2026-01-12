import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

filename = sys.argv[1]

record_re = re.compile(
    r"BENCHMARK_RECORD\s*:\s*([A-Za-z0-9_]+),\s*([0-9.]+),\s*([0-9]+)(?:,\s*([0-9]+))?"
)

loads_re = re.compile(r"([\d,]+)\s+L1-dcache-loads")
misses_re = re.compile(r"([\d,]+)\s+L1-dcache-load-misses")

versions = defaultdict(lambda: defaultdict(dict))

with open(filename,"r") as f:
    content = f.read()

for rec in record_re.finditer(content):
    name = rec.group(1)
    exec_time = float(rec.group(2))
    size = int(rec.group(3))
    tile = rec.group(4)

    # chiave versione leggibile
    key = name if tile is None else f"{name}_tile{tile}"

    tail = content[rec.end():]

    loads = int(loads_re.search(tail).group(1).replace(",", ""))
    misses = int(misses_re.search(tail).group(1).replace(",", ""))

    miss_rate = (misses / loads) * 100

    versions[key][size] = {
        "time": exec_time,
        "miss_rate": miss_rate,
        "loads": loads,
        "misses": misses
    }

# ---------- grafico tempi ----------
plt.figure()
for version, vals in versions.items():
    xs = sorted(vals.keys())
    ys = [vals[x]["time"] for x in xs]
    plt.plot(xs, ys, marker="o", label=version)

plt.xlabel("Input size")
plt.ylabel("Tempo di esecuzione (s)")
plt.title("Execution time vs input size")
plt.legend()
plt.grid()
plt.tight_layout()

# ---------- grafico miss rate ----------
plt.figure()
for version, vals in versions.items():
    xs = sorted(vals.keys())
    ys = [vals[x]["miss_rate"] for x in xs]
    plt.plot(xs, ys, marker="o", label=version)

plt.xlabel("Input size")
plt.ylabel("Cache miss rate (%)")
plt.title("L1 miss rate vs input size")
plt.legend()
plt.grid()
plt.tight_layout()

# ---------- grafici a barre side-by-side ----------
sizes = sorted({s for v in versions.values() for s in v.keys()})

for s in sizes:
    labels = []
    loads = []
    misses = []

    for version in versions:
        if s in versions[version]:
            labels.append(version)
            loads.append(versions[version][s]["loads"])
            misses.append(versions[version][s]["misses"])

    x = range(len(labels))

    bar_width = 0.4
    x_loads = [i - bar_width/2 for i in x]
    x_misses = [i + bar_width/2 for i in x]

    plt.figure()

    # colonne affiancate
    plt.bar(x_loads, loads, width=bar_width, label="L1-dcache-loads")
    plt.bar(x_misses, misses, width=bar_width, label="L1-dcache-load-misses")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.title(f"L1 loads vs misses – size {s}")
    plt.ylabel("Event count")
    plt.legend()
    plt.tight_layout()

plt.show()
