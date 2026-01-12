import re
import sys
import csv

# Uso:
#   python parse_bench.py benchmark.txt

filename = sys.argv[1]

header = [
    "Version",
    "Input size",
    "KernelTile size",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "Cache miss rate (%)",
    "Exec Time (s)"
]

record_re = re.compile(
    r"BENCHMARK_RECORD\s*:\s*([A-Za-z0-9_]+),\s*([0-9.]+),\s*([0-9]+)(?:,\s*([0-9]+))?"
)

loads_re = re.compile(r"([\d,]+)\s+L1-dcache-loads")
misses_re = re.compile(r"([\d,]+)\s+L1-dcache-load-misses")

data = []

with open(filename, "r") as f:
    content = f.read()

records = record_re.finditer(content)
for rec in records:
    version = rec.group(1)
    exec_time = float(rec.group(2))
    size = int(rec.group(3))
    tile = rec.group(4) if rec.group(4) else ""

    # Cerca solo dopo questo record
    tail = content[rec.end():]

    loads_match = loads_re.search(tail)
    misses_match = misses_re.search(tail)

    if not loads_match or not misses_match:
        continue

    loads = int(loads_match.group(1).replace(",", ""))
    misses = int(misses_match.group(1).replace(",", ""))

    miss_rate = (misses / loads) * 100 if loads else 0.0

    data.append([
        version,
        size,
        tile,
        loads,
        misses,
        round(miss_rate, 3),
        exec_time
    ])

writer = csv.writer(sys.stdout)
writer.writerow(header)
for row in data:
    writer.writerow(row)
