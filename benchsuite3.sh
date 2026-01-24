#!/bin/bash

echo "Benchsuite V3 - (only var size) perf analysis"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <executable_path>"
    echo "Example: $0 out.txt ./build/riscv64/smatmulop_f32_baseline"
    exit 1
fi

FILENAME="$1"
EXE="$2"
FILE="report/$FILENAME"

# Crea la directory report se non esiste
mkdir -p report

# Verifica che l'eseguibile esista
if [ ! -x "$EXE" ]; then
    echo "Error: executable '$EXE' not found or not executable"
    exit 1
fi

echo "> saving to file: $FILE"
echo "Benchsuite V3 - (only var size) perf analysis (executable: $EXE)" >> "$FILE"

sizes=(256 512 1024 2048 4096)


# Modalità baseline: solo size, nessun kernel
for size in "${sizes[@]}"; do
    perf stat -e L1-dcache-loads,L1-dcache-load-misses "$EXE" "$size" &>> "$FILE"
done

echo "Done."