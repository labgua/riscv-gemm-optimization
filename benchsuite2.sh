#!/bin/bash

echo "Benchsuite V3 - test outer product f32 version"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILENAME="$1"

FILE="report/$FILENAME"

echo "> save on file: $FILE"


# size to analyze: {256, 512, 1024, 2048, 4096}

# kernel selection on fixed size, rvv_smatmulop_f32_tiling
echo "rvv_smatmulop_f32_tiling (with selection kernel on 256)"
echo "rvv_smatmulop_f32_tiling (with selection kernel on 256)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_tiling 256 2 | grep "BENCHMARK_RECORD" >> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 256 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 256 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_tiling (with selection kernel on 512)"
echo "rvv_smatmulop_f32_tiling (with selection kernel on 512)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_tiling 512 2 | grep "BENCHMARK_RECORD" >> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 512 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 512 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_tiling (with selection kernel on 1024)"
echo "rvv_smatmulop_f32_tiling (with selection kernel on 1024)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_tiling 1024 2 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 1024 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 1024 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_tiling (with selection kernel on 2048)"
echo "rvv_smatmulop_f32_tiling (with selection kernel on 2048)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_tiling 2048 2 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 2048 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 2048 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_tiling (with selection kernel on 4096)"
echo "rvv_smatmulop_f32_tiling (with selection kernel on 4096)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_tiling 4096 2 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 4096 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 4096 8 | grep "BENCHMARK_RECORD">> $FILE

####

# kernel selection on fixed size, rvv_smatmulop_f32_reordered_tiling
echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 256)"
echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 256)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_reordered_tiling 256 2 | grep "BENCHMARK_RECORD" >> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 256 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 256 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 512)"
echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 512)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_reordered_tiling 512 2 | grep "BENCHMARK_RECORD" >> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 512 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 512 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 1024)"
echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 1024)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_reordered_tiling 1024 2 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 1024 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 1024 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 2048)"
echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 2048)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_reordered_tiling 2048 2 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 2048 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 2048 8 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 4096)"
echo "rvv_smatmulop_f32_reordered_tiling (with selection kernel on 4096)" >> $FILE
#./build/riscv64/rvv_smatmulop_f32_reordered_tiling 4096 2 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 4096 4 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 4096 8 | grep "BENCHMARK_RECORD">> $FILE


# variable size

echo "rvv_smatmulop_f32_tiling"
echo "rvv_smatmulop_f32_tiling" >> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 256 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 512 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 1024 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 2048 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_tiling 4096 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_reordered_tiling"
echo "rvv_smatmulop_f32_reordered_tiling" >> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 256 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 512 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 1024 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 2048 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling 4096 | grep "BENCHMARK_RECORD">> $FILE

echo "rvv_smatmulop_f32_reordered_tiling_parallel"
echo "rvv_smatmulop_f32_reordered_tiling_parallel" >> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel 256 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel 512 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel 1024 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel 2048 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel 4096 | grep "BENCHMARK_RECORD">> $FILE


# baseline: slow, at the end..

echo "baseline"
echo "baseline" >> $FILE
./build/riscv64/smatmulop_f32_baseline 256 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/smatmulop_f32_baseline 512 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/smatmulop_f32_baseline 1024 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/smatmulop_f32_baseline 2048 | grep "BENCHMARK_RECORD">> $FILE
./build/riscv64/smatmulop_f32_baseline 4096 | grep "BENCHMARK_RECORD">> $FILE