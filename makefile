# COMPILATION SUITE V4
#
# DON'T CHANGE THIS FILE
# this file takes only order in compilation  
#
# this file uses builder.sh script, resolve association host-target
# change setting in builder.sh file for compiler settings

#resolve compilers in makefile
CC_X86_64 := $(shell ./builder.sh x86_64)
CC_RISCV64 := $(shell ./builder.sh riscv64)
CC_RISCV64_EMU := $(shell ./builder.sh riscv64_emu)
CC_RISCV64_14 := gcc

RISCV_OPT = -march=rv64gcv -mabi=lp64d
RISCV_OPT_NOVET = -march=rv64gc -mabi=lp64d

TARGETS = smatmul_baseline \
          smatmul_loopinterchange_O0 \
          smatmul_loopinterchange_O1 \
          smatmul_loopinterchange_O2 \
          smatmul_loopinterchange_O3 \
          smatmul_pfor \
          smatmul_tiling \
          smatmul_tiling_v2 \
          smatmul_tiling_v3 \
          smatmul_tiling_v4 \
          smatmul_recursive \
          rvv_smatmul_recursive \
          rvv2_smatmul_recursive \
          rvv_smatmul_recursive_O3 \
          rvv2_smatmul_recursive_O3 \
		  rvv_smatmulop_reordered_tiling \
#         TODO ...

# Shared objects paths
UTILS_O_X86    = build/x86_64/utils.o
UTILS_O_QEMU   = build/qemu/utils.o
UTILS_O_RISCV  = build/riscv64/utils.o

all: $(TARGETS)

# Compile utils.o foreach arch
$(UTILS_O_X86):
	@mkdir -p $(dir $@)
	$(CC_X86_64) -c -o $@ utils.c

$(UTILS_O_QEMU):
	@mkdir -p $(dir $@)
	$(CC_RISCV64_EMU) -c -o $@ utils.c $(RISCV_OPT)

$(UTILS_O_RISCV):
	@mkdir -p $(dir $@)
	$(CC_RISCV64) -c -o $@ utils.c $(RISCV_OPT)

# Explicit rule: make utils.o for all arch
utils: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)



# compile with -O0
smatmul_baseline: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O0 -o build/x86_64/smatmul_baseline smatmul_baseline.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O0 -o build/qemu/smatmul_baseline smatmul_baseline.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O0 -o build/riscv64/smatmul_baseline smatmul_baseline.c $(UTILS_O_RISCV) $(RISCV_OPT)

smatmul_loopinterchange_O0: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O0 -o build/x86_64/smatmul_loopinterchange_O0 smatmul_loopinterchange.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O0 -o build/qemu/smatmul_loopinterchange_O0 smatmul_loopinterchange.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O0 -o build/riscv64/smatmul_loopinterchange_O0 smatmul_loopinterchange.c $(UTILS_O_RISCV) $(RISCV_OPT)

smatmul_loopinterchange_O1: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O1 -o build/x86_64/smatmul_loopinterchange_O1 smatmul_loopinterchange.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O1 -o build/qemu/smatmul_loopinterchange_O1 smatmul_loopinterchange.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O1 -o build/riscv64/smatmul_loopinterchange_O1 smatmul_loopinterchange.c $(UTILS_O_RISCV) $(RISCV_OPT)

smatmul_loopinterchange_O2: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O2 -o build/x86_64/smatmul_loopinterchange_O2 smatmul_loopinterchange.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O2 -o build/qemu/smatmul_loopinterchange_O2 smatmul_loopinterchange.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O2 -o build/riscv64/smatmul_loopinterchange_O2 smatmul_loopinterchange.c $(UTILS_O_RISCV) $(RISCV_OPT)

smatmul_loopinterchange_O3: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O3 -o build/x86_64/smatmul_loopinterchange_O3 smatmul_loopinterchange.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/smatmul_loopinterchange_O3 smatmul_loopinterchange.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/smatmul_loopinterchange_O3 smatmul_loopinterchange.c $(UTILS_O_RISCV) $(RISCV_OPT)

# these only possible for risc-v ...
rvv_smatmul_recursive: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O2 -o build/qemu/rvv_smatmul_recursive rvv_smatmul_recursive.c -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O2 -o build/riscv64/rvv_smatmul_recursive rvv_smatmul_recursive.c -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)

rvv2_smatmul_recursive: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O2 -o build/qemu/rvv2_smatmul_recursive rvv2_smatmul_recursive.c -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O2 -o build/riscv64/rvv2_smatmul_recursive rvv2_smatmul_recursive.c -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)

rvv_smatmul_recursive_O3: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/rvv_smatmul_recursive_O3 rvv_smatmul_recursive.c -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/rvv_smatmul_recursive_O3 rvv_smatmul_recursive.c -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)

rvv2_smatmul_recursive_O3: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/rvv2_smatmul_recursive_O3 rvv2_smatmul_recursive.c -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/rvv2_smatmul_recursive_O3 rvv2_smatmul_recursive.c -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)


## NEW VERSIONS
smatmul_f32_baseline: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O3 -fno-tree-vectorize -o build/x86_64/smatmul_f32_baseline smatmul_f32_baseline.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O3 -fno-tree-vectorize -o build/qemu/smatmul_f32_baseline smatmul_f32_baseline.c $(UTILS_O_QEMU) $(RISCV_OPT_NOVET)
	$(CC_RISCV64) -O3 -fno-tree-vectorize -o build/riscv64/smatmul_f32_baseline smatmul_f32_baseline.c $(UTILS_O_RISCV) $(RISCV_OPT_NOVET)

smatmulop_f32_baseline: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O3 -fno-tree-vectorize -o build/x86_64/smatmulop_f32_baseline smatmulop_f32_baseline.c $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O3 -fno-tree-vectorize -o build/qemu/smatmulop_f32_baseline smatmulop_f32_baseline.c $(UTILS_O_QEMU) $(RISCV_OPT_NOVET)
	$(CC_RISCV64) -O3 -fno-tree-vectorize -o build/riscv64/smatmulop_f32_baseline smatmulop_f32_baseline.c $(UTILS_O_RISCV) $(RISCV_OPT_NOVET)

smatmulop_f32_baseline_autovect: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_14) -O3 -o build/riscv64/smatmulop_f32_baseline_autovect smatmulop_f32_baseline.c $(UTILS_O_RISCV) $(RISCV_OPT) -ffast-math

rvv_smatmulop_f32_tiling: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/rvv_smatmulop_f32_tiling rvv_smatmulop_f32_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/rvv_smatmulop_f32_tiling rvv_smatmulop_f32_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT)

rvv_smatmulop_f32_reordered_tiling: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/rvv_smatmulop_f32_reordered_tiling rvv_smatmulop_f32_reordered_tiling.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/rvv_smatmulop_f32_reordered_tiling rvv_smatmulop_f32_reordered_tiling.c $(UTILS_O_RISCV) $(RISCV_OPT)

	$(CC_RISCV64_EMU) -O3 -o build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel rvv_smatmulop_f32_reordered_tiling_parallel.c -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/rvv_smatmulop_f32_reordered_tiling_parallel rvv_smatmulop_f32_reordered_tiling_parallel.c -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)

rvv_smatmulop_f32_ktiling: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/rvv_smatmulop_f32_ktiling rvv_smatmulop_f32_ktiling.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/rvv_smatmulop_f32_ktiling rvv_smatmulop_f32_ktiling.c $(UTILS_O_RISCV) $(RISCV_OPT)

onednn_rvv_smatmul_f32: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/onednn_rvv_smatmul_f32 onednn_rvv_smatmul_f32.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/onednn_rvv_smatmul_f32 onednn_rvv_smatmul_f32.c $(UTILS_O_RISCV) $(RISCV_OPT)


## TEST code

test_onednn_copy: $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_RISCV64_EMU) -O3 -o build/qemu/test_onednn_copy test/test_onednn_copy.c $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/test_onednn_copy test/test_onednn_copy.c $(UTILS_O_RISCV) $(RISCV_OPT)

test_fma_vv_sv:
	$(CC_RISCV64_EMU) -O3 -o build/qemu/test_fma_vv_sv test/test_fma_vv_sv.c $(RISCV_OPT)
	$(CC_RISCV64) -O3 -o build/riscv64/test_fma_vv_sv test/test_fma_vv_sv.c $(UTILS_O_RISCV) $(RISCV_OPT)


#smatmul_pfor:
#smatmul_tiling:
#smatmul_tiling_v2:
#smatmul_tiling_v3:
#smatmul_tiling_v4:
#smatmul_recursive:
# ...
# compile target with -O2 (because the best is -O2 in this case, vectorization will do apart, not with -O3)
%: %.c $(UTILS_O_X86) $(UTILS_O_QEMU) $(UTILS_O_RISCV)
	$(CC_X86_64) -O2 -o build/x86_64/$@ $< -fopenmp $(UTILS_O_X86)
	$(CC_RISCV64_EMU) -O2 -o build/qemu/$@ $< -fopenmp $(UTILS_O_QEMU) $(RISCV_OPT)
	$(CC_RISCV64) -O2 -o build/riscv64/$@ $< -fopenmp $(UTILS_O_RISCV) $(RISCV_OPT)

clean:
	rm -f build/x86_64/*
	rm -f build/qemu/*
	rm -f build/riscv64/*
