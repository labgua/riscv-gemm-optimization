#!/bin/bash

# builder.sh v2
# usage: <ARCH_TARGET>
# - this file chooses the compiler for the target if it is possible
# - if the arch can, return the exec-compiler else return true 


# Options

CMD_X86=gcc
CMD_QEMU=/home/sergio/UniSpec/HPC/PROGETTO/riscv64-elf-gcc/bin/riscv64-unknown-elf-gcc
CMD_RISCV=riscv64-linux-gnu-gcc

#----------------------------------------

ARCH_HOST=$(uname -m)
ARCH_TARGET=$1

if [ "$ARCH_HOST" == "x86_64" ] && [ "$ARCH_TARGET" == "riscv64_emu" ]; then
    echo "$CMD_QEMU"
elif [ "$ARCH_HOST" == "x86_64" ] && [ "$ARCH_TARGET" == "x86_64" ]; then
    echo "$CMD_X86"
elif [ "$ARCH_HOST" == "riscv64" ] && [ "$ARCH_TARGET" == "riscv64" ]; then
    echo "$CMD_RISCV"
else
    # Nessun compilatore disponibile → usa un comando neutro
    echo "true"
    exit 0
fi