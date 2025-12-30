#ifndef UTILS_H_
#define UTILS_H_

#ifdef __riscv
#include <riscv_vector.h>
#endif

#define IFDEBUG if(DEBUG_FLAG)

void print_matrix(double* M, int size);

void print_matrixf32(float* M, int tile_size, int size, int print_inline);

#ifdef __riscv
void print_vmatrixf32(int size, vfloat32m1_t c1, vfloat32m1_t c2);
void print_vmatrixf32_4x4(vfloat32m1_t c1, vfloat32m1_t c2, vfloat32m1_t c3, vfloat32m1_t c4);
#endif

#endif /* PROJECTUTILS_H_ */