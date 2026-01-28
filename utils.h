#ifndef UTILS_H_
#define UTILS_H_

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#define IFDEBUG if(DEBUG_FLAG)

enum InputCase {
    RANDOM_INPUT = 0,                 // Matrici con numeri interi casuali
    IDENTITY_MATRIX = 1,                // Matrice identità
    ALL_ONES_MATRICES = 2,              // Matrici composte da soli uno
    CONST_ROWS_CONST_COLS = 3,    // Righe costanti e colonne costanti
    ARBITRARY_MATRIX_IDENTITY = 4,      // Matrice arbitraria moltiplicata per identità
    IDENTITY_INDEX_MATRIX = 5,          // Identità moltiplicata per indice
};

void print_matrix(double* M, int size);

void print_matrixf32(float* M, int tile_size, int size, int print_inline);

void print_lmatrixf32(float* M, int row_size, int num_elements);

void init_matrix_input(int input_case, float* A, float* B, int M, int N, int K);

#if defined(__riscv) && defined(__riscv_vector)
void print_vmatrixf32(int size, vfloat32m1_t c1, vfloat32m1_t c2);
void print_vmatrixf32_4x4(vfloat32m1_t c1, vfloat32m1_t c2, vfloat32m1_t c3, vfloat32m1_t c4);
#endif

#endif /* PROJECTUTILS_H_ */