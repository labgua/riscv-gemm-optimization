#include "utils.h"

#include <stdio.h>

void print_matrix(double* M, int size){
    printf("> Print Matrix:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.1f ", M[i * size + j]);
        }
        printf("\n");
    }

    printf(">inline: ");
    printf("{");
    for (int i = 0; i < size; i++) {
        printf("{");
        for (int j = 0; j < size; j++) {
            printf("%.1f ", M[i * size + j]);
            if( j != size - 1 ) printf(", ");
        }
        printf("}");
        if( i != size - 1 ) printf(", ");
    }
    printf("}\n\n");    
}

void print_matrixf32(float* M, int tile_size, int size, int print_inline){
    printf("> Print Matrix f32:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < tile_size; j++) {
            printf("%.1f\t", M[i * size + j]);
        }
        printf("\n");
    }

    if( print_inline == 1 ){
        printf(">inline: ");
        printf("{");
        for (int i = 0; i < size; i++) {
            printf("{");
            for (int j = 0; j < tile_size; j++) {
                printf("%.1f ", M[i * size + j]);
                if( j != size - 1 ) printf(", ");
            }
            printf("}");
            if( i != size - 1 ) printf(", ");
        }
        printf("}\n\n");
    }
}

void print_lmatrixf32(float* M, int row_size, int num_elements){
    printf("> Print line-Matrix f32 (row_size:%d,  num_elements:%d):\n", row_size, num_elements);
    for( int i = 0; i < num_elements; i++ ){
        printf("%.1f\t", M[i]);
        if( (i+1) % row_size == 0 ) printf("\n");
    }
}

#ifdef __riscv

void print_vmatrixf32(int size, vfloat32m1_t c1, vfloat32m1_t c2){
    printf("> Print VectorMatrix f32 %dx%d:\n", size, size);
    float tmp[size];

    __riscv_vse32_v_f32m1(tmp, c1, size);
    printf("%.2f\t%.2f\n", tmp[0], tmp[1]);

    __riscv_vse32_v_f32m1(tmp, c1, size);
    printf("%.2f\t%.2f\n", tmp[0], tmp[1]);
}

void print_vmatrixf32_4x4(vfloat32m1_t c1, vfloat32m1_t c2, vfloat32m1_t c3, vfloat32m1_t c4){
    printf("> Print VectorMatrix f32 4x4:\n");
    float tmp[4];

    __riscv_vse32_v_f32m1(tmp, c1, 4);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", tmp[0], tmp[1], tmp[2], tmp[3]);

    __riscv_vse32_v_f32m1(tmp, c2, 4);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", tmp[0], tmp[1], tmp[2], tmp[3]);

    __riscv_vse32_v_f32m1(tmp, c3, 4);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", tmp[0], tmp[1], tmp[2], tmp[3]);

    __riscv_vse32_v_f32m1(tmp, c4, 4);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
}

#endif