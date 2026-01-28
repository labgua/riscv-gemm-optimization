#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

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
    printf("> Print Matrix f32 (row_size:%d,  num_elements:%d):\n", row_size, num_elements);
    for( int i = 0; i < num_elements; i++ ){
        printf("%.1f\t", M[i]);
        if( (i+1) % row_size == 0 ) printf("\n");
    }
}

void init_matrix_input(int input_case, float* _A, float* _B, int M, int N, int K){

    float (*A)[K] = (float (*)[K]) _A;   // M x K
    float (*B)[N] = (float (*)[N]) _B;   // K x N

    // A matrix  M x K
    for(int i = 0; i < M; i++){
        for(int j = 0; j<K; j++){
            switch (input_case)
            {
            case IDENTITY_MATRIX:
                if( i == j ) A[i][j] = 1.0;
                else A[i][j] = 0.0;
                break;

            case ALL_ONES_MATRICES:
                A[i][j] = 1.0;
                break;

            case CONST_ROWS_CONST_COLS:
                A[i][j] = (float) (i+1);
                break;

            case ARBITRARY_MATRIX_IDENTITY:  //??
                A[i][j] = (float) (i * K + j);
                break;

            case IDENTITY_INDEX_MATRIX:
                if( i == j ) A[i][j] = 1.0;
                else A[i][j] = 0.0;
                break;
            
            default: //RANDOM_INTEGERS
                A[i][j] = rand() % 10 + 1;
                break;
            }
        }
    }

    // B matrix  K x N
    for(int i = 0; i < K; i++){
        for(int j = 0; j<N; j++){
            switch (input_case)
            {
            case IDENTITY_MATRIX:
                if( i == j ) B[i][j] = 1.0;
                else B[i][j] = 0.0;
                break;

            case ALL_ONES_MATRICES:
                B[i][j] = 1.0;
                break;

            case CONST_ROWS_CONST_COLS:
                B[i][j] = (float) (j+1);
                break;

            case ARBITRARY_MATRIX_IDENTITY:  //??
                if( i == j ) B[i][j] = 1.0;
                else B[i][j] = 0.0;
                break;

            case IDENTITY_INDEX_MATRIX:
                B[i][j] = (float)(i * N + j);
                break;
            
            default: //RANDOM_INTEGERS
                B[i][j] = rand() % 10 + 1;
                break;
            }
        }
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