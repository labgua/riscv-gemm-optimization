#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include <omp.h>
#include "utils.h"


#define DEBUG_FLAG 0
#define SIZE 8

// TILE SIZE SETTING
// 0   -> auto
// {i} -> force select ixi tile
#define DEFAULT_TILE_SIZE 0

// ATTENTION
// PARALLEL VERSION, work in progrss
// TODO test this version


//carica e riordina .... OTTIMIZZAZIONI???
inline void reordering(float* mat2, float* omat2,int size, int ts){

    float* ptr_omat2 = omat2;
    for( int i = 0; i < ts; i++ ){
        memcpy(ptr_omat2, mat2 + (size * i), sizeof(float) * ts);
        ptr_omat2 += ts;
    }

}


// Function to perform matrix multiplication with 2x2 tile
void multiply_gemm_2x2(float* mat1, float* mat2, float* res, const int size) {

    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    //float* oB = malloc( sizeof(float) * 2 * 2 );

    // Tiling (h:high) [ts x ts] (default 4x4)
    #pragma omp parallel for collapse(2)
    for( int ih = 0; ih < size; ih += 2 ){
        for( int jh = 0; jh < size; jh += 2 ){

            //printf("Computation block C(%d:%d) \n", ih, jh);

            // tile of orderedB, ts x ts : 
            // takes the B column blocks for the computation
            //float oB[ 2 * 2 ];
            float* oB = malloc( sizeof(float) * 2 * 2 );
            
            size_t vl = __riscv_vsetvl_e32m1( 2 );
            vfloat32m1_t z0, z1, z2, z3;

            // init/load submatrix C (load for future alpha beta BRGEMM)
            z2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            z3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for( int k = 0; k < size; k += 2 ){

                // LOAD & REORDERING B
                //printf("Load & reordering B(%d:%d) \n", k, jh);
                reordering( &B[k][jh], oB, size, 2);
                //printf("> Print Matrix: orderedB\n");
                //print_matrixf32(omat2, ts, ts, 0);

                /*
                printf(">> kernel gemm_2x2 :: execution-%d of kernel %dx%d ::  A[%d][%d] x B[%d][%d]\n", (k/ts)+1 , ts, ts, ih, k, k, jh);

                printf("Submatrix A\n");
                print_matrixf32(&A[ih][k], 2, size, 0);

                printf("Submatrix ordered B\n");
                print_matrixf32(omat2, 2, 2, 0);

                printf("Submatrix C\n");
                print_vmatrixf32(2, z2, z3);
                */

                // step1
                z0 = __riscv_vfmv_v_f_f32m1(A[ih][k], vl);
                z1 = __riscv_vle32_v_f32m1( oB, vl);
                z2 = __riscv_vfmacc_vv_f32m1(z2, z0, z1, vl);

                // step2
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k], vl);
                z3 = __riscv_vfmacc_vv_f32m1(z3, z0, z1, vl);

                // step 3
                z0 = __riscv_vfmv_v_f_f32m1(A[ih][k+1], vl);
                z1 = __riscv_vle32_v_f32m1( oB + 2, vl);
                z2 = __riscv_vfmacc_vv_f32m1(z2, z0, z1, vl);

                // step4
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 1], vl);
                z3 = __riscv_vfmacc_vv_f32m1(z3, z0, z1, vl);

                ///printf("------------------------------\n");
            }

            /// scrivi in memoria la matrice 2x2
            __riscv_vse32_v_f32m1(&C[ih][jh], z2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], z3, vl);
            
        }
    }

}

// Function to perform matrix multiplication with 4x4 tile
void multiply_gemm_4x4(float* mat1, float* mat2, float* res, const int size) {
    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    // Buffer per la sottomatrice riordinata di B (4x4)
    // float* oB = malloc(sizeof(float) * 4 * 4);

    // Tiling esterno: blocchi 4x4 
    #pragma omp parallel for collapse(2)
    for (int ih = 0; ih < size; ih += 4) {
        for (int jh = 0; jh < size; jh += 4) {

            size_t vl = __riscv_vsetvl_e32m1(4);

            //float oB[ 4 * 4 ];
            float* oB = malloc( sizeof(float) * 4 * 4 );

            // Carica il blocco corrente di C (4x4) in registri vettoriali
            // Inizializza accumulatore a zero (assumiamo C non pre-inizializzata)
            vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            // Loop sulle colonne di A / righe di B (in blocchi di 4)
            for (int k = 0; k < size; k += 4) {

                reordering( &B[k][jh], oB, size, 4);

                // Microkernel 4x4 completamente srotolato (senza loop)
                vfloat32m1_t z0, b_row;

                // --- k_offset = 0 (riga 0 di oB) ---
                b_row = __riscv_vle32_v_f32m1(&oB[0], vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 0], vl);
                c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 0], vl);
                c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 0], vl);
                c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 0], vl);
                c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);

                // --- k_offset = 1 (riga 1 di oB) ---
                b_row = __riscv_vle32_v_f32m1(&oB[4], vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 1], vl);
                c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 1], vl);
                c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 1], vl);
                c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 1], vl);
                c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);

                // --- k_offset = 2 (riga 2 di oB) ---
                b_row = __riscv_vle32_v_f32m1(&oB[8], vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 2], vl);
                c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 2], vl);
                c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 2], vl);
                c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 2], vl);
                c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);

                // --- k_offset = 3 (riga 3 di oB) ---
                b_row = __riscv_vle32_v_f32m1(&oB[12], vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 3], vl);
                c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 3], vl);
                c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 3], vl);
                c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);

                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 3], vl);
                c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
            }

            // Scrivi il blocco 4x4 risultante in C
            __riscv_vse32_v_f32m1(&C[ih + 0][jh], c0, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], c1, vl);
            __riscv_vse32_v_f32m1(&C[ih + 2][jh], c2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 3][jh], c3, vl);
        }
    }

}

void multiply_gemm_8x8(float* mat1, float* mat2, float* res, const int size) {
    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    #pragma omp parallel for collapse(2)
    for (int ih = 0; ih < size; ih += 8) {
        for (int jh = 0; jh < size; jh += 8) {

            //float oB[8 * 8];
            float* oB = malloc( sizeof(float) * 8 * 8 );

            size_t vl = __riscv_vsetvl_e32m1(8);

            // Accumulatori per le 8 righe di C
            vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (int k = 0; k < size; k += 8) {
                reordering(&B[k][jh], oB, size, 8);

                vfloat32m1_t z0, b_row;

                // --- r = 0 ---
                b_row = __riscv_vle32_v_f32m1(&oB[0], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 0], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 0], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 0], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 0], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 0], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 0], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 0], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 0], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 1 ---
                b_row = __riscv_vle32_v_f32m1(&oB[8], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 1], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 1], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 1], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 1], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 1], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 1], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 1], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 1], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 2 ---
                b_row = __riscv_vle32_v_f32m1(&oB[16], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 2], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 2], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 2], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 2], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 2], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 2], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 2], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 2], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 3 ---
                b_row = __riscv_vle32_v_f32m1(&oB[24], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 3], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 3], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 3], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 3], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 3], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 3], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 3], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 3], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 4 ---
                b_row = __riscv_vle32_v_f32m1(&oB[32], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 4], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 4], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 4], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 4], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 4], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 4], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 4], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 4], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 5 ---
                b_row = __riscv_vle32_v_f32m1(&oB[40], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 5], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 5], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 5], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 5], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 5], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 5], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 5], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 5], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 6 ---
                b_row = __riscv_vle32_v_f32m1(&oB[48], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 6], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 6], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 6], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 6], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 6], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 6], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 6], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 6], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);

                // --- r = 7 ---
                b_row = __riscv_vle32_v_f32m1(&oB[56], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 7], vl); c0 = __riscv_vfmacc_vv_f32m1(c0, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 7], vl); c1 = __riscv_vfmacc_vv_f32m1(c1, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 7], vl); c2 = __riscv_vfmacc_vv_f32m1(c2, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 7], vl); c3 = __riscv_vfmacc_vv_f32m1(c3, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 7], vl); c4 = __riscv_vfmacc_vv_f32m1(c4, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 7], vl); c5 = __riscv_vfmacc_vv_f32m1(c5, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 7], vl); c6 = __riscv_vfmacc_vv_f32m1(c6, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 7], vl); c7 = __riscv_vfmacc_vv_f32m1(c7, z0, b_row, vl);
            }

            // Store C
            __riscv_vse32_v_f32m1(&C[ih + 0][jh], c0, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], c1, vl);
            __riscv_vse32_v_f32m1(&C[ih + 2][jh], c2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 3][jh], c3, vl);
            __riscv_vse32_v_f32m1(&C[ih + 4][jh], c4, vl);
            __riscv_vse32_v_f32m1(&C[ih + 5][jh], c5, vl);
            __riscv_vse32_v_f32m1(&C[ih + 6][jh], c6, vl);
            __riscv_vse32_v_f32m1(&C[ih + 7][jh], c7, vl);
        }
    }
}

void multiply_gemm_16x16(float* mat1, float* mat2, float* res, const int size) {
    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    #pragma omp parallel for collapse(2)
    for (int ih = 0; ih < size; ih += 16) {
        for (int jh = 0; jh < size; jh += 16) {

            //float oB[16 * 16];
            float* oB = malloc( sizeof(float) * 16 * 16 );

            size_t vl = __riscv_vsetvl_e32m1(16);

            // Accumulatori per le 16 righe di C
            vfloat32m1_t c0  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c1  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c2  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c3  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c4  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c5  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c6  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c7  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c8  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c9  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c10 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c11 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c12 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c13 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c14 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c15 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (int k = 0; k < size; k += 16) {
                reordering(&B[k][jh], oB, size, 16);

                vfloat32m1_t z0, b_row;

                // --- r = 0 ---
                b_row = __riscv_vle32_v_f32m1(&oB[0], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 0], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 0], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 0], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 0], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 0], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 0], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 0], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 0], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 0], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 0], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 0], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 0], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 0], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 0], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 0], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 0], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 1 ---
                b_row = __riscv_vle32_v_f32m1(&oB[16], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 1], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 1], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 1], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 1], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 1], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 1], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 1], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 1], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 1], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 1], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 1], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 1], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 1], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 1], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 1], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 1], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 2 ---
                b_row = __riscv_vle32_v_f32m1(&oB[32], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 2], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 2], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 2], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 2], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 2], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 2], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 2], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 2], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 2], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 2], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 2], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 2], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 2], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 2], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 2], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 2], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 3 ---
                b_row = __riscv_vle32_v_f32m1(&oB[48], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 3], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 3], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 3], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 3], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 3], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 3], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 3], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 3], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 3], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 3], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 3], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 3], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 3], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 3], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 3], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 3], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 4 ---
                b_row = __riscv_vle32_v_f32m1(&oB[64], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 4], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 4], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 4], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 4], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 4], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 4], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 4], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 4], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 4], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 4], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 4], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 4], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 4], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 4], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 4], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 4], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 5 ---
                b_row = __riscv_vle32_v_f32m1(&oB[80], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 5], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 5], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 5], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 5], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 5], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 5], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 5], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 5], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 5], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 5], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 5], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 5], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 5], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 5], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 5], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 5], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 6 ---
                b_row = __riscv_vle32_v_f32m1(&oB[96], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 6], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 6], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 6], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 6], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 6], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 6], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 6], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 6], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 6], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 6], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 6], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 6], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 6], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 6], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 6], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 6], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 7 ---
                b_row = __riscv_vle32_v_f32m1(&oB[112], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 7], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 7], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 7], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 7], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 7], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 7], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 7], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 7], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 7], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 7], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 7], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 7], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 7], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 7], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 7], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 7], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 8 ---
                b_row = __riscv_vle32_v_f32m1(&oB[128], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 8], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 8], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 8], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 8], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 8], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 8], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 8], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 8], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 8], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 8], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 8], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 8], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 8], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 8], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 8], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 8], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 9 ---
                b_row = __riscv_vle32_v_f32m1(&oB[144], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 9], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 9], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 9], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 9], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 9], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 9], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 9], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 9], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 9], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 9], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 9], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 9], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 9], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 9], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 9], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 9], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 10 ---
                b_row = __riscv_vle32_v_f32m1(&oB[160], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 10], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 10], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 10], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 10], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 10], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 10], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 10], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 10], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 10], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 10], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 10], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 10], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 10], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 10], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 10], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 10], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 11 ---
                b_row = __riscv_vle32_v_f32m1(&oB[176], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 11], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 11], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 11], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 11], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 11], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 11], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 11], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 11], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 11], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 11], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 11], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 11], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 11], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 11], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 11], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 11], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 12 ---
                b_row = __riscv_vle32_v_f32m1(&oB[192], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 12], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 12], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 12], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 12], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 12], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 12], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 12], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 12], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 12], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 12], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 12], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 12], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 12], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 12], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 12], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 12], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 13 ---
                b_row = __riscv_vle32_v_f32m1(&oB[208], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 13], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 13], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 13], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 13], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 13], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 13], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 13], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 13], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 13], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 13], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 13], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 13], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 13], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 13], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 13], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 13], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 14 ---
                b_row = __riscv_vle32_v_f32m1(&oB[224], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 14], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 14], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 14], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 14], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 14], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 14], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 14], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 14], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 14], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 14], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 14], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 14], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 14], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 14], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 14], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 14], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);

                // --- r = 15 ---
                b_row = __riscv_vle32_v_f32m1(&oB[240], vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + 15], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 15], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + 15], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + 15], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + 15], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + 15], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + 15], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + 15], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 8][k + 15], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 9][k + 15], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + 15], vl); c10 = __riscv_vfmacc_vv_f32m1(c10, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + 15], vl); c11 = __riscv_vfmacc_vv_f32m1(c11, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + 15], vl); c12 = __riscv_vfmacc_vv_f32m1(c12, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + 15], vl); c13 = __riscv_vfmacc_vv_f32m1(c13, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + 15], vl); c14 = __riscv_vfmacc_vv_f32m1(c14, z0, b_row, vl);
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + 15], vl); c15 = __riscv_vfmacc_vv_f32m1(c15, z0, b_row, vl);
            }

            // Store C
            __riscv_vse32_v_f32m1(&C[ih + 0][jh],  c0,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh],  c1,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 2][jh],  c2,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 3][jh],  c3,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 4][jh],  c4,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 5][jh],  c5,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 6][jh],  c6,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 7][jh],  c7,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 8][jh],  c8,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 9][jh],  c9,  vl);
            __riscv_vse32_v_f32m1(&C[ih + 10][jh], c10, vl);
            __riscv_vse32_v_f32m1(&C[ih + 11][jh], c11, vl);
            __riscv_vse32_v_f32m1(&C[ih + 12][jh], c12, vl);
            __riscv_vse32_v_f32m1(&C[ih + 13][jh], c13, vl);
            __riscv_vse32_v_f32m1(&C[ih + 14][jh], c14, vl);
            __riscv_vse32_v_f32m1(&C[ih + 15][jh], c15, vl);
        }
    }
}

// ts: tile size {auto:0, 2, 4, 8, 16}  
void multiply_gemm(float* A, float* B, float* C, int N, int ts) {

    if( ts == 0 ){
        // Chiedi il massimo VL possibile per e32m1 (fino a 32)
        size_t vl = __riscv_vsetvl_e32m1(32);

        if (vl >= 16 && N >= 16) {
            ts = 16;
        } else if (vl >= 8 && N >= 8) {
            ts = 8;
        } else if (vl >= 4 && N >= 4) {
            ts = 4;
        } else {
            ts = 2;
        }
    }

    IFDEBUG{
        printf("kernel> ts=%d\n", ts);
    }

    switch (ts) {
        case 16: multiply_gemm_16x16(A, B, C, N); break;
        case 8:  multiply_gemm_8x8(A, B, C, N);  break;
        case 4:  multiply_gemm_4x4(A, B, C, N);  break;
        default: multiply_gemm_2x2(A, B, C, N); break;
    }
}

int main(int argc, char* argv[]) {

    printf("Testing matrix ");

    int size = SIZE;
    int tile_size = DEFAULT_TILE_SIZE;

    if (argc == 2) {
        size = atoi( argv[1] );
    }
    else if (argc == 3) {
        size = atoi( argv[1] );
        tile_size = atoi( argv[2] );
    }

    printf("PARALLEL   size: %d x %d    tile_size:%d %s\n", size, size, tile_size, tile_size==0 ? "AUTO\0" : "");
    ///printf("Num proc: %d\n", omp_get_max_threads());

    // check tile_size is correct ( tile_size <= size )
    if( tile_size > size ){
        printf("ERROR: tile_size:%d must be less than or equal to size:%d\n", tile_size, size);
        exit(EXIT_FAILURE);
    }
    
    // Allocate memory for matrices
    float *A = (float*)malloc(size * size * sizeof(float));
    float *B = (float*)malloc(size * size * sizeof(float));
    float *C = (float*)malloc(size * size * sizeof(float));

    // Init matrix values pseudorandom

    // set initial seed for rand, 1 if debug-mode
    srand( DEBUG_FLAG ? 1 : time(NULL) );

    for(int i = 0; i < size * size; i++ ){
       //A[i] = rand() % 10 + 1;
       A[i] = (float)(i+1 + 100);
       ///B[i] = rand() % 10 + 1;
       B[i] = (float)(i+1);
       C[i] = 0.0;
    }

    IFDEBUG{
        printf("A");
        print_matrixf32(A, size, size, 0);
        printf("B");
        print_matrixf32(B, size, size, 0);
    }

    // Start timer
    double start_time = omp_get_wtime();
    //clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply_gemm(A, B, C, size, tile_size;

    // Stop timer
    double end_time = omp_get_wtime();
    //clock_t end_time = clock();

    IFDEBUG{
        printf("C");
        print_matrixf32(C, size, size, 0);
    }

    // Calculate and print execution time
    double execution_time = (end_time - start_time); // [sec]
    //double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : rvv_smatmulop_f32_reordered_tiling_parallel, %f, %d, %d\n", execution_time, size, tile_size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
