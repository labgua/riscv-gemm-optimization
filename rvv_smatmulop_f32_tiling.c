#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include "utils.h"

#define DEBUG_FLAG 0
#define SIZE 8

// TILE SIZE SETTING
// 0   -> auto
// {i} -> force select ixi tile
#define DEFAULT_TILE_SIZE 0

// Function to perform matrix multiplication
void multiply_gemm_2x2(float* mat1, float* mat2, float* res, const int size) {

    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    // Tiling (h:high) [ts x ts]
    for( int ih = 0; ih < size; ih += 2 ){
        for( int jh = 0; jh < size; jh += 2 ){

            //printf("Computation block C(%d:%d) \n", ih, jh);
            
            size_t vl = __riscv_vsetvl_e32m1( 2 );
            vfloat32m1_t z0, z1, z2, z3;

            // init/load submatrix C (load for future alpha beta BRGEMM)
            z2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            z3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for( int k = 0; k < size; k += 2 ){

                /*
                printf(">> kernel gemm_2x2 :: execution-%d of kernel %dx%d ::  A[%d][%d] x B[%d][%d]\n", (k/ts)+1 , ts, ts, ih, k, k, jh);

                printf("Submatrix A\n");
                print_matrixf32(&A[ih][k], 2, size, 0);

                printf("Submatrix ordered B\n");
                print_matrixf32(&B[k][jh], 2, size, 0);

                printf("Submatrix C\n");
                print_vmatrixf32(2, z2, z3);
                */

                // step1
                z0 = __riscv_vfmv_v_f_f32m1(A[ih][k], vl);
                z1 = __riscv_vle32_v_f32m1( &B[k][jh], vl);
                z2 = __riscv_vfmacc_vv_f32m1(z2, z0, z1, vl);

                // step2
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k], vl);
                z3 = __riscv_vfmacc_vv_f32m1(z3, z0, z1, vl);

                // step 3
                z0 = __riscv_vfmv_v_f_f32m1(A[ih][k+1], vl);
                z1 = __riscv_vle32_v_f32m1( &B[k+1][jh], vl);
                z2 = __riscv_vfmacc_vv_f32m1(z2, z0, z1, vl);

                // step4
                z0 = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + 1], vl);
                z3 = __riscv_vfmacc_vv_f32m1(z3, z0, z1, vl);

                ///printf("------------------------------\n");
            }


            // Store the result in memory of block 2x2
            __riscv_vse32_v_f32m1(&C[ih][jh], z2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], z3, vl);
            
        }
    }

}


void multiply_gemm_4x4(float* mat1, float* mat2, float* res, const int size) {
    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    // Tiling esterno: blocchi 4x4 
    for (int ih = 0; ih < size; ih += 4) {
        for (int jh = 0; jh < size; jh += 4) {

            size_t vl = __riscv_vsetvl_e32m1(4);

            // Carica il blocco corrente di C (4x4) in registri vettoriali
            // Inizializza accumulatore a zero (assumiamo C non pre-inizializzata)
            vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            // Loop sulle colonne di A / righe di B (in blocchi di 4)
            for (int k = 0; k < size; k += 4) {

                // --- Loop interno su k_offset = 0,1,2,3 ---
                for (int k_off = 0; k_off < 4; ++k_off) {
                    // Carica una riga intera di B: B[k + k_off][jh : jh+4]
                    vfloat32m1_t b_row = __riscv_vle32_v_f32m1(&B[k + k_off][jh], vl);

                    vfloat32m1_t va;

                    // Elemento A[ih + 0][k + k_off]
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + k_off], vl);
                    c0 = __riscv_vfmacc_vv_f32m1(c0, va, b_row, vl);

                    // Elemento A[ih + 1][k + k_off]
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + k_off], vl);
                    c1 = __riscv_vfmacc_vv_f32m1(c1, va, b_row, vl);

                    // Elemento A[ih + 2][k + k_off]
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + k_off], vl);
                    c2 = __riscv_vfmacc_vv_f32m1(c2, va, b_row, vl);

                    // Elemento A[ih + 3][k + k_off]
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + k_off], vl);
                    c3 = __riscv_vfmacc_vv_f32m1(c3, va, b_row, vl);
                }
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

    for (int ih = 0; ih < size; ih += 8) {
        for (int jh = 0; jh < size; jh += 8) {

            
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

                // --- Loop interno su k_off = 0..7 ---
                for (int k_off = 0; k_off < 8; ++k_off) {
                    // Carica una riga intera di B: B[k + k_off][jh : jh+8]
                    vfloat32m1_t b_row = __riscv_vle32_v_f32m1(&B[k + k_off][jh], vl);

                    vfloat32m1_t va;

                    // Processa ogni elemento della colonna di A sequenzialmente
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 0][k + k_off], vl);
                    c0 = __riscv_vfmacc_vv_f32m1(c0, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 1][k + k_off], vl);
                    c1 = __riscv_vfmacc_vv_f32m1(c1, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 2][k + k_off], vl);
                    c2 = __riscv_vfmacc_vv_f32m1(c2, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 3][k + k_off], vl);
                    c3 = __riscv_vfmacc_vv_f32m1(c3, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 4][k + k_off], vl);
                    c4 = __riscv_vfmacc_vv_f32m1(c4, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 5][k + k_off], vl);
                    c5 = __riscv_vfmacc_vv_f32m1(c5, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 6][k + k_off], vl);
                    c6 = __riscv_vfmacc_vv_f32m1(c6, va, b_row, vl);

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 7][k + k_off], vl);
                    c7 = __riscv_vfmacc_vv_f32m1(c7, va, b_row, vl);
                }
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

    for (int ih = 0; ih < size; ih += 16) {
        for (int jh = 0; jh < size; jh += 16) {


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

                // --- Loop interno su k_off = 0..15 ---
                for (int k_off = 0; k_off < 16; ++k_off) {
                    // Carica una riga intera di B: B[k + k_off][jh : jh+16]
                    vfloat32m1_t b_row = __riscv_vle32_v_f32m1(&B[k + k_off][jh], vl);

                    vfloat32m1_t va;

                    va = __riscv_vfmv_v_f_f32m1(A[ih + 0 ][k + k_off], vl);  c0  = __riscv_vfmacc_vv_f32m1(c0,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 1 ][k + k_off], vl);  c1  = __riscv_vfmacc_vv_f32m1(c1,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 2 ][k + k_off], vl);  c2  = __riscv_vfmacc_vv_f32m1(c2,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 3 ][k + k_off], vl);  c3  = __riscv_vfmacc_vv_f32m1(c3,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 4 ][k + k_off], vl);  c4  = __riscv_vfmacc_vv_f32m1(c4,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 5 ][k + k_off], vl);  c5  = __riscv_vfmacc_vv_f32m1(c5,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 6 ][k + k_off], vl);  c6  = __riscv_vfmacc_vv_f32m1(c6,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 7 ][k + k_off], vl);  c7  = __riscv_vfmacc_vv_f32m1(c7,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 8 ][k + k_off], vl);  c8  = __riscv_vfmacc_vv_f32m1(c8,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 9 ][k + k_off], vl);  c9  = __riscv_vfmacc_vv_f32m1(c9,  va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 10][k + k_off], vl);  c10 = __riscv_vfmacc_vv_f32m1(c10, va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 11][k + k_off], vl);  c11 = __riscv_vfmacc_vv_f32m1(c11, va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 12][k + k_off], vl);  c12 = __riscv_vfmacc_vv_f32m1(c12, va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 13][k + k_off], vl);  c13 = __riscv_vfmacc_vv_f32m1(c13, va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 14][k + k_off], vl);  c14 = __riscv_vfmacc_vv_f32m1(c14, va, b_row, vl);
                    va = __riscv_vfmv_v_f_f32m1(A[ih + 15][k + k_off], vl);  c15 = __riscv_vfmacc_vv_f32m1(c15, va, b_row, vl);
                }
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

    printf("size: %d x %d    tile_size:%d %s\n", size, size, tile_size, tile_size==0 ? "AUTO\0" : "");
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
       A[i] = rand() % 10 + 1;
       B[i] = rand() % 10 + 1;
       C[i] = 0.0;
    }

    IFDEBUG{
        printf("A");
        print_matrixf32(A, size, size, 0);
        printf("B");
        print_matrixf32(B, size, size, 0);
    }

    // Start timer
    //double start_time = omp_get_wtime();
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply_gemm(A, B, C, size, tile_size);

    // Stop timer
    //double end_time = omp_get_wtime();
    clock_t end_time = clock();

    IFDEBUG{
        printf("C");
        print_matrixf32(C, size, size, 0);
    }

    // Calculate and print execution time
    //double execution_time = (end_time - start_time); // [sec]
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : rvv_smatmulop_f32_tiling, %f, %d, %d\n", execution_time, size, tile_size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
