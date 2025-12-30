#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include <omp.h>
#include "utils.h"


#define DEBUG_FLAG 0
#define SIZE 8

#define DEFAULT_TILE_SIZE 2

// ATTENTION
// PARALLEL VERSION, work in progrss
// !! the synchronisation and reduction part is missing !!


//carica e riordina .... OTTIMIZZAZIONI???
inline void reordering(float* mat2, float* omat2,int size, int ts){

    float* ptr_omat2 = omat2;
    for( int i = 0; i < ts; i++ ){
        memcpy(ptr_omat2, mat2 + (size * i), sizeof(float) * ts);
        ptr_omat2 += ts;
    }

}


// Function to perform matrix multiplication
// ts: tile size, DEFAULT=4 ,passed as parameter (for tuning)
void multiply_gemm_2x2(float* mat1, float* mat2, float* res, const int size) {

    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    /// TODO handle shared memory of oB ?!?! (check..)
    // tile of orderedB, ts x ts : 
    // takes the B column blocks for the computation
    float* oB = malloc( sizeof(float) * 2 * 2 );

    // Tiling (h:high) [ts x ts] (default 4x4)
    #pragma omp parallel for collapse(2)
    for( int ih = 0; ih < size; ih += 2 ){
        for( int jh = 0; jh < size; jh += 2 ){

            //printf("Computation block C(%d:%d) \n", ih, jh);
            
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

            // TODO :: SYNCRONIZATION??? 
            /// scrivi in memoria la matrice 2x2
            __riscv_vse32_v_f32m1(&C[ih][jh], z2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], z3, vl);
            
        }
    }

}


void multiply_gemm_4x4(float* mat1, float* mat2, float* res, const int size) {
    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    /// TODO GESTIONE di oB CONDIVISA !!! (vedere..)
    // Buffer per la sottomatrice riordinata di B (4x4)
    float* oB = malloc(sizeof(float) * 4 * 4);

    // Tiling esterno: blocchi 4x4 
    #pragma omp parallel for collapse(2)
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

    free(oB);
}

// at this point, there will be other implementations for a reasonable number of vector register ...
// sure 8x8 and 16x16
// BUT take in mind that the these registers will be shared with all threads in execution
// SO it 


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

    printf("PARALLEL   size: %d x %d    tile_size:%d\n", size, size, tile_size);
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
    if( tile_size == 2 ){
        multiply_gemm_2x2(A, B, C, size);
    }
    else if( tile_size == 4 ){
        multiply_gemm_4x4(A, B, C, size);
    }

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
    printf("> BENCHMARK_RECORD : rvv_smatmulop_reordered_tiling_parallel, %f, %d, %d\n", execution_time, size, tile_size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
