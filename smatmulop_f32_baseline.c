#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

#define DEBUG_FLAG 0
#define SIZE 2


// Function to perform matrix multiplication
void multiply(float* mat1, float* mat2, float* res, int size) {

    // Outer product accumulation: for each k, add A[:,k] * B[k,:]
    for (int k = 0; k < size; ++k) {
        for (int i = 0; i < size; ++i) {
            float a_ik = mat1[i * size + k];        // A[i][k] — element of column k in row i
            for (int j = 0; j < size; ++j) {
                float b_kj = mat2[k * size + j];    // B[k][j] — element of row k in column j
                res[i * size + j] += a_ik * b_kj;  // Accumulate outer product
            }
        }
    }
}

int main(int argc, char* argv[]) {

    printf("Testing matrix ");

    int size = SIZE;
    if (argc == 2) {
        size = atoi( argv[1] );
    }

    printf("size: %d x %d\n", size, size);
    
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
    }

    IFDEBUG{
        print_matrixf32(A, size, size, 1);
        print_matrixf32(B, size, size, 1);
    }

    // Start timer
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply(A, B, C, size);

    // Stop timer
    clock_t end_time = clock();

    IFDEBUG{
        print_matrixf32(C, size, size, 1);
    }

    // Calculate and print execution time
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : smatmulop_f32_baseline, %f, %d\n", execution_time, size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
