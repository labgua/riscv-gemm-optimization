#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
//#include "utils.h"


// fujitsu usa vv,  onednn vf
// quale è il migliore???????
// è vero che con una istruzione in meno onednn (vf) va meglio?

int main(int argc, char const *argv[])
{   

    int N = 4096;
    int test = 0;
    int REPEAT = 5000000;

    if (argc == 2) {
        test = atoi(argv[1] );
    }
    if (argc == 3) {
        test = atoi(argv[1] );
        N = atoi( argv[2] );
    }
    if(argc == 4) {
        test = atoi(argv[1] );
        N = atoi( argv[2] );
        REPEAT = atoi(argv[3]);        
    }

    printf("Test fma scalar-vector vs broadcast-vector-vector\n");
    printf("N=%d      REPEAT:%d    TEST:%s\n", N, REPEAT, test==0 ? "scalar-vector\0" : "vector-vector\0");


    float* A = aligned_alloc( N * sizeof(float) , 4096);
    for(int i = 0; i< N; i++){
        A[i] = i;
    }


    vfloat32m1_t v0, vread0;
    vfloat32m1_t vbroadcast, v1, vread1;
    double exec_time;

    size_t vl = __riscv_vsetvl_e32m1(4);


    if( test == 0 ){
        /// caso scalar-vector
        clock_t start_time_test1 = clock();
        for( int j = 0; j < REPEAT; j++){
            for( int i = 0; i < N; i += 4 ){
                float scalar = i;
                vread0 = __riscv_vle32_v_f32m1(A + i, vl);
                v0 = __riscv_vfmacc_vf_f32m1(v0, scalar, vread0, vl);
            }
        }
        clock_t end_time_test1 = clock();

        exec_time = ((double)(end_time_test1 - start_time_test1)) / CLOCKS_PER_SEC;
    }
    else{
        /// caso broadcast-vector-vector
        clock_t start_time_test2 = clock();
        for( int j = 0; j < REPEAT; j++){
            for( int i = 0; i < N; i += 4 ){
                float scalar = i;
                vbroadcast = __riscv_vfmv_v_f_f32m1(scalar, vl);
                vread1 = __riscv_vle32_v_f32m1( A + i, vl);
                v1 = __riscv_vfmacc_vv_f32m1(v1, vbroadcast, vread1, vl);
            }
        }
        clock_t end_time_test2 = clock();

        exec_time = ((double)(end_time_test2 - start_time_test2)) / CLOCKS_PER_SEC;
    }

    printf("Exec time: %f\n", exec_time);

}
