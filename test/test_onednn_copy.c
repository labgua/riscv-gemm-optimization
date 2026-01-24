#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdbool.h> 
#include "../utils.h"

#define PAGE_4K 4096
#define L1D_SIZE_QEMU 32 * 1024  //32kB su bananapi (emulazione per QEMU)

static inline int rnd_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

int get_m_unroll_factor() {
    return 16;
}

long get_l1d_cache_size() {

    #if defined(__linux__)
        long l1d_size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    #else
        // For bare-metal platform (es. QEMU senza OS)
        long l1d_size = L1D_SIZE_QEMU;
    #endif

    if (l1d_size == -1) { l1d_size = 32 * 1024; }
    return l1d_size;
}

int get_n_unroll_factor() {
    long l1d_size = get_l1d_cache_size();
    if (l1d_size >= 128 * 1024)
        return 16;
    else if (l1d_size >= 64 * 1024)
        return 8;
    else if (l1d_size >= 32 * 1024)
        return 4;
    else
        return 2;
}

void copy_A(bool isTransA, int K, const float *A, const int lda, float *ws) {
    
    printf(">>>> START-execution: copy_A\n");

    const int m = get_m_unroll_factor();
    float* ws_buffers_debug = ws;

    // Two-way software pipelining: overlap load and store
    for (int k = 0; k < K; k++) {
        int i = 0;

        const float *a_ptr = A + k * lda;
        if (i < m) {
            size_t vl0 = __riscv_vsetvl_e32m4(m - i);
            vfloat32m4_t v_a0 = __riscv_vle32_v_f32m4(a_ptr + i, vl0);
            int i0 = i;
            i += vl0;

            while (i < m) {
                size_t vl1 = __riscv_vsetvl_e32m4(m - i);
                vfloat32m4_t v_a1 = __riscv_vle32_v_f32m4(a_ptr + i, vl1);
                __riscv_vse32_v_f32m4(ws + i0, v_a0, vl0);

                printf(" step k:%d i:%d", k, i);
                //print_matrixf32(ws_buffers, K, lda, 0);
                print_lmatrixf32(ws_buffers_debug, get_m_unroll_factor(), K);

                i0 = i;
                vl0 = vl1;
                v_a0 = v_a1;
                i += vl1;
            }
            __riscv_vse32_v_f32m4(ws + i0, v_a0, vl0);

            printf(" step k:%d i:%d", k, i);
            //print_matrixf32(ws_buffers, K, lda, 0);
            print_lmatrixf32(ws_buffers_debug, get_m_unroll_factor(), K);
        }

        ws += m;
    }

    printf(">>>> STOP-execution: copy_A\n");
}

int main(int argc, char* argv[]) {

    printf("test_onednn_copy\n");

    int size = -1;

    if (argc == 2) {
        size = atoi( argv[1] );
    }
    if( size == -1 ) return -1;

    printf("size: %d x %d \n", size, size);

    float *A = (float*)malloc(size * size * sizeof(float));

    bool do_copy = (size / get_n_unroll_factor() > 3);
    printf("--> do_copy := size:%d / n_unroll_factor:%d >? 3  => %s\n", size, get_n_unroll_factor(), do_copy ? "VERO\0" : "FALSO\0");

    int elem_buffer = size * get_m_unroll_factor();
    int size_buffer = rnd_up(elem_buffer * sizeof(float), PAGE_4K);
    printf("size_buffer: allocated %d byte [for #%d float]\n", size_buffer, elem_buffer);

    float* ws_buffers = (float *)aligned_alloc( size_buffer , PAGE_4K );

    int x = 0;
    for(int i = 0; i < size; i++){
        for( int j = 0; j < size; j++ ){
            A[ i*size + j] = x++;
        }
    }

    printf("A");
    print_matrixf32(A, size, size, 0);

    int i = 0, z=0;
    while( size - i ){
        const float *a = &A[i];
        copy_A(false, size, a, size, ws_buffers);

        printf("\n\n%d) ws_buffer (&A[%d])", ++z, i);
        //print_matrixf32(ws_buffers, size, size, 0);
        print_lmatrixf32(ws_buffers, get_m_unroll_factor(), size);
        printf("\n\n");

        i += get_m_unroll_factor();
    }


}