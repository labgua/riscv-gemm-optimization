#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdbool.h> 
#include "utils.h"

/**
 * onednn_rvv_gemm_f32 WITHOUT oneDNN
 * - The onednn code implements matrix multiplication following BLAS conventions,
 *   treating A and B as column-major matrices from a logical standpoint.
 * - Their kernel handles the row-major format but are logical reinterpreted
 *   as column-major matrixes
 * - As in BLAS, the implemented product is defined in column-major order:
 *   C_col(N×M) = A_col(N×K) × B_col(K×M)
 * - Assuming transA=false and transB=false, if A and B are stored in row-major layout
 *   without reinterpretation, the computed result corresponds to:
 *   matmul(A, B) = ((A^T) × (B^T))^T = B × A
 * - Therefore, swapping A and B in the call yields:
 *   matmul(B, A) = ((B^T) × (A^T))^T = A × B
 * - In this first implementation between square matrices, it is not necessary to 
 *   manage the dimensions (N=M=K), but in the case of a generic product, it is also 
 *   necessary to swap N with M for a matrix product between generic non-square matrices.
 */

// [0, 1]
#define DEBUG_INPUT_FLAG 0

// [0, 1, 2, 3, 4]
#define DEBUG_KERNEL 4

// [0, 1]
#define DEBUG_PRINT_IO 1

// [0, 1]
#define RANDOM 0

#define SIZE 2



// TODO verificare sia questo il valore
#define PAGE_4K 4096

#define L1D_SIZE_QEMU 32 * 1024  //32kB su bananapi (emulazione per QEMU)


#define MIN(a, b) (((a) < (b)) ? (a) : (b))


/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Modifications:
 * - extrapolation code from oneDNN risc-v code
 *   https://github.com/uxlfoundation/oneDNN/blob/main/src/cpu/rv64/gemm/rvv_gemm_f32.cpp
 */


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

int get_m_unroll_factor() {
    return 16;
}

static inline int rnd_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static inline int rnd_dn(int a, int b) {
    return (b == 0) ? 0 : (a / b) * b;
}

inline void get_thr_block (int* from, int* to, int* myN, int NB, int N, int ithr) {
    *from = NB * (ithr);
    *to = NB * (ithr + 1);
    if (*to > N) *to = N;
    *myN = *to - *from;
};

void copy_A(bool isTransA, int K, const float *A, const int lda, float *ws) {
    
    const int m = get_m_unroll_factor();

    float* ws_buffer_debug = NULL;
    if( DEBUG_KERNEL >= 3 ) ws_buffer_debug = ws;

    // Two-way software pipelining: overlap load and store
    for (int k = 0; k < K; k++) {
        int i = 0;

        if (isTransA) { /// No TransA
            ptrdiff_t stride = lda * sizeof(float);
            if (i < m) {
                size_t vl0 = __riscv_vsetvl_e32m4(m - i);
                const float *a_ptr0 = A + i * lda + k;
                vfloat32m4_t v_a0 = __riscv_vlse32_v_f32m4(a_ptr0, stride, vl0);
                int i0 = i;
                i += vl0;

                while (i < m) {
                    size_t vl1 = __riscv_vsetvl_e32m4(m - i);
                    const float *a_ptr1 = A + i * lda + k;
                    vfloat32m4_t v_a1
                            = __riscv_vlse32_v_f32m4(a_ptr1, stride, vl1);
                    __riscv_vse32_v_f32m4(ws + i0, v_a0, vl0);

                    i0 = i;
                    vl0 = vl1;
                    v_a0 = v_a1;
                    i += vl1;
                }
                __riscv_vse32_v_f32m4(ws + i0, v_a0, vl0);
            }
        
        } else {
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

                    i0 = i;
                    vl0 = vl1;
                    v_a0 = v_a1;
                    i += vl1;
                }
                __riscv_vse32_v_f32m4(ws + i0, v_a0, vl0);
            }
        }
        ws += m;

        if( DEBUG_KERNEL >= 3 ) {
            printf("copy_A>: step-%d", i);
            ///print_matrixf32(ws_buffer_debug, K, lda, 0);
            print_lmatrixf32(ws_buffer_debug, get_m_unroll_factor(), K * get_m_unroll_factor());
        }
    }
}


/*
#define STORE_C(C_PTR, V_C_REG, ALPHA, BETA, VL) \
    do { \
        float *c_final_ptr = (C_PTR); \
        if ((BETA) == 0.0f) { \
            vfloat32m1_t v_res \
                    = __riscv_vfmul_vf_f32m1((V_C_REG), (ALPHA), (VL)); \
            __riscv_vse32_v_f32m1(c_final_ptr, v_res, (VL)); \
        } else { \
            vfloat32m1_t v_c_old = __riscv_vle32_v_f32m1(c_final_ptr, (VL)); \
            vfloat32m1_t v_res \
                    = __riscv_vfmul_vf_f32m1(v_c_old, (BETA), (VL)); \
            v_res = __riscv_vfmacc_vf_f32m1(v_res, (ALPHA), (V_C_REG), (VL)); \
            __riscv_vse32_v_f32m1(c_final_ptr, v_res, (VL)); \
        } \
    } while (0)
*/

#define STORE_C(C_PTR, V_C_REG, ALPHA, BETA, VL)  __riscv_vse32_v_f32m1((float*)(C_PTR), (V_C_REG), (VL))


/*
 * KERNEL TEMPLATE 
 * X:kernel_size
 *
 * static void kernel_mxn_XxX(bool isTransA, bool isTransB, int K, const float *A, int lda, const float *B,
 *        int ldb, float *C, int ldc, float alpha, float beta,
 *        int ithr);
 */

static void kernel_mxn_2x2(bool isTransA, bool isTransB, int K, const float *A, int lda, const float *B,
        int ldb, float *C, int ldc, float alpha, float beta,
        int ithr)
{
    if(DEBUG_KERNEL > 0)printf("> kernel_mxn_2x2\n");

    const int m = get_m_unroll_factor();
    //const int n = 2;
    //MAYBE_UNUSED(ithr);
    //MAYBE_UNUSED(n);

    int i = 0;
    while (i < m) {
        size_t vl = __riscv_vsetvl_e32m1(m - i);

        vfloat32m1_t v_c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            vfloat32m1_t v_a;
            if (isTransA) {  // isTransA == false
                ptrdiff_t stride_a = lda * sizeof(float);
                v_a = __riscv_vlse32_v_f32m1(A + i * lda + k, stride_a, vl);
            } else {
                v_a = __riscv_vle32_v_f32m1(A + i + k * lda, vl);
            }

            const float *b_ptr = isTransB ? &B[k * ldb] : &B[k];
            const int b_stride = isTransB ? 1 : ldb;

            v_c0 = __riscv_vfmacc_vf_f32m1(
                    v_c0, b_ptr[0 * b_stride], v_a, vl);
            v_c1 = __riscv_vfmacc_vf_f32m1(
                    v_c1, b_ptr[1 * b_stride], v_a, vl);
        }

        STORE_C(C + 0 * ldc + i, v_c0, alpha, beta, vl);
        STORE_C(C + 1 * ldc + i, v_c1, alpha, beta, vl);

        i += vl;
    }   
}

static void kernel_mxn_4x4(bool isTransA, bool isTransB, int K, const float *A, int lda, const float *B,
        int ldb, float *C, int ldc, float alpha, float beta,
        int ithr)
{
    if(DEBUG_KERNEL > 0)printf("> kernel_mxn_4x4 isTransA:%s isTransB:%s\n",
        isTransA ? "TRUE\0" : "FALSE\0",
        isTransB ? "TRUE\0" : "FALSE\0"
    );

    const int m = get_m_unroll_factor();
    //const int n = 4;

    int i = 0;
    while (i < m) {
        size_t vl = __riscv_vsetvl_e32m1(m - i);

        vfloat32m1_t v_c0, v_c1, v_c2, v_c3;

        v_c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            vfloat32m1_t v_a;
            if (isTransA) { // isTransA == false
                ptrdiff_t stride_a = lda * sizeof(float);
                v_a = __riscv_vlse32_v_f32m1(A + i * lda + k, stride_a, vl);
            } else {
                v_a = __riscv_vle32_v_f32m1(A + i + k * lda, vl);
            }

            const float *b_ptr = isTransB ? &B[k * ldb] : &B[k];
            const int b_stride = isTransB ? 1 : ldb;

            if( DEBUG_KERNEL >= 4 ){
                float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                size_t vl2 = __riscv_vsetvl_e32m1(4);
                __riscv_vse32_v_f32m1(tmp, v_a, vl2);
                printf(">> indexA[%d + %d * %d]  indexB[%d] \n   calc: row-Va:(%f, %f, %f, %f) (x) col-b:(%f, %f, %f, %f)\n",
                    i, k, lda,   k,
                    tmp[0], tmp[1], tmp[2], tmp[3],
                    b_ptr[0 * b_stride], b_ptr[1 * b_stride], b_ptr[2 * b_stride], b_ptr[3 * b_stride]
                );
            }

            v_c0 = __riscv_vfmacc_vf_f32m1(
                    v_c0, b_ptr[0 * b_stride], v_a, vl);

            v_c1 = __riscv_vfmacc_vf_f32m1(
                    v_c1, b_ptr[1 * b_stride], v_a, vl);

            v_c2 = __riscv_vfmacc_vf_f32m1(
                    v_c2, b_ptr[2 * b_stride], v_a, vl);

            v_c3 = __riscv_vfmacc_vf_f32m1(
                    v_c3, b_ptr[3 * b_stride], v_a, vl);

            if(DEBUG_KERNEL >= 2){
                printf("> vector matrix\n");
                print_vmatrixf32_4x4(v_c0, v_c1, v_c2, v_c3);
            }
        }

        STORE_C(C + 0 * ldc + i, v_c0, alpha, beta, vl);
        STORE_C(C + 1 * ldc + i, v_c1, alpha, beta, vl);
        STORE_C(C + 2 * ldc + i, v_c2, alpha, beta, vl);
        STORE_C(C + 3 * ldc + i, v_c3, alpha, beta, vl);

        i += vl;
    }

}

static void kernel_mxn_8x8(bool isTransA, bool isTransB, int K, const float *A, int lda, const float *B,
        int ldb, float *C, int ldc, float alpha, float beta,
        int ithr)
{
    if(DEBUG_KERNEL > 0)printf("> kernel_mxn_8x8\n");

    const int m = get_m_unroll_factor();
    //const int n = 8;

    int i = 0;
    while (i < m) {
        size_t vl = __riscv_vsetvl_e32m1(m - i);

        vfloat32m1_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7;

        v_c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            vfloat32m1_t v_a;
            if (isTransA) {  // isTransA == false
                ptrdiff_t stride_a = lda * sizeof(float);
                v_a = __riscv_vlse32_v_f32m1(A + i * lda + k, stride_a, vl);
            } else {
                v_a = __riscv_vle32_v_f32m1(A + i + k * lda, vl);
            }

            const float *b_ptr = isTransB ? &B[k * ldb] : &B[k];
            const int b_stride = isTransB ? 1 : ldb;

            v_c0 = __riscv_vfmacc_vf_f32m1(
                    v_c0, b_ptr[0 * b_stride], v_a, vl);
            v_c1 = __riscv_vfmacc_vf_f32m1(
                    v_c1, b_ptr[1 * b_stride], v_a, vl);
            v_c2 = __riscv_vfmacc_vf_f32m1(
                    v_c2, b_ptr[2 * b_stride], v_a, vl);
            v_c3 = __riscv_vfmacc_vf_f32m1(
                    v_c3, b_ptr[3 * b_stride], v_a, vl);
            v_c4 = __riscv_vfmacc_vf_f32m1(
                    v_c4, b_ptr[4 * b_stride], v_a, vl);
            v_c5 = __riscv_vfmacc_vf_f32m1(
                    v_c5, b_ptr[5 * b_stride], v_a, vl);
            v_c6 = __riscv_vfmacc_vf_f32m1(
                    v_c6, b_ptr[6 * b_stride], v_a, vl);
            v_c7 = __riscv_vfmacc_vf_f32m1(
                    v_c7, b_ptr[7 * b_stride], v_a, vl);
        }

        STORE_C(C + 0 * ldc + i, v_c0, alpha, beta, vl);
        STORE_C(C + 1 * ldc + i, v_c1, alpha, beta, vl);
        STORE_C(C + 2 * ldc + i, v_c2, alpha, beta, vl);
        STORE_C(C + 3 * ldc + i, v_c3, alpha, beta, vl);
        STORE_C(C + 4 * ldc + i, v_c4, alpha, beta, vl);
        STORE_C(C + 5 * ldc + i, v_c5, alpha, beta, vl);
        STORE_C(C + 6 * ldc + i, v_c6, alpha, beta, vl);
        STORE_C(C + 7 * ldc + i, v_c7, alpha, beta, vl);

        i += vl;
    }
}

static void kernel_mxn_16x16(bool isTransA, bool isTransB, int K, const float *A, int lda, const float *B,
        int ldb, float *C, int ldc, float alpha, float beta,
        int ithr)
{
    if(DEBUG_KERNEL > 0)printf("> kernel_mxn_16x16\n");

    const int m = get_m_unroll_factor();
    //const int n = 16;

    int i = 0;
    while (i < m) {
        size_t vl = __riscv_vsetvl_e32m1(m - i);

        vfloat32m1_t v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7;
        vfloat32m1_t v_c8, v_c9, v_c10, v_c11, v_c12, v_c13, v_c14, v_c15;

        v_c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c8 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c9 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c10 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c11 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c12 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c13 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c14 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_c15 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        for (int k = 0; k < K; ++k) {
            vfloat32m1_t v_a;
            if (isTransA) {  // isTransA == false
                ptrdiff_t stride_a = lda * sizeof(float);
                v_a = __riscv_vlse32_v_f32m1(A + i * lda + k, stride_a, vl);
            } else {
                v_a = __riscv_vle32_v_f32m1(A + i + k * lda, vl);
            }

            const float *b_ptr = isTransB ? &B[k * ldb] : &B[k];
            const int b_stride = isTransB ? 1 : ldb;

            v_c0 = __riscv_vfmacc_vf_f32m1(
                    v_c0, b_ptr[0 * b_stride], v_a, vl);
            v_c1 = __riscv_vfmacc_vf_f32m1(
                    v_c1, b_ptr[1 * b_stride], v_a, vl);
            v_c2 = __riscv_vfmacc_vf_f32m1(
                    v_c2, b_ptr[2 * b_stride], v_a, vl);
            v_c3 = __riscv_vfmacc_vf_f32m1(
                    v_c3, b_ptr[3 * b_stride], v_a, vl);
            v_c4 = __riscv_vfmacc_vf_f32m1(
                    v_c4, b_ptr[4 * b_stride], v_a, vl);
            v_c5 = __riscv_vfmacc_vf_f32m1(
                    v_c5, b_ptr[5 * b_stride], v_a, vl);
            v_c6 = __riscv_vfmacc_vf_f32m1(
                    v_c6, b_ptr[6 * b_stride], v_a, vl);
            v_c7 = __riscv_vfmacc_vf_f32m1(
                    v_c7, b_ptr[7 * b_stride], v_a, vl);
            v_c8 = __riscv_vfmacc_vf_f32m1(
                    v_c8, b_ptr[8 * b_stride], v_a, vl);
            v_c9 = __riscv_vfmacc_vf_f32m1(
                    v_c9, b_ptr[9 * b_stride], v_a, vl);
            v_c10 = __riscv_vfmacc_vf_f32m1(
                    v_c10, b_ptr[10 * b_stride], v_a, vl);
            v_c11 = __riscv_vfmacc_vf_f32m1(
                    v_c11, b_ptr[11 * b_stride], v_a, vl);
            v_c12 = __riscv_vfmacc_vf_f32m1(
                    v_c12, b_ptr[12 * b_stride], v_a, vl);
            v_c13 = __riscv_vfmacc_vf_f32m1(
                    v_c13, b_ptr[13 * b_stride], v_a, vl);
            v_c14 = __riscv_vfmacc_vf_f32m1(
                    v_c14, b_ptr[14 * b_stride], v_a, vl);
            v_c15 = __riscv_vfmacc_vf_f32m1(
                    v_c15, b_ptr[15 * b_stride], v_a, vl);
        }

        STORE_C(C + 0 * ldc + i, v_c0, alpha, beta, vl);
        STORE_C(C + 1 * ldc + i, v_c1, alpha, beta, vl);
        STORE_C(C + 2 * ldc + i, v_c2, alpha, beta, vl);
        STORE_C(C + 3 * ldc + i, v_c3, alpha, beta, vl);
        STORE_C(C + 4 * ldc + i, v_c4, alpha, beta, vl);
        STORE_C(C + 5 * ldc + i, v_c5, alpha, beta, vl);
        STORE_C(C + 6 * ldc + i, v_c6, alpha, beta, vl);
        STORE_C(C + 7 * ldc + i, v_c7, alpha, beta, vl);
        STORE_C(C + 8 * ldc + i, v_c8, alpha, beta, vl);
        STORE_C(C + 9 * ldc + i, v_c9, alpha, beta, vl);
        STORE_C(C + 10 * ldc + i, v_c10, alpha, beta, vl);
        STORE_C(C + 11 * ldc + i, v_c11, alpha, beta, vl);
        STORE_C(C + 12 * ldc + i, v_c12, alpha, beta, vl);
        STORE_C(C + 13 * ldc + i, v_c13, alpha, beta, vl);
        STORE_C(C + 14 * ldc + i, v_c14, alpha, beta, vl);
        STORE_C(C + 15 * ldc + i, v_c15, alpha, beta, vl);

        i += vl;
    } 
}

void kernel_mxn(bool isTransA, bool isTransB, int K, const float *A, const int lda, const float *B,
        const int ldb, float *C, const int ldc, const float alpha,
        const float beta, int ithr) 
{
    if(DEBUG_KERNEL > 0) printf("> kernel_mxn: alpha:%f  beta:%f\n", alpha, beta);
    int n_unroll = get_n_unroll_factor();

    switch (n_unroll) {
        case 2:
            kernel_mxn_2x2(
                    isTransA, isTransB, K, A, lda, B, ldb, C, ldc, alpha, beta, ithr);
            break;
        case 4:
            kernel_mxn_4x4(
                    isTransA, isTransB, K, A, lda, B, ldb, C, ldc, alpha, beta, ithr);
            break;
        case 8:
            kernel_mxn_8x8(
                    isTransA, isTransB, K, A, lda, B, ldb, C, ldc, alpha, beta, ithr);
            break;
        case 16:
            kernel_mxn_16x16(
                    isTransA, isTransB, K, A, lda, B, ldb, C, ldc, alpha, beta, ithr);
            break;
        default:
            kernel_mxn_2x2(
                    isTransA, isTransB, K, A, lda, B, ldb, C, ldc, alpha, beta, ithr);
    }
}

void block_ker(bool isTransA, bool isTransB, const int M, const int N, const int K, const float *A,
        const int lda, const float *B, const int ldb, float *C,
        const int ldc, const float alpha, const float beta, float *ws,
        bool do_copy, int ithr) 
{

    const int n_unroll = get_n_unroll_factor();
    const int m_unroll = get_m_unroll_factor();

    int Nu = rnd_dn(N, n_unroll);
    int Mu = rnd_dn(M, m_unroll);

    if(DEBUG_KERNEL > 0)
    printf("> block_ker: n_unroll:%d   m_unroll:%d        Nu:%d   Mu:%d\n",n_unroll, m_unroll, Nu, Mu);

    if(DEBUG_KERNEL > 0){
        if( ws != NULL ){
            printf("Print WS matrix\n");
            //print_matrixf32(ws, K, lda, 0);
            print_lmatrixf32(ws, m_unroll, K);
        }
    }

    // no JIT
    //const bool use_jit_ker = !isTransA && !isTransB && (n_unroll == 4)
    //        && (m_unroll == 8 || m_unroll == 16);

    // also JIT branch disabled ...

    if (do_copy) {
        if(DEBUG_KERNEL > 0)printf("branch DO_COPY=TRUE\n");
        for (int i = 0; i < Mu; i += m_unroll) {
            for (int j = 0; j < Nu; j += n_unroll) {
                const float *b = isTransB ? &B[j] : &B[j * ldb];
                const float *a = isTransA ? &A[i * lda] : &A[i];

                if (j == 0) {
                    if(DEBUG_KERNEL > 0)
                    printf("copy_A(isTransA:%s, K:%d, a:&A[%ld], lda:%d, ws)\n", 
                        isTransA ? "TRUE\0" : "FALSE\0",
                        K, (a - A), lda
                    );

                    copy_A(isTransA, K, a, lda, ws); 

                    if(DEBUG_KERNEL > 0){
                        printf("Print WS matrix (after copy)\n");
                        //print_matrixf32(ws, K, lda, 0);
                        print_lmatrixf32(ws, m_unroll, m_unroll * K);
                    }
                }  

                if(DEBUG_KERNEL > 0)
                printf("KERNEL_MXN(false, isTransB:%s, K:%d, ws, m_unroll:%d, b:&B[%ld], ldb:%d, &C[%d + %d * %d], ldc:%d, alpha:%f, beta:%f, ithr:%d)\n",
                    isTransB ? "TRUE\0" : "FALSE\0",
                    K, m_unroll,
                    (b - B), ldb,
                    i, j, ldc,  ldc,
                    alpha, beta,
                    ithr
                );

                kernel_mxn(false, isTransB, K, ws, m_unroll, b, ldb, &C[i + j * ldc], ldc, alpha, beta, ithr);
                
                if(DEBUG_KERNEL > 0){
                    printf("Print C matrix\n");
                    //print_matrixf32(C, K, K, 0);
                    print_lmatrixf32(C, ldc, ldc * N);
                }
            }
        }
    } else {
        if(DEBUG_KERNEL > 0)printf("branch DO_COPY=FALSE\n");
        for (int i = 0; i < Mu; i += m_unroll) {
            for (int j = 0; j < Nu; j += n_unroll) {
                const float *b = isTransB ? &B[j] : &B[j * ldb];
                const float *a = isTransA ? &A[i * lda] : &A[i];

                if(DEBUG_KERNEL > 0)
                printf("KERNEL_MXN(isTransA:%s, isTransB:%s, K:%d, a:&A[%ld], lda:%d, b:&B[%ld], ldb:%d, &C[%d + %d * %d], ldc:%d, alpha:%f, beta:%f, ithr:%d)\n",
                    isTransA ? "TRUE\0" : "FALSE\0",
                    isTransB ? "TRUE\0" : "FALSE\0",
                    K, (a - A), lda,
                    (b - B), ldb,
                    i, j, ldc,  ldc,
                    alpha, beta,
                    ithr
                );

                kernel_mxn(isTransA, isTransB, K, a, lda, b, ldb, &C[i + j * ldc], ldc, alpha, beta, ithr);
            }
        }
    }


    // tail processing: columns Nu to N (vectorized over i for contiguous C access)
    // Process all M rows for the remaining (N-Nu) columns
    for (int j = Nu; j < N; j++) {
        float *c_ptr = &C[j * ldc];
        const float *b_col = isTransB ? &B[j] : &B[j * ldb];

        int i = 0;
        while (i < M) {
            size_t vl = __riscv_vsetvl_e32m4(M - i);
            vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (int p = 0; p < K; p++) {
                float b_val = isTransB ? b_col[p * ldb] : b_col[p];
                vfloat32m4_t v_a;

                if (isTransA) {  // no TransA
                    // A(p, i:i+vl) - strided access
                    ptrdiff_t stride_a = lda * sizeof(float);
                    v_a = __riscv_vlse32_v_f32m4(&A[p + i * lda], stride_a, vl);
                } else {
                    // A(i:i+vl, p) - contiguous access
                    v_a = __riscv_vle32_v_f32m4(&A[i + p * lda], vl);
                }
                v_acc = __riscv_vfmacc_vf_f32m4(v_acc, b_val, v_a, vl);
            }

            // disabling alpha e beta
            // Apply alpha and beta, store result (contiguous)  
            //v_acc = __riscv_vfmul_vf_f32m4(v_acc, alpha, vl);
            //if (beta != 0.0f) {
            //    vfloat32m4_t v_c_old = __riscv_vle32_v_f32m4(c_ptr + i, vl);
            //    v_acc = __riscv_vfmacc_vf_f32m4(v_acc, beta, v_c_old, vl);
            //}

            __riscv_vse32_v_f32m4(c_ptr + i, v_acc, vl);
            i += vl;
        }
    }


    // tail processing: rows Mu to M (vectorized over i for contiguous C access)
    // Process remaining (M-Mu) rows for the first Nu columns
    if (Mu < M) {
        const int m_tail = M - Mu;

        for (int j = 0; j < Nu; j++) {
            float *c_ptr = &C[Mu + j * ldc];
            const float *b_col = isTransB ? &B[j] : &B[j * ldb];

            int i = 0;
            while (i < m_tail) {
                size_t vl = __riscv_vsetvl_e32m4(m_tail - i);
                vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                for (int p = 0; p < K; p++) {
                    float b_val = b_col[p];  // no TransB
                    vfloat32m4_t v_a;
                    if (isTransA) { // no TransA
                        // A(p, Mu+i:Mu+i+vl) - strided access
                        ptrdiff_t stride_a = lda * sizeof(float);
                        v_a = __riscv_vlse32_v_f32m4(
                                &A[p + (Mu + i) * lda], stride_a, vl);
                    } else {
                        // A(Mu+i:Mu+i+vl, p) - contiguous access
                        v_a = __riscv_vle32_v_f32m4(&A[Mu + i + p * lda], vl);
                    }
                    v_acc = __riscv_vfmacc_vf_f32m4(v_acc, b_val, v_a, vl);
                }

                // disabling alpha e beta
                // Apply alpha and beta, store result (contiguous)
                //v_acc = __riscv_vfmul_vf_f32m4(v_acc, alpha, vl);
                //if (beta != 0.0f) {
                //    vfloat32m4_t v_c_old = __riscv_vle32_v_f32m4(c_ptr + i, vl);
                //    v_acc = __riscv_vfmacc_vf_f32m4(v_acc, beta, v_c_old, vl);
                //}

                __riscv_vse32_v_f32m4(c_ptr + i, v_acc, vl);
                i += vl;
            }
        }
    }


}


void gemm_ithr(bool isTransA, bool isTransB, const int M, const int N, const int K, const float alpha,
        const float *A, const int lda, const float *B, const int ldb,
        const float beta, float *C, const int ldc, bool do_copy, float *ws,
        int ithr) 
{


    const int BM = 4032;
    const int BN = 256;
    const int BK = 256;


    const float *curA;
    const float *curB;
    float *curC;

    if ((M <= 0) || (N <= 0)) return;

    if ((K <= 0) || (alpha == 0.f)) {
        int MN = N * M;
        //if (beta == 0.f) {             /// forever, beta == 0
            for (int j = 0; j < MN; j++)
                C[j] = 0.f;
        //} else if (beta != 1.f) {       ///  never, beta == 0
        //    for (int j = 0; j < MN; j++)
        //        C[j] *= beta;
        //}
        return;
    }

    for (int Bk = 0; Bk < K; Bk += BK) {
        int kb = MIN(K - Bk, BK);
        for (int Bm = 0; Bm < M; Bm += BM) {
            int mb = MIN(M - Bm, BM);
            for (int Bn = 0; Bn < N; Bn += BN) {
                int nb = MIN(N - Bn, BN);
                
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                curB = isTransB ? B + Bn + Bk * ldb : B + Bk + Bn * ldb;
                curC = C + Bm + Bn * ldc;

                // only in the first stage, pass beta otherwise pass beta=1 ...
                
                /*
                if (Bk == 0) {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, beta, ws, do_copy, ithr);
                } else {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, 1.0f, ws, do_copy, ithr);
                }
                */

                // SO: same invocation, because alpha=1 and beta=0 ---> unique branch

                if(DEBUG_KERNEL > 0)
                printf("BLOCK_KER(isTransA:%s, isTransB:%s, mb:%d, nb:%d, kb:%d, curA:&A[%ld], lda:%d, curB:&B[%ld], ldb:%d, curC:&C[%ld], ldc:%d, alpha:%f, beta:%f, ws, do_copy:%s, ithr:%d)\n",
                    isTransA ? "TRUE\0" : "FALSE\0", 
                    isTransB ? "TRUE\0" : "FALSE\0",
                    mb, nb, kb,
                    (curA - A), lda,
                    (curB - B), ldb,
                    (curC - C), ldc,
                    alpha, beta,
                    do_copy ? "TRUE\0" : "FALSE\0",
                    ithr
                );

                block_ker(isTransA, isTransB, mb, nb, kb, curA, lda, curB,ldb, curC, ldc, alpha, beta, ws, do_copy, ithr);


            }
        }
    }

}


void multiply(
    bool isTransA, bool isTransB,
    const float* _A,
    const float* _B,
    float* C,
    int _N
) {

    // Perform matrix multiplication (GEMM) with operands swapped
    const float* A = _B;
    const float* B = _A;

    // Let op(X) be such that if transx == 'N', then op(X) = X;
    // otherwise, if transx == 'T', then op(X) = X^T.
    // M: number of rows of C --> number of rows of op(A)
    // N: number of columns of C --> number of columns of op(B)
    // K: common dimension, i.e., the length of the summation index in the matrix multiplication
    // ld_x: leading dimension of matrix X — the number of elements between two consecutive columns
    // So, A is [M x K], B is [K x N], and ldX = N.
    //
    // Alpha and beta are disabled,
    // since C = alpha * (A * B) + beta * C
    // 1. by definition, alpha = 1 and beta = 0
    // 2. where possible, branches based on alpha and beta are disabled
    int N = _N, M = N, K = N;
    int lda = N, ldb = N, ldc = N;
    float alpha = 1.0, beta = 0.0;

    // calc_nthr_nocopy_rvv --> Quick exit for single thread.
    int MB = M, NB = N, KB = K;
    int nthr_m = 1, nthr_n = 1, nthr_k = 1;

    float *c_buffers = NULL;
    float *ws_buffers = NULL;


    bool do_copy = (NB / get_n_unroll_factor() > 3);
    const int nthr_mn = nthr_m * nthr_n;       // --> 1
    const int nthr_to_use = nthr_mn * nthr_k;  // --> 1

    const size_t ws_elems_per_thr = K * get_m_unroll_factor(); // --> N * 16
    const size_t ws_size_per_thr = rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);

    if (do_copy) {
        ws_buffers = (float *)aligned_alloc(nthr_to_use * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers) do_copy = false;
    }

    //parallel(nthr_to_use, [&](int ithr, int nthr)
    // nthr_to_use = 1     ithr = 0 (idx thread)      nthr = 1 (tot threads)
    ///int nthr_to_use = 1, ithr = 0, nthr = 1;
    int ithr = 0, nthr = 1;


    int ithr_mn = ithr % nthr_mn;   // --> 0
    int ithr_m = ithr_mn % nthr_m;  // --> 0
    int ithr_n = ithr_mn / nthr_m;  // --> 0
    int ithr_k = ithr / nthr_mn;    // --> 0

    int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

    float *ws = do_copy
        ? ws_buffers + ithr * ws_size_per_thr / sizeof(float)
        : NULL;

    int m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0,
            k_from = 0, k_to = 0, myK = 0;
    
    get_thr_block(&m_from, &m_to, &myM, MB, M, ithr_m);
    get_thr_block(&n_from, &n_to, &myN, NB, N, ithr_n);
    get_thr_block(&k_from, &k_to, &myK, KB, K, ithr_k);


    //if (myM > 0 && myN > 0) {   --> true
    float myBeta, *myC;
    int ld;
    //if (ithr_k == 0) {  --> true
    myC = &(C[m_from + n_from * ldc]);
    myBeta = beta;
    ld = ldc;
    /*
    } else {
        myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
        myBeta = 0.0f;
        ld = MB;
    }
    */


    
    const float *myA = isTransA ? &(A[k_from + m_from * lda])
                                : &(A[m_from + k_from * lda]);
    const float *myB = isTransB ? &(B[n_from + k_from * ldb])
                                : &(B[k_from + n_from * ldb]);
    /*
    if (!isTransA) {
        if (!isTransB) {
            gemm_ithr<false, false>(myM, myN, myK, alpha, myA, lda, myB,
                    ldb, myBeta, myC, ld, do_copy, ws, ithr);
        } else {
            gemm_ithr<false, true>(myM, myN, myK, alpha, myA, lda, myB,
                    ldb, myBeta, myC, ld, do_copy, ws, ithr);
        }
    } else {
        if (!isTransB) {
            gemm_ithr<true, false>(myM, myN, myK, alpha, myA, lda, myB,
                    ldb, myBeta, myC, ld, do_copy, ws, ithr);
        } else {
            gemm_ithr<true, true>(myM, myN, myK, alpha, myA, lda, myB,
                    ldb, myBeta, myC, ld, do_copy, ws, ithr);
        }
    }

    */
    
    if(DEBUG_KERNEL > 0)
    printf("GEMM_ITHR(isTransA:%s, isTransB:%s, myM:%d, myN:%d, myK:%d, alpha:%f, myA:&A[%ld], lda:%d, myB:&B[%ld], ldb:%d, myBeta:%f, myC:&C[%ld], ld:%d, do_copy:%s, ws, ithr:%d )\n",
        isTransA ? "TRUE\0" : "FALSE\0", 
        isTransB ? "TRUE\0" : "FALSE\0",        
        myM, myN, myK, 
        alpha, 
        (A - myA), lda, 
        (B - myB), ldb,
        myBeta,
        (C - myC), ld,
        do_copy ? "TRUE\0" : "FALSE\0",
        ithr
    );

    gemm_ithr(isTransA, isTransB, myM, myN, myK, alpha, myA, lda, myB, ldb, myBeta, myC, ld, do_copy, ws, ithr);

    /// GEMM_ITHR(myM:1, myN:1, myK:1, alpha:1.000000, myA:&A[0], lda:8, myB:&B[0], ldb:8, myBeta:0.000000, myC:&C[0], ld:8, do_copy:FALSE, ws:0, ithr:0 )


    ///}

}

int main(int argc, char* argv[]) {

    printf("Testing matrix ");

    int size = SIZE;
    int input_case = DEBUG_INPUT_FLAG;
    bool isTransA=false, isTransB=false;

    if (argc == 2) {
        size = atoi( argv[1] );
    }
    if (argc == 4){
        size = atoi( argv[1] );
        isTransA = atoi( argv[2] ) == 1 ? true : false;
        isTransB = atoi( argv[3] ) == 1 ? true : false;
    }
    if (argc == 5){
        size = atoi( argv[1] );
        isTransA = atoi( argv[2] ) == 1 ? true : false;
        isTransB = atoi( argv[3] ) == 1 ? true : false;
        input_case = atoi( argv[4] );
    }

    printf("size: %d x %d    input_case: %d\n", size, size, input_case );
    printf("isTransA:%s   isTransB:%s\n", 
        isTransA ? "TRUE\0" : "FALSE\0", 
        isTransB ? "TRUE\0" : "FALSE\0"   
    );
    ///printf("Num proc: %d\n", omp_get_max_threads());
    
    // Allocate memory for matrices
    float *A = (float*)malloc(size * size * sizeof(float));
    float *B = (float*)malloc(size * size * sizeof(float));
    float *C = (float*)malloc(size * size * sizeof(float));

    // Init matrix values pseudorandom

    // set initial seed for rand, 1 if debug-mode
    srand( RANDOM == 0 ? 1 : time(NULL)  );

    for(int i = 0; i < size * size; i++ ){

        // Identity matrix for both A and B --> C = I * I = I
        if( input_case == 1 ){ 
            int row = i / size;
            int col = i % size;
            A[i] = (row == col) ? 1.0f : 0.0f;
            B[i] = (row == col) ? 1.0f : 0.0f;
        }

        // All-ones matrices: A = B = ones --> each element of C = sum_{k=1}^{size} 1*1 = size
        else if( input_case == 2 ){
            A[i] = 1.0f;
            B[i] = 1.0f;
        }

        // A has constant rows (A[i,j] = i+1), B has constant columns (B[i,j] = j+1)
        // Resulting C[i,j] = sum_{k=1}^{size} (i+1)*(k+1) = (i+1) * sum_{k=1}^{size}(k+1)
        else if( input_case == 3 ){
            int row = i / size;
            int col = i % size;
            A[i] = (float)(row + 1);
            B[i] = (float)(col + 1);
        }

        // A = arbitrary matrix (filled with linear index), B = identity --> C = A * I = A
        else if( input_case == 4 ){ // X x ID -> X
            int row = i / size;
            int col = i % size;
            A[i] = i;
            B[i] = (row == col) ? 1.0f : 0.0f;
        }

        else if( input_case == 5 ){ // ID x index -> X
            int row = i / size;
            int col = i % size;
            A[i] = (row == col) ? 1.0f : 0.0f;
            B[i] = i;
        }

        // Default: random integer matrices in [1, 10]
        else{
            A[i] = rand() % 10 + 1;
            B[i] = rand() % 10 + 1;           
        }
        C[i] = 0.0; 
    }

    if(DEBUG_PRINT_IO){
        printf("A");
        print_matrixf32(A, size, size, 0);
        printf("B");
        print_matrixf32(B, size, size, 0);
    }

    // Start timer
    //double start_time = omp_get_wtime();
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM) -- invertito!
    multiply(isTransA, isTransB, A, B, C, size);

    // Stop timer
    //double end_time = omp_get_wtime();
    clock_t end_time = clock();

    if(DEBUG_PRINT_IO){
        printf("C");
        print_matrixf32(C, size, size, 0);
    }

    // Calculate and print execution time
    //double execution_time = (end_time - start_time); // [sec]
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : onednn_rvv_gemm_f32_v2, %f, %d\n", execution_time, size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
