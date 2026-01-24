#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include "utils.h"

#define DEBUG_FLAG 0
#define DEBUG_PRINT_IO 1
#define SIZE 8

// TILE SIZE SETTING
// 0   -> auto
// {i} -> force select ixi tile
#define DEFAULT_TILE_SIZE 0

// k-BLOCK SIZE SETTING
// 0   -> auto
// {i} -> force select k-size block
#define K_BLOCK_SIZE 0


static inline void rvv_copy_f32_m4(float *dst, const float *src, int n)
{
    int i = 0;

    while (i < n) {
        // set VL dinamico, LMUL=4
        size_t vl = __riscv_vsetvl_e32m4(n - i);

        // load vettoriale
        vfloat32m4_t v = __riscv_vle32_v_f32m4(&src[i], vl);

        // store vettoriale
        __riscv_vse32_v_f32m4(&dst[i], v, vl);

        i += vl;
    }
}

//TODO ...
// Function to perform matrix multiplication
void multiply_gemm_2x2(float* mat1, float* mat2, float* res, const int size, const int k_block) {

    IFDEBUG{
        printf("kernel> KERNEL_2x2 [size=%d, k_block=%d]\n", size, k_block);
    }

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

/// MOD v3
void multiply_gemm_4x4_k_tiled_packedB(
    float* mat1,    // A (row-major)
    float* mat2,    // B (row-major)
    float* res,     // C (row-major, inizializzata a zero)
    const int size,
    const int k_block)
{
    IFDEBUG{
        printf("kernel> KERNEL_4x4 [size=%d, k_block=%d]\n", size, k_block);
    }

    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    const int MR = 4;   // blocco righe A
    const int NR = 4;   // blocco colonne B
    const int K_BLOCK = k_block;

    // j-blocking livello L2/L3 (riuso di B)
    const int J_BLOCK = 128;

    // pannello grande di B riutilizzabile
    float* B_panel_L2 = malloc(sizeof(float) * size * J_BLOCK);

    // pannelli microkernel
    float (*A_panel)[K_BLOCK] = malloc(sizeof(float) * MR * K_BLOCK);
    float (*B_panel)[NR]      = malloc(sizeof(float) * K_BLOCK * NR);

    // ---- loop su pannelli di colonne (riuso di B) ----
    for (int jb = 0; jb < size; jb += J_BLOCK) {

        int jb_size = (jb + J_BLOCK < size) ? J_BLOCK : (size - jb);

        // ---- packing grande pannello B una sola volta ----
        for (int k = 0; k < size; ++k) {
            memcpy(&B_panel_L2[k * jb_size],
                   &B[k][jb],
                   sizeof(float) * jb_size);
        }

        // ---- blocchi su righe di A ----
        for (int ih = 0; ih < size; ih += MR) {

            // ---- blocchi interni nel pannello jb ----
            for (int jh = 0; jh < jb_size; jh += NR) {

                size_t vl = __riscv_vsetvl_e32m1(NR);

                // accumulatori vettoriali per 4 righe
                vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                // ---- loop su k bloccato ----
                for (int kb = 0; kb < size; kb += K_BLOCK) {

                    int k_end = (kb + K_BLOCK < size) ? kb + K_BLOCK : size;
                    int k_blk = k_end - kb;

                    // ---- packing A contiguo ----
                    for (int r = 0; r < MR; ++r) {
                        memcpy(&A_panel[r][0],
                               &A[ih + r][kb],
                               sizeof(float) * k_blk);
                    }

                    // ---- packing B dal pannello L2 ----
                    for (int k = 0; k < k_blk; ++k) {
                        memcpy(&B_panel[k][0],
                               &B_panel_L2[(kb + k) * jb_size + jh],
                               sizeof(float) * NR);
                    }

                    // ---- microkernel 4×4 vettoriale ----
                    for (int k = 0; k < k_blk; ++k) {

                        vfloat32m1_t b_vec =
                            __riscv_vle32_v_f32m1(&B_panel[k][0], vl);

                        float a0 = A_panel[0][k];
                        float a1 = A_panel[1][k];
                        float a2 = A_panel[2][k];
                        float a3 = A_panel[3][k];

                        c0 = __riscv_vfmacc_vf_f32m1(c0, a0, b_vec, vl);
                        c1 = __riscv_vfmacc_vf_f32m1(c1, a1, b_vec, vl);
                        c2 = __riscv_vfmacc_vf_f32m1(c2, a2, b_vec, vl);
                        c3 = __riscv_vfmacc_vf_f32m1(c3, a3, b_vec, vl);
                    }
                }

                // ---- store finale 4×4 ----
                __riscv_vse32_v_f32m1(&C[ih + 0][jb + jh], c0, vl);
                __riscv_vse32_v_f32m1(&C[ih + 1][jb + jh], c1, vl);
                __riscv_vse32_v_f32m1(&C[ih + 2][jb + jh], c2, vl);
                __riscv_vse32_v_f32m1(&C[ih + 3][jb + jh], c3, vl);
            }
        }
    }

    free(A_panel);
    free(B_panel);
    free(B_panel_L2);
}



/// MOD v2
void multiply_gemm_8x8_k_tiled_packedB(
    float* mat1,    // A row-major
    float* mat2,    // B row-major
    float* res,     // C row-major
    const int size,
    const int k_block)
{
    IFDEBUG{
        printf("kernel> KERNEL_8x8 [size=%d, k_block=%d]\n", size, k_block);
    }
    
    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    const int MR = 8;
    const int NR = 8;

    const int K_BLOCK = k_block;

    // ---- nuovo: j-blocking livello L2 ----
    const int J_BLOCK = 128;

    // pannello grande riusabile di B
    float* B_panel_L2 = malloc(sizeof(float) * size * J_BLOCK);        /// K_BLOCK * J_BLOCK

    // pannelli A e B per microkernel
    float (*A_panel)[K_BLOCK] = malloc(sizeof(float) * MR * K_BLOCK);  ///  MR x K_BLOCK
    float (*B_panel)[NR]      = malloc(sizeof(float) * K_BLOCK * NR);  ///  K_BLOCK x NR

    // ---- loop su blocchi di colonne per riuso di B ----
    for (int jb = 0; jb < size; jb += J_BLOCK) {

        int jb_size = (jb + J_BLOCK < size) ? J_BLOCK : (size - jb);

        // ---- pack grande pannello B una sola volta ----
        for (int k = 0; k < size; ++k) {
            /*
            memcpy(&B_panel_L2[k * jb_size],
                   &B[k][jb],
                   sizeof(float) * jb_size);
            */
            rvv_copy_f32_m4(&B_panel_L2[k * jb_size], &B[k][jb], jb_size);
        }

        if( DEBUG_FLAG > 0 ){
            printf("B_panel_L2 ");
            //print_matrixf32(B_panel_L2, size, J_BLOCK, 0);
            print_lmatrixf32(B_panel_L2, size, size * J_BLOCK);
        }

        // ---- blocchi su righe di A ----
        for (int ih = 0; ih < size; ih += MR) {

            // blocchi dentro il pannello jb
            for (int jh = 0; jh < jb_size; jh += NR) {

                size_t vl = __riscv_vsetvl_e32m1(NR);

                // accumulatori 8 vettori
                vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vfloat32m1_t c7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

                // ---- loop K con blocking ----
                for (int kb = 0; kb < size; kb += K_BLOCK) {

                    int k_end = (kb + K_BLOCK < size) ? kb + K_BLOCK : size;
                    int k_blk = k_end - kb;

                    // --- pack A contiguo ---
                    for (int r = 0; r < MR; ++r) {
                        /*
                        memcpy(&A_panel[r][0],
                               &A[ih + r][kb],
                               sizeof(float) * k_blk);
                        */
                        rvv_copy_f32_m4(&A_panel[r][0], &A[ih + r][kb],k_blk);
                    }

                    if( DEBUG_FLAG > 0 ){
                        printf("A_panel ");
                        print_lmatrixf32((float*)A_panel, K_BLOCK, MR * K_BLOCK);
                    }

                    // --- pack B da pannello L2 ---
                    for (int k = 0; k < k_blk; ++k) {
                        /*
                        memcpy(&B_panel[k][0],
                               &B_panel_L2[(kb + k) * jb_size + jh],
                               sizeof(float) * NR);
                        */
                        rvv_copy_f32_m4(&B_panel[k][0],&B_panel_L2[(kb + k) * jb_size + jh],NR);
                    }

                    if( DEBUG_FLAG > 0 ){
                        printf("B_panel ");
                        print_lmatrixf32((float*)B_panel, NR, K_BLOCK * NR);
                    }

                    // ---- calcolo vettoriale ----
                    for (int k = 0; k < k_blk; ++k) {

                        vfloat32m1_t b_vec =
                            __riscv_vle32_v_f32m1(&B_panel[k][0], vl);

                        float a0 = A_panel[0][k];
                        float a1 = A_panel[1][k];
                        float a2 = A_panel[2][k];
                        float a3 = A_panel[3][k];
                        float a4 = A_panel[4][k];
                        float a5 = A_panel[5][k];
                        float a6 = A_panel[6][k];
                        float a7 = A_panel[7][k];

                        c0 = __riscv_vfmacc_vf_f32m1(c0, a0, b_vec, vl);
                        c1 = __riscv_vfmacc_vf_f32m1(c1, a1, b_vec, vl);
                        c2 = __riscv_vfmacc_vf_f32m1(c2, a2, b_vec, vl);
                        c3 = __riscv_vfmacc_vf_f32m1(c3, a3, b_vec, vl);
                        c4 = __riscv_vfmacc_vf_f32m1(c4, a4, b_vec, vl);
                        c5 = __riscv_vfmacc_vf_f32m1(c5, a5, b_vec, vl);
                        c6 = __riscv_vfmacc_vf_f32m1(c6, a6, b_vec, vl);
                        c7 = __riscv_vfmacc_vf_f32m1(c7, a7, b_vec, vl);
                    }
                }

                // ---- scrittura finale ----
                __riscv_vse32_v_f32m1(&C[ih + 0][jb + jh], c0, vl);
                __riscv_vse32_v_f32m1(&C[ih + 1][jb + jh], c1, vl);
                __riscv_vse32_v_f32m1(&C[ih + 2][jb + jh], c2, vl);
                __riscv_vse32_v_f32m1(&C[ih + 3][jb + jh], c3, vl);
                __riscv_vse32_v_f32m1(&C[ih + 4][jb + jh], c4, vl);
                __riscv_vse32_v_f32m1(&C[ih + 5][jb + jh], c5, vl);
                __riscv_vse32_v_f32m1(&C[ih + 6][jb + jh], c6, vl);
                __riscv_vse32_v_f32m1(&C[ih + 7][jb + jh], c7, vl);
            }
        }
    }

    free(A_panel);
    free(B_panel);
    free(B_panel_L2);
}



/// MOD v2
void multiply_gemm_16x16_k_tiled_packedB(
    float* mat1,    // A (row-major)
    float* mat2,    // B (row-major)
    float* res,     // C (row-major, inizializzata a zero)
    const int size,
    const int k_block)
{
    IFDEBUG{
        printf("kernel> KERNEL_16x16 [size=%d, k_block=%d]\n", size, k_block);
    }

    float (*A)[size] = (float (*)[size]) mat1;
    float (*B)[size] = (float (*)[size]) mat2;
    float (*C)[size] = (float (*)[size]) res;

    const int MR = 16;  // righe microkernel
    const int NR = 16;  // colonne microkernel
    const int K_BLOCK = k_block;

    // j-blocking per aumentare riuso di B
    const int J_BLOCK = 128;

    // pannello L2 per B (riutilizzato tra più blocchi di A)
    float* B_panel_L2 = malloc(sizeof(float) * size * J_BLOCK);

    // pannelli microkernel
    float (*A_panel)[K_BLOCK] = malloc(sizeof(float) * MR * K_BLOCK);
    float (*B_panel)[NR]      = malloc(sizeof(float) * K_BLOCK * NR);

    for (int jb = 0; jb < size; jb += J_BLOCK) {

        int jb_size = (jb + J_BLOCK < size) ? J_BLOCK : (size - jb);

        // -----------------------------
        // Packing pannello grande di B
        // -----------------------------
        for (int k = 0; k < size; ++k) {
            memcpy(&B_panel_L2[k * jb_size],
                   &B[k][jb],
                   sizeof(float) * jb_size);
        }

        // -----------------------------
        // Loop sui blocchi di A
        // -----------------------------
        for (int ih = 0; ih < size; ih += MR) {

            for (int jh = 0; jh < jb_size; jh += NR) {

                size_t vl = __riscv_vsetvl_e32m1(NR);

                // accumulatori 16 righe × 16 colonne (vettoriali)
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

                // -----------------------------
                // K blocking
                // -----------------------------
                for (int kb = 0; kb < size; kb += K_BLOCK) {

                    int k_end = (kb + K_BLOCK < size) ? kb + K_BLOCK : size;
                    int k_blk = k_end - kb;

                    // ---- packing A (16 × k_blk) ----
                    for (int r = 0; r < MR; ++r) {
                        memcpy(&A_panel[r][0],
                               &A[ih + r][kb],
                               sizeof(float) * k_blk);
                    }

                    // ---- packing B (k_blk × 16) ----
                    for (int k = 0; k < k_blk; ++k) {
                        memcpy(&B_panel[k][0],
                               &B_panel_L2[(kb + k) * jb_size + jh],
                               sizeof(float) * NR);
                    }

                    // ---- microkernel 16×16 ----
                    for (int k = 0; k < k_blk; ++k) {

                        vfloat32m1_t b_vec =
                            __riscv_vle32_v_f32m1(&B_panel[k][0], vl);

                        // scalari da A
                        float a0  = A_panel[0][k];
                        float a1  = A_panel[1][k];
                        float a2  = A_panel[2][k];
                        float a3  = A_panel[3][k];
                        float a4  = A_panel[4][k];
                        float a5  = A_panel[5][k];
                        float a6  = A_panel[6][k];
                        float a7  = A_panel[7][k];
                        float a8  = A_panel[8][k];
                        float a9  = A_panel[9][k];
                        float a10 = A_panel[10][k];
                        float a11 = A_panel[11][k];
                        float a12 = A_panel[12][k];
                        float a13 = A_panel[13][k];
                        float a14 = A_panel[14][k];
                        float a15 = A_panel[15][k];

                        // FMA vector × scalar
                        c0  = __riscv_vfmacc_vf_f32m1(c0,  a0,  b_vec, vl);
                        c1  = __riscv_vfmacc_vf_f32m1(c1,  a1,  b_vec, vl);
                        c2  = __riscv_vfmacc_vf_f32m1(c2,  a2,  b_vec, vl);
                        c3  = __riscv_vfmacc_vf_f32m1(c3,  a3,  b_vec, vl);
                        c4  = __riscv_vfmacc_vf_f32m1(c4,  a4,  b_vec, vl);
                        c5  = __riscv_vfmacc_vf_f32m1(c5,  a5,  b_vec, vl);
                        c6  = __riscv_vfmacc_vf_f32m1(c6,  a6,  b_vec, vl);
                        c7  = __riscv_vfmacc_vf_f32m1(c7,  a7,  b_vec, vl);
                        c8  = __riscv_vfmacc_vf_f32m1(c8,  a8,  b_vec, vl);
                        c9  = __riscv_vfmacc_vf_f32m1(c9,  a9,  b_vec, vl);
                        c10 = __riscv_vfmacc_vf_f32m1(c10, a10, b_vec, vl);
                        c11 = __riscv_vfmacc_vf_f32m1(c11, a11, b_vec, vl);
                        c12 = __riscv_vfmacc_vf_f32m1(c12, a12, b_vec, vl);
                        c13 = __riscv_vfmacc_vf_f32m1(c13, a13, b_vec, vl);
                        c14 = __riscv_vfmacc_vf_f32m1(c14, a14, b_vec, vl);
                        c15 = __riscv_vfmacc_vf_f32m1(c15, a15, b_vec, vl);
                    }
                }

                // -----------------------------
                // store finale 16×16
                // -----------------------------
                __riscv_vse32_v_f32m1(&C[ih +  0][jb + jh], c0,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  1][jb + jh], c1,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  2][jb + jh], c2,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  3][jb + jh], c3,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  4][jb + jh], c4,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  5][jb + jh], c5,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  6][jb + jh], c6,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  7][jb + jh], c7,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  8][jb + jh], c8,  vl);
                __riscv_vse32_v_f32m1(&C[ih +  9][jb + jh], c9,  vl);
                __riscv_vse32_v_f32m1(&C[ih + 10][jb + jh], c10, vl);
                __riscv_vse32_v_f32m1(&C[ih + 11][jb + jh], c11, vl);
                __riscv_vse32_v_f32m1(&C[ih + 12][jb + jh], c12, vl);
                __riscv_vse32_v_f32m1(&C[ih + 13][jb + jh], c13, vl);
                __riscv_vse32_v_f32m1(&C[ih + 14][jb + jh], c14, vl);
                __riscv_vse32_v_f32m1(&C[ih + 15][jb + jh], c15, vl);
            }
        }
    }

    free(A_panel);
    free(B_panel);
    free(B_panel_L2);
}


// ts: tile size {auto:0, 2, 4, 8, 16}  
void multiply_gemm(float* A, float* B, float* C, int N, int ts, int bs) {


    /// kernel tile size selection
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

    /// k-block size selection

    if( bs == 0 ){
        if (N >= 1024) {
            bs = 64;      // matrici grandi → meno miss
        } else if (N >= 512) {
            bs = 96;      // medio-grandi
        } else {
            bs = 128;     // matrici piccole → massimo riuso in cache
        } 
    }

    IFDEBUG{
        printf("kernel> ts=%d, bs=%d\n", ts, bs);
    }

    switch (ts) {
        case 16: multiply_gemm_16x16_k_tiled_packedB(A, B, C, N, bs); break;
        case 8:  multiply_gemm_8x8_k_tiled_packedB(A, B, C, N, bs);  break;
        case 4:  multiply_gemm_4x4_k_tiled_packedB(A, B, C, N, bs);  break;
        default: multiply_gemm_2x2(A, B, C, N, bs); break;
    }
}

int main(int argc, char* argv[]) {

    printf("Testing matrix ");

    int size = SIZE;
    int tile_size = DEFAULT_TILE_SIZE;
    int k_block_size = K_BLOCK_SIZE;

    if (argc == 2) {
        size = atoi( argv[1] );
    }
    else if (argc == 3) {
        size = atoi( argv[1] );
        tile_size = atoi( argv[2] );
    }
    else if (argc == 4){
        size = atoi( argv[1] );
        tile_size = atoi( argv[2] );
        k_block_size = atoi( argv[3] );     
    }

    printf("size: %d x %d    tile_size:%d %s    k_block_size:%d %s\n", size, size, tile_size, tile_size==0 ? "AUTO\0" : "", k_block_size, k_block_size==0 ? "AUTO\0" : "");
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
       A[i] = rand() % 10 + 1;;
       B[i] = rand() % 10 + 1;;
       C[i] = 0.0;
    }

    if(DEBUG_PRINT_IO == 1){
        printf("A");
        print_matrixf32(A, size, size, 0);
        printf("B");
        print_matrixf32(B, size, size, 0);
    }

    // Start timer
    //double start_time = omp_get_wtime();
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply_gemm(A, B, C, size, tile_size, k_block_size);

    // Stop timer
    //double end_time = omp_get_wtime();
    clock_t end_time = clock();

    if(DEBUG_PRINT_IO == 1){
        printf("C");
        print_matrixf32(C, size, size, 0);
    }

    // Calculate and print execution time
    //double execution_time = (end_time - start_time); // [sec]
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : rvv_smatmulop_f32_ktiling, %f, %d, %d, %d\n", execution_time, size, tile_size, k_block_size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
