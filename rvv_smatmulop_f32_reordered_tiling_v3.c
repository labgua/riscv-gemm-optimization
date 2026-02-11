#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include "utils.h"

#define DEBUG_ENABLED 0
int DEBUG_LEVEL;

#define SIZE 8

#define DEFAULT_TILE_SIZE 0

// [0, 1]
#define RANDOM 0

#define DEFAULT_LMUL 1

inline int get_vlen(){
    size_t VLMAX8 = __riscv_vsetvlmax_e8m1();
    int VLEN = VLMAX8 * 8;
    return VLEN;
}

inline void reordering_rvv(float* mat2, float* omat2, int size, int ts) {
    for (int i = 0; i < size; i++) {
        const float* src = mat2 + (size * i);
        float* dst = omat2 + (ts * i);
        size_t remaining = ts;
        
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m8(remaining);  // LMUL=8
            
            vfloat32m8_t vec = __riscv_vle32_v_f32m8(src, vl);
            __riscv_vse32_v_f32m8(dst, vec, vl);
            
            src += vl;
            dst += vl;
            remaining -= vl;
        }
    }
}


void kernel_2_m1(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 2;
    int Tw = __riscv_vsetvl_e32m1(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            vl = __riscv_vsetvl_e32m1(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m1_t vc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m1_t vb =
                    __riscv_vle32_v_f32m1(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m1(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m1(
                    vc1, A[ih + 1][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m1(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], vc1, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_2_m2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 2;
    int Tw = __riscv_vsetvl_e32m2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            vl = __riscv_vsetvl_e32m2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m2_t vc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m2_t vb =
                    __riscv_vle32_v_f32m2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m2(
                    vc1, A[ih + 1][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m2(&C[ih + 1][jh], vc1, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_2_m4(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 2;
    int Tw = __riscv_vsetvl_e32m4(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            vl = __riscv_vsetvl_e32m4(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m4_t vc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m4_t vb =
                    __riscv_vle32_v_f32m4(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m4(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m4(
                    vc1, A[ih + 1][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m4(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m4(&C[ih + 1][jh], vc1, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_2_m8(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 2;
    int Tw = __riscv_vsetvl_e32m8(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            vl = __riscv_vsetvl_e32m8(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m8_t vb =
                    __riscv_vle32_v_f32m8(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m8(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m8(
                    vc1, A[ih + 1][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m8(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m8(&C[ih + 1][jh], vc1, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_2_mf2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 2;
    int Tw = __riscv_vsetvl_e32mf2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            vl = __riscv_vsetvl_e32mf2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32mf2_t vc0 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc1 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32mf2_t vb =
                    __riscv_vle32_v_f32mf2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32mf2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32mf2(
                    vc1, A[ih + 1][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32mf2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 1][jh], vc1, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}



void kernel_4_m1(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 4;
    int Tw = __riscv_vsetvl_e32m1(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m1(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m1_t vc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m1_t vb =
                    __riscv_vle32_v_f32m1(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m1(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m1(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m1(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m1(
                    vc3, A[ih + 3][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m1(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m1(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 3][jh], vc3, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_4_m2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 4;
    int Tw = __riscv_vsetvl_e32m2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m2_t vc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m2_t vb =
                    __riscv_vle32_v_f32m2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m2(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m2(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m2(
                    vc3, A[ih + 3][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m2(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m2(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m2(&C[ih + 3][jh], vc3, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_4_m4(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 4;
    int Tw = __riscv_vsetvl_e32m4(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m4(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m4_t vc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m4_t vb =
                    __riscv_vle32_v_f32m4(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m4(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m4(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m4(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m4(
                    vc3, A[ih + 3][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m4(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m4(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m4(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m4(&C[ih + 3][jh], vc3, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }
        }
    }
}

void kernel_4_m8(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 4;
    int Tw = __riscv_vsetvl_e32m8(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m8(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc3 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m8_t vb =
                    __riscv_vle32_v_f32m8(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m8(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m8(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m8(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m8(
                    vc3, A[ih + 3][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m8(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m8(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m8(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m8(&C[ih + 3][jh], vc3, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_4_mf2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 4;
    int Tw = __riscv_vsetvl_e32mf2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32mf2(N - jh);
            vl = __riscv_vsetvl_e32mf2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32mf2_t vc0 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc1 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc2 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc3 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32mf2_t vb =
                    __riscv_vle32_v_f32mf2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32mf2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32mf2(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32mf2(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32mf2(
                    vc3, A[ih + 3][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32mf2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 3][jh], vc3, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}


void kernel_8_m1(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 8;
    int Tw = __riscv_vsetvl_e32m1(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m1(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m1_t vc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m1_t vb =
                    __riscv_vle32_v_f32m1(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m1(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m1(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m1(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m1(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m1(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m1(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m1(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m1(
                    vc7, A[ih + 7][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m1(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m1(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m1(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m1(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m1(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m1(&C[ih + 7][jh], vc7, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_8_m2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 8;
    int Tw = __riscv_vsetvl_e32m2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m2_t vc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m2_t vb =
                    __riscv_vle32_v_f32m2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m2(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m2(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m2(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m2(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m2(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m2(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m2(
                    vc7, A[ih + 7][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m2(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m2(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m2(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m2(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m2(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m2(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m2(&C[ih + 7][jh], vc7, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_8_m4(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 8;
    int Tw = __riscv_vsetvl_e32m4(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m4_t vc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc4 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc5 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc6 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc7 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m4_t vb =
                    __riscv_vle32_v_f32m4(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m4(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m4(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m4(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m4(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m4(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m4(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m4(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m4(
                    vc7, A[ih + 7][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m4(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m4(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m4(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m4(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m4(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m4(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m4(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m4(&C[ih + 7][jh], vc7, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_8_m8(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 8;
    int Tw = __riscv_vsetvl_e32m8(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m8(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc3 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc4 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc5 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc6 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc7 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m8_t vb =
                    __riscv_vle32_v_f32m8(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m8(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m8(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m8(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m8(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m8(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m8(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m8(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m8(
                    vc7, A[ih + 7][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m8(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m8(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m8(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m8(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m8(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m8(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m8(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m8(&C[ih + 7][jh], vc7, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_8_mf2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 8;
    int Tw = __riscv_vsetvl_e32mf2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32mf2(N - jh);
            vl = __riscv_vsetvl_e32mf2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32mf2_t vc0 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc1 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc2 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc3 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc4 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc5 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc6 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc7 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32mf2_t vb =
                    __riscv_vle32_v_f32mf2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32mf2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32mf2(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32mf2(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32mf2(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32mf2(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32mf2(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32mf2(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32mf2(
                    vc7, A[ih + 7][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32mf2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 7][jh], vc7, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}


void kernel_16_m1(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 16;
    int Tw = __riscv_vsetvl_e32m1(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m1(N - jh);
            vl = __riscv_vsetvl_e32m1(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m1_t vc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc8 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc9 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc10 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc11 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc12 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc13 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc14 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t vc15 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m1_t vb =
                    __riscv_vle32_v_f32m1(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m1(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m1(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m1(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m1(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m1(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m1(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m1(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m1(
                    vc7, A[ih + 7][k], vb, vl);

                vc8 = __riscv_vfmacc_vf_f32m1(
                    vc8, A[ih + 8][k], vb, vl);

                vc9 = __riscv_vfmacc_vf_f32m1(
                    vc9, A[ih + 9][k], vb, vl);

                vc10 = __riscv_vfmacc_vf_f32m1(
                    vc10, A[ih + 10][k], vb, vl);

                vc11 = __riscv_vfmacc_vf_f32m1(
                    vc11, A[ih + 11][k], vb, vl);

                vc12 = __riscv_vfmacc_vf_f32m1(
                    vc12, A[ih + 12][k], vb, vl);

                vc13 = __riscv_vfmacc_vf_f32m1(
                    vc13, A[ih + 13][k], vb, vl);

                vc14 = __riscv_vfmacc_vf_f32m1(
                    vc14, A[ih + 14][k], vb, vl);

                vc15 = __riscv_vfmacc_vf_f32m1(
                    vc15, A[ih + 15][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m1(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m1(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m1(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m1(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m1(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m1(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m1(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m1(&C[ih + 7][jh], vc7, vl);
            __riscv_vse32_v_f32m1(&C[ih + 8][jh], vc8, vl);
            __riscv_vse32_v_f32m1(&C[ih + 9][jh], vc9, vl);
            __riscv_vse32_v_f32m1(&C[ih + 10][jh], vc10, vl);
            __riscv_vse32_v_f32m1(&C[ih + 11][jh], vc11, vl);
            __riscv_vse32_v_f32m1(&C[ih + 12][jh], vc12, vl);
            __riscv_vse32_v_f32m1(&C[ih + 13][jh], vc13, vl);
            __riscv_vse32_v_f32m1(&C[ih + 14][jh], vc14, vl);
            __riscv_vse32_v_f32m1(&C[ih + 15][jh], vc15, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_16_m2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 16;
    int Tw = __riscv_vsetvl_e32m2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m2(N - jh);
            vl = __riscv_vsetvl_e32m2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m2_t vc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc8 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc9 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc10 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc11 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc12 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc13 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc14 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            vfloat32m2_t vc15 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m2_t vb =
                    __riscv_vle32_v_f32m2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m2(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m2(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m2(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m2(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m2(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m2(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m2(
                    vc7, A[ih + 7][k], vb, vl);

                vc8 = __riscv_vfmacc_vf_f32m2(
                    vc8, A[ih + 8][k], vb, vl);

                vc9 = __riscv_vfmacc_vf_f32m2(
                    vc9, A[ih + 9][k], vb, vl);

                vc10 = __riscv_vfmacc_vf_f32m2(
                    vc10, A[ih + 10][k], vb, vl);

                vc11 = __riscv_vfmacc_vf_f32m2(
                    vc11, A[ih + 11][k], vb, vl);

                vc12 = __riscv_vfmacc_vf_f32m2(
                    vc12, A[ih + 12][k], vb, vl);

                vc13 = __riscv_vfmacc_vf_f32m2(
                    vc13, A[ih + 13][k], vb, vl);

                vc14 = __riscv_vfmacc_vf_f32m2(
                    vc14, A[ih + 14][k], vb, vl);

                vc15 = __riscv_vfmacc_vf_f32m2(
                    vc15, A[ih + 15][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m2(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m2(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m2(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m2(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m2(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m2(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m2(&C[ih + 7][jh], vc7, vl);
            __riscv_vse32_v_f32m2(&C[ih + 8][jh], vc8, vl);
            __riscv_vse32_v_f32m2(&C[ih + 9][jh], vc9, vl);
            __riscv_vse32_v_f32m2(&C[ih + 10][jh], vc10, vl);
            __riscv_vse32_v_f32m2(&C[ih + 11][jh], vc11, vl);
            __riscv_vse32_v_f32m2(&C[ih + 12][jh], vc12, vl);
            __riscv_vse32_v_f32m2(&C[ih + 13][jh], vc13, vl);
            __riscv_vse32_v_f32m2(&C[ih + 14][jh], vc14, vl);
            __riscv_vse32_v_f32m2(&C[ih + 15][jh], vc15, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_16_m4(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 16;
    int Tw = __riscv_vsetvl_e32m4(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m4(N - jh);
            vl = __riscv_vsetvl_e32m4(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m4_t vc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc4 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc5 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc6 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc7 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc8 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc9 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc10 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc11 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc12 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc13 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc14 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vc15 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m4_t vb =
                    __riscv_vle32_v_f32m4(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m4(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m4(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m4(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m4(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m4(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m4(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m4(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m4(
                    vc7, A[ih + 7][k], vb, vl);

                vc8 = __riscv_vfmacc_vf_f32m4(
                    vc8, A[ih + 8][k], vb, vl);

                vc9 = __riscv_vfmacc_vf_f32m4(
                    vc9, A[ih + 9][k], vb, vl);

                vc10 = __riscv_vfmacc_vf_f32m4(
                    vc10, A[ih + 10][k], vb, vl);

                vc11 = __riscv_vfmacc_vf_f32m4(
                    vc11, A[ih + 11][k], vb, vl);

                vc12 = __riscv_vfmacc_vf_f32m4(
                    vc12, A[ih + 12][k], vb, vl);

                vc13 = __riscv_vfmacc_vf_f32m4(
                    vc13, A[ih + 13][k], vb, vl);

                vc14 = __riscv_vfmacc_vf_f32m4(
                    vc14, A[ih + 14][k], vb, vl);

                vc15 = __riscv_vfmacc_vf_f32m4(
                    vc15, A[ih + 15][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m4(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m4(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m4(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m4(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m4(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m4(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m4(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m4(&C[ih + 7][jh], vc7, vl);
            __riscv_vse32_v_f32m4(&C[ih + 8][jh], vc8, vl);
            __riscv_vse32_v_f32m4(&C[ih + 9][jh], vc9, vl);
            __riscv_vse32_v_f32m4(&C[ih + 10][jh], vc10, vl);
            __riscv_vse32_v_f32m4(&C[ih + 11][jh], vc11, vl);
            __riscv_vse32_v_f32m4(&C[ih + 12][jh], vc12, vl);
            __riscv_vse32_v_f32m4(&C[ih + 13][jh], vc13, vl);
            __riscv_vse32_v_f32m4(&C[ih + 14][jh], vc14, vl);
            __riscv_vse32_v_f32m4(&C[ih + 15][jh], vc15, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_16_m8(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 16;
    int Tw = __riscv_vsetvl_e32m8(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32m8(N - jh);
            vl = __riscv_vsetvl_e32m8(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc3 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc4 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc5 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc6 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc7 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc8 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc9 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc10 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc11 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc12 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc13 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc14 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc15 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32m8_t vb =
                    __riscv_vle32_v_f32m8(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32m8(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32m8(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32m8(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32m8(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32m8(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32m8(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32m8(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32m8(
                    vc7, A[ih + 7][k], vb, vl);

                vc8 = __riscv_vfmacc_vf_f32m8(
                    vc8, A[ih + 8][k], vb, vl);

                vc9 = __riscv_vfmacc_vf_f32m8(
                    vc9, A[ih + 9][k], vb, vl);

                vc10 = __riscv_vfmacc_vf_f32m8(
                    vc10, A[ih + 10][k], vb, vl);

                vc11 = __riscv_vfmacc_vf_f32m8(
                    vc11, A[ih + 11][k], vb, vl);

                vc12 = __riscv_vfmacc_vf_f32m8(
                    vc12, A[ih + 12][k], vb, vl);

                vc13 = __riscv_vfmacc_vf_f32m8(
                    vc13, A[ih + 13][k], vb, vl);

                vc14 = __riscv_vfmacc_vf_f32m8(
                    vc14, A[ih + 14][k], vb, vl);

                vc15 = __riscv_vfmacc_vf_f32m8(
                    vc15, A[ih + 15][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32m8(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32m8(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32m8(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32m8(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32m8(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32m8(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32m8(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32m8(&C[ih + 7][jh], vc7, vl);
            __riscv_vse32_v_f32m8(&C[ih + 8][jh], vc8, vl);
            __riscv_vse32_v_f32m8(&C[ih + 9][jh], vc9, vl);
            __riscv_vse32_v_f32m8(&C[ih + 10][jh], vc10, vl);
            __riscv_vse32_v_f32m8(&C[ih + 11][jh], vc11, vl);
            __riscv_vse32_v_f32m8(&C[ih + 12][jh], vc12, vl);
            __riscv_vse32_v_f32m8(&C[ih + 13][jh], vc13, vl);
            __riscv_vse32_v_f32m8(&C[ih + 14][jh], vc14, vl);
            __riscv_vse32_v_f32m8(&C[ih + 15][jh], vc15, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}

void kernel_16_mf2(float* mat1, float* mat2, float* res, int N) {

    float (*A)[N] = (float (*)[N]) mat1;
    float (*B)[N] = (float (*)[N]) mat2;
    float (*C)[N] = (float (*)[N]) res;

    const int Th = 16;
    int Tw = __riscv_vsetvl_e32mf2(N);

    printf("Tw:%d\n", Tw);
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);

                if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                    printf("ih==0 ==> orderedB");
                    print_lmatrixf32(oB, Tw, N * Tw);
                }
            }

            // Tw deciso a runtime
            ////vl = __riscv_vsetvl_e32mf2(N - jh);
            vl = __riscv_vsetvl_e32mf2(N - jh);

            // accumulatori: uno per ogni riga del tile
            vfloat32mf2_t vc0 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc1 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc2 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc3 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc4 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc5 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc6 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc7 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc8 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc9 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc10 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc11 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc12 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc13 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc14 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);
            vfloat32mf2_t vc15 = __riscv_vfmv_v_f_f32mf2(0.0f, vl);

            #if UNROLL == 2
            #pragma GCC unroll 2
            #elif UNROLL == 4
            #pragma GCC unroll 4
            #elif UNROLL == 8
            #pragma GCC unroll 8
            #elif UNROLL == 16
            #pragma GCC unroll 16
            #else
            #pragma GCC unroll 1
            #endif
            for (int k = 0; k < N; ++k) {

                // carica B[k][jh : jh+vl]
                vfloat32mf2_t vb =
                    __riscv_vle32_v_f32mf2(&oB[k * Tw], vl);

                // outer product
                vc0 = __riscv_vfmacc_vf_f32mf2(
                    vc0, A[ih + 0][k], vb, vl);

                vc1 = __riscv_vfmacc_vf_f32mf2(
                    vc1, A[ih + 1][k], vb, vl);

                vc2 = __riscv_vfmacc_vf_f32mf2(
                    vc2, A[ih + 2][k], vb, vl);

                vc3 = __riscv_vfmacc_vf_f32mf2(
                    vc3, A[ih + 3][k], vb, vl);

                vc4 = __riscv_vfmacc_vf_f32mf2(
                    vc4, A[ih + 4][k], vb, vl);

                vc5 = __riscv_vfmacc_vf_f32mf2(
                    vc5, A[ih + 5][k], vb, vl);

                vc6 = __riscv_vfmacc_vf_f32mf2(
                    vc6, A[ih + 6][k], vb, vl);

                vc7 = __riscv_vfmacc_vf_f32mf2(
                    vc7, A[ih + 7][k], vb, vl);

                vc8 = __riscv_vfmacc_vf_f32mf2(
                    vc8, A[ih + 8][k], vb, vl);

                vc9 = __riscv_vfmacc_vf_f32mf2(
                    vc9, A[ih + 9][k], vb, vl);

                vc10 = __riscv_vfmacc_vf_f32mf2(
                    vc10, A[ih + 10][k], vb, vl);

                vc11 = __riscv_vfmacc_vf_f32mf2(
                    vc11, A[ih + 11][k], vb, vl);

                vc12 = __riscv_vfmacc_vf_f32mf2(
                    vc12, A[ih + 12][k], vb, vl);

                vc13 = __riscv_vfmacc_vf_f32mf2(
                    vc13, A[ih + 13][k], vb, vl);

                vc14 = __riscv_vfmacc_vf_f32mf2(
                    vc14, A[ih + 14][k], vb, vl);

                vc15 = __riscv_vfmacc_vf_f32mf2(
                    vc15, A[ih + 15][k], vb, vl);
            }

            // store
            __riscv_vse32_v_f32mf2(&C[ih + 0][jh], vc0, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 1][jh], vc1, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 2][jh], vc2, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 3][jh], vc3, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 4][jh], vc4, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 5][jh], vc5, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 6][jh], vc6, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 7][jh], vc7, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 8][jh], vc8, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 9][jh], vc9, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 10][jh], vc10, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 11][jh], vc11, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 12][jh], vc12, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 13][jh], vc13, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 14][jh], vc14, vl);
            __riscv_vse32_v_f32mf2(&C[ih + 15][jh], vc15, vl);

            if( DEBUG_ENABLED && DEBUG_LEVEL >= 3 ){
                ///printf("Computation block C(%d:%d) finished! \n", ih, jh);
                printf("> C  iter!!\n");
                print_lmatrixf32((float*) C, N, N * N);
            }

        }
    }
}


void multiply_gemm(float* A, float* B, float* C, int N, int th, int lmul) {

    if( DEBUG_ENABLED && DEBUG_LEVEL >= 0 ){
        printf("kernel> th=%d lmul=%d\n", th, lmul);
    }
    
         if( th == 2 && lmul == 1 ) kernel_2_m1(A, B, C, N);
    else if( th == 2 && lmul == 2 ) kernel_2_m2(A, B, C, N);
    else if( th == 2 && lmul == 4 ) kernel_2_m4(A, B, C, N);
    else if( th == 2 && lmul == 8 ) kernel_2_m8(A, B, C, N);
    else if( th == 2 && lmul == -2 ) kernel_2_mf2(A, B, C, N);

    else if( th == 4 && lmul == 1 ) kernel_4_m1(A, B, C, N);
    else if( th == 4 && lmul == 2 ) kernel_4_m2(A, B, C, N);
    else if( th == 4 && lmul == 4 ) kernel_4_m4(A, B, C, N);
    else if( th == 4 && lmul == 8 ) kernel_4_m8(A, B, C, N);
    else if( th == 4 && lmul == -2 ) kernel_4_mf2(A, B, C, N);

    else if( th == 8 && lmul == 1 ) kernel_8_m1(A, B, C, N);
    else if( th == 8 && lmul == 2 ) kernel_8_m2(A, B, C, N);
    else if( th == 8 && lmul == 4 ) kernel_8_m4(A, B, C, N);
    else if( th == 8 && lmul == 8 ) kernel_8_m8(A, B, C, N);
    else if( th == 8 && lmul == -2 ) kernel_8_mf2(A, B, C, N);

    else if( th == 16 && lmul == 1 ) kernel_16_m1(A, B, C, N);
    else if( th == 16 && lmul == 2 ) kernel_16_m2(A, B, C, N);
    else if( th == 16 && lmul == 4 ) kernel_16_m4(A, B, C, N);
    else if( th == 16 && lmul == 8 ) kernel_16_m8(A, B, C, N);
    else if( th == 16 && lmul == -2 ) kernel_16_mf2(A, B, C, N);
}

int main(int argc, char* argv[]) {

    printf("Testing matrix %s\n", DEBUG_ENABLED ? "(DEBUGGER ENABLED)\0" : "\0");
    printf("> VLEN: %d\n", get_vlen() );

    int size = SIZE;
    int kernel_size = DEFAULT_TILE_SIZE;
    int input_case = RANDOM_INPUT;
    DEBUG_LEVEL = 0;
    int DEBUG_PRINT_IO = 0;
    int lmul = DEFAULT_LMUL;

    if(IS_HELP){
        printf("options:\n");
        printf("> DEBUG_PRINT_IO\n> DEBUG_LEVEL\n> SIZE\n> KERNEL\n> INPUT_CASE\n> LMUL\n\n");
        printf("default values:\n");
        printf("> size: %d x %d \n> kernel_size:%d lmul:%d \n> input_case:%d (%s)\n", 
            size, size, 
            kernel_size,
            lmul,
            input_case, input_case_name(input_case)
        );
        exit(0);
    }

    if( ARG("DEBUG_PRINT_IO") ){
        printf("> passing DEBUG_PRINT_IO");
        DEBUG_PRINT_IO = atoi( ARG("DEBUG_PRINT_IO") );
        printf(" %d\n", DEBUG_PRINT_IO);        
    }
    if( ARG("DEBUG_LEVEL") ){
        printf("> passing DEBUG_LEVEL");
        DEBUG_LEVEL = atoi( ARG("DEBUG_LEVEL") );
        printf(" %d\n", DEBUG_LEVEL);        
    }
    if( ARG("SIZE") ){
        printf("> passing SIZE");
        size = atoi( ARG("SIZE") );
        printf(" %d\n", size);
    }
    if( ARG("KERNEL") ){
        printf("> passing KERNEL");
        kernel_size = atoi( ARG("KERNEL") );
        printf(" %d\n", kernel_size);        
    }
    if( ARG("INPUT_CASE") ){
        printf("> passing INPUT_CASE");
        input_case = atoi( ARG("INPUT_CASE") );
        printf(" %d\n", input_case);        
    }
    if( ARG("LMUL") ){
        printf("> passing LMUL");
        lmul = atoi( ARG("LMUL") );
        printf(" %d%s\n", lmul, lmul < 0 ? " (FRACTIONAL)\0" : "\0");
    }


    printf("size: %d x %d  kernel_size:%d %s  lmul:%d  input_case:%d (%s)\n", 
            size, size, 
            kernel_size, kernel_size==0 ? "AUTO\0" : "",
            lmul,
            input_case, input_case_name(input_case)
    );
    ///printf("Num proc: %d\n", omp_get_max_threads());

    // check tile_size is correct ( tile_size <= size )
    if( kernel_size > size ){
        printf("ERROR: kernel_size:%d must be less than or equal to size:%d\n", kernel_size, size);
        exit(EXIT_FAILURE);
    }
    
    // Allocate memory for matrices
    float *A = (float*)malloc(size * size * sizeof(float));
    float *B = (float*)malloc(size * size * sizeof(float));
    float *C = (float*)malloc(size * size * sizeof(float));

    // Init matrix values pseudorandom

    // set initial seed for rand if required
    srand( RANDOM ? time(NULL) : 1  );
    
    init_matrix_input(input_case, A, B, size, size, size);

    if(DEBUG_PRINT_IO){
        printf("A");
        print_matrixf32(A, size, size, 0);
        printf("B");
        print_matrixf32(B, size, size, 0);
    }

    // Start timer
    //double start_time = omp_get_wtime();
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply_gemm(A, B, C, size, kernel_size, lmul);

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
    printf("> BENCHMARK_RECORD : rvv_smatmulop_f32_reordered_tiling_v3, %f, %d, %d, %d\n", execution_time, size, kernel_size, lmul);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
