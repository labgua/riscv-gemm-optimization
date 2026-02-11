#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include "../utils.h"

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
    
    // ordered B: size x 2
    float* oB = malloc(sizeof(float) * N * Tw);

    size_t vl;

    for (int jh = 0; jh < N; jh += vl) {
        for (int ih = 0; ih < N; ih += Th) {
        
            if (ih == 0 && jh % Tw == 0) {
                reordering_rvv(&B[ih][jh], oB, N, Tw);
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

        }
    }
}