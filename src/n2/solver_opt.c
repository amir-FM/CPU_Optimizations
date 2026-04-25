/*
 * Tema 2 ASC
 * 2026 Spring
 */
#pragma GCC target("avx2,fma")
#include "utils.h"

/*
 * Add your optimized implementation here
 */

#include <string.h>
#include <stdio.h>
#include <immintrin.h>

static inline double reduce_m256d(__m256d v) {
    // Split the 256-bit register into two 128-bit halves (2 doubles each)
    __m128d vlow  = _mm256_castpd256_pd128(v);       // Lower half
    __m128d vhigh = _mm256_extractf128_pd(v, 1);     // Upper half

    // Add the two halves together
    __m128d vsum  = _mm_add_pd(vlow, vhigh);

    // Swap the top and bottom of the resulting 128-bit register
    __m128d vshuf = _mm_unpackhi_pd(vsum, vsum);

    // Add them one last time to get the final sum
    __m128d vres  = _mm_add_pd(vsum, vshuf);

    // Extract the lowest 64-bit double to return to standard C++
    return _mm_cvtsd_f64(vres);
}

static void mat_vec_mul_avx2(int N, const double *  A, 
                             const double *  x, double *  y) 
{
    for(int i = 0; i < N; i++) {
        register const double *Ai = A + i * N;
        
        // Two accumulators to unroll by 8
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        
        for(int k = 0; k < N; k += 8) {
            // Load 8 elements from row i of A
            __m256d v_Ai0 = _mm256_loadu_pd(&Ai[k]);
            __m256d v_Ai1 = _mm256_loadu_pd(&Ai[k + 4]);
            
            // Load 8 elements from vector x
            __m256d v_x0 = _mm256_loadu_pd(&x[k]);
            __m256d v_x1 = _mm256_loadu_pd(&x[k + 4]);
            
            // Fused Multiply-Add
            acc0 = _mm256_fmadd_pd(v_Ai0, v_x0, acc0);
            acc1 = _mm256_fmadd_pd(v_Ai1, v_x1, acc1);
        }
        
        // Combine, reduce, and store the final scalar
        __m256d final_acc = _mm256_add_pd(acc0, acc1);
        y[i] = reduce_m256d(final_acc);
    }
}

static void matT_vec_mul_avx2(int N, const double *  A, 
                              const double *  x, double *  y) 
{
    // Clear the output vector first
    memset(y, 0, (size_t)N * sizeof(double));
    
    // Notice the loops are swapped! k is now on the outside.
    for(int k = 0; k < N; k++) {
        register const double *Ak = A + k * N;
        
        // Because k is the outer loop, x[k] is a scalar for the inner loop.
        // We broadcast it to all 4 slots of the register.
        __m256d v_xk = _mm256_set1_pd(x[k]);
        
        for(int i = 0; i < N; i += 8) {
            
            // Load 20 elements from y
            __m256d v_y0 = _mm256_loadu_pd(&y[i]);
            __m256d v_y1 = _mm256_loadu_pd(&y[i + 4]);

            // Multiply the scalar x[k] by the sequential row Ak, add to y
            v_y0 = _mm256_fmadd_pd(v_xk, _mm256_loadu_pd(&Ak[i]), v_y0);
            v_y1 = _mm256_fmadd_pd(v_xk, _mm256_loadu_pd(&Ak[i + 4]), v_y1);

            // Store back to y
            _mm256_storeu_pd(&y[i], v_y0);
            _mm256_storeu_pd(&y[i + 4], v_y1);
        }
    }
}

/* -------------------------------------------------------------------------
 * my_solver — main entry point
 *
 *   Inputs : N     — dimension (must be a multiple of 40)
 *            A, B  — N×N matrices, row-major (caller owns)
 *            x     — N×1 vector (caller owns)
 *   Returns: y     — N×1 result vector (caller must free)
 * -----------------------------------------------------------------------*/
double *my_solver(int N, double *A, double *B, double *x)
{
    double *y = (double *)aligned_alloc(32, (size_t)N     * sizeof(double));
    double *t = (double *)aligned_alloc(32, (size_t)N     * sizeof(double));

    for(int i = 0; i < N; i++)
        t[i] = 1.0f;

    if (!t || !y) {
        printf("malloc failed\n");
        free(t); free(y);
        return NULL;
    }

    // O(n^2) solution
    mat_vec_mul_avx2(N, A, t, y);
    matT_vec_mul_avx2(N, B, y, t);
    mat_vec_mul_avx2(N, A, t, y);
    matT_vec_mul_avx2(N, B, y, t);
    mat_vec_mul_avx2(N, B, t, y);
    matT_vec_mul_avx2(N, A, y, t);

    for(int i = 0; i < N; i++)
        t[i] += x[i];

    free(y);

    return t;
}
