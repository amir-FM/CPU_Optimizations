/*
 * Tema 2 ASC
 * 2026 Spring
 */
#pragma GCC target("avx512f,fma")
#include "utils.h"

/*
 * Add your optimized implementation here
 */

#include <string.h>
#include <stdio.h>
#include <immintrin.h>

/* -------------------------------------------------------------------------
 * C = A^T * B  (N x N, row-major)
 * C[i][j] = sum_k  A[k][i] * B[k][j]
 *
 * Loop order k-i-j keeps B row-major sequential (hot inner loop).
 * A is accessed as column k, strided — kept in register a_ki.
 * -----------------------------------------------------------------------*/
static void mat_AtB(int N, const double *  A,
                           const double *  B,
                           double       *  C)
{
    memset(C, 0, (size_t)N * N * sizeof(double));

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            register double a_ki = A[k * N + i];           
            register const double *Bk = B + k * N;
            double       *Ci = C + i * N;

            __m512d v_a_ki = _mm512_set1_pd(a_ki);

            int j = 0;
            for (; j < N; j += 8) {
                __m512d v_Bk0 = _mm512_loadu_pd(&Bk[j]);
                __m512d v_Ci0 = _mm512_loadu_pd(&Ci[j]);
                v_Ci0 = _mm512_fmadd_pd(v_a_ki, v_Bk0, v_Ci0);
                _mm512_storeu_pd(&Ci[j], v_Ci0);
            }
        }
    }
}

static void mat_vec_mul(int N, const double *  A, 
                             const double *  x, double *  y) 
{
    for(int i = 0; i < N; i++) {
        register const double *Ai = A + i * N;
        
        // Two accumulators to unroll by 8
        __m512d acc0 = _mm512_setzero_pd();
        
        for(int k = 0; k < N; k += 8) {
            // Load 8 elements from row i of A
            __m512d v_Ai0 = _mm512_loadu_pd(&Ai[k]);
            
            // Load 8 elements from vector x
            __m512d v_x0 = _mm512_loadu_pd(&x[k]);
            
            // Fused Multiply-Add
            acc0 = _mm512_fmadd_pd(v_Ai0, v_x0, acc0);
        }
        
        // Combine, reduce, and store the final scalar
        y[i] = _mm512_reduce_add_pd(acc0);
    }
}

static void matT_vec_mul(int N, const double *  A, 
                              const double *  x, double *  y) 
{
    // Clear the output vector first
    memset(y, 0, (size_t)N * sizeof(double));
    
    // Notice the loops are swapped! k is now on the outside.
    for(int k = 0; k < N; k++) {
        register const double *Ak = A + k * N;
        
        // Because k is the outer loop, x[k] is a scalar for the inner loop.
        // We broadcast it to all 4 slots of the register.
        __m512d v_xk = _mm512_set1_pd(x[k]);
        
        for(int i = 0; i < N; i += 8) {
            
            // Load 20 elements from y
            __m512d v_y0 = _mm512_loadu_pd(&y[i]);

            // Multiply the scalar x[k] by the sequential row Ak, add to y
            v_y0 = _mm512_fmadd_pd(v_xk, _mm512_loadu_pd(&Ak[i]), v_y0);

            // Store back to y
            _mm512_storeu_pd(&y[i], v_y0);
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
    double *C = (double *)aligned_alloc(32, (size_t)N * N * sizeof(double));
    double *y = (double *)aligned_alloc(32, (size_t)N     * sizeof(double));
    double *t = (double *)aligned_alloc(32, (size_t)N     * sizeof(double));

    for(int i = 0; i < N; i++)
        t[i] = 1.0f;

    if (!C || !t || !y) {
        printf("malloc failed\n");
        free(C); free(t); free(y);
        return NULL;
    }

    // O(n ^ 3) solution
    mat_AtB    (N, A, B, C);
    matT_vec_mul(N, C, t, y);
    matT_vec_mul(N, C, y, t);
    mat_vec_mul(N, C, t, y);

    for(int i = 0; i < N; i++)
        y[i] += x[i];


    free(C);
    free(t);

    return y;
}
