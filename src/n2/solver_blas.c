/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <cblas.h>

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B, double *x) {
    double *y = (double *)aligned_alloc(32, (size_t)N     * sizeof(double));
    double *t = (double *)aligned_alloc(32, (size_t)N     * sizeof(double));

    for(int i = 0; i < N; i++)
        t[i] = 1.0f;

    if (!t || !y) {
        printf("malloc failed\n");
        free(t); free(y);
        return NULL;
    }

    // O(n ^ 3) solution
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, t, 1, 0.0, y, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, N, N, 1.0, B, N, y, 1, 0.0, t, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, t, 1, 0.0, y, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, N, N, 1.0, B, N, y, 1, 0.0, t, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, B, N, t, 1, 0.0, y, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, N, N, 1.0, A, N, y, 1, 0.0, t, 1);

    cblas_daxpy(N, 1.0, x, 1, t, 1);

    free(y);

    return t;
}
