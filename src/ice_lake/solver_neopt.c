/*
 * Tema 2 ASC
 * 2026 Spring
 */
#include "utils.h"

void mat_tran(int N, double *A, double *At) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            At[i * N + j] = A[j * N + i];
        }
    }
}

void mat_mul(int N, double *A, double *B, double *C) {
    for(int i = 0; i < N; i++) {
        for(int j = i; j < N; j++) {
            C[i * N + j] = 0;
            for(int k = 0; k < N; k++) {
              C[i * N + j] += A[i * N + k] * B[k * N + j];
            }

            C[j * N + i] = C[i * N + j];
        }
    }
}

void vector_add(int N, double *dest, double *src) {
    for(int i = 0; i < N; i++) {
        dest[i] += src[i];
    }
}

void vector_mul(int N, double *v, double *A, double *c) {
    for(int i = 0; i < N; i++) {
        c[i] = 0;
        for(int k = 0; k < N; k++) {
            c[i] += v[k] * A[i * N + k];
        }
    }
}

void get_line(int N, int i, double *A, double *a) {
    for(int j = 0; j < N; j++) {
        a[j] = A[i * N + j];
    }
}

double* my_solver(int N, double *A, double *B, double *x) {
    double *C = (double*)calloc(N * N, sizeof(double));
    double *At = (double*)calloc(N * N, sizeof(double));
    double *Ct = (double*)calloc(N * N, sizeof(double));
    double *D = (double*)calloc(N * N, sizeof(double));
    double *y = (double*)calloc(N, sizeof(double));
    double *aux = (double*)calloc(N, sizeof(double));
    double *line = (double*)calloc(N, sizeof(double));

    if(!C || !D || !y || !At || !Ct || !aux || !line) {
        printf("Calloc Failed\n");
        return NULL;
    }

    mat_tran(N, A, At);
    mat_mul(N, At, B, C);
    mat_tran(N, C, Ct);
    mat_mul(N, C, Ct, D);

    int i = 0;
    while(i < N) {
        get_line(N, i, C, line);
        vector_mul(N, line, D, aux);
        vector_add(N, y, aux);
        i++;
    }

    vector_add(N, y, x);

    free(C);
    free(Ct);
    free(At);
    free(D);
    free(aux);
    free(line);

    return y;
}
