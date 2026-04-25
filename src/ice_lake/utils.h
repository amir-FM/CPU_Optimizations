/*
 * Tema 2 ASC
 * 2026 Spring
 * !!! Do not modify this file !!!
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef double* (*Solver)(int, double *, double*, double*);

#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (limit))

struct test {
	int seed;
	int N;
	// path to the file where the result matrix will be written
	char output_save_file[100];
};

double* my_solver(int, double *, double *, double *);

int run_test(struct test, Solver, float *);

void free_data(double **);

int generate_data(int, int, int, double **);

int read_input_file(char *, int *, struct test **);
