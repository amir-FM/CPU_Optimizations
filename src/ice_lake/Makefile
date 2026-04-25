# !!! Do not modify this file !!!
CC=gcc
CFLAGS=-Wall -Werror -O0 -march=native
LIBS=-lopenblas

all: compare tema1_blas tema1_neopt tema1_opt_m 

tema1_blas: solver_blas.c main.c utils.h
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

tema1_neopt: solver_neopt.c main.c utils.h
	$(CC) $(CFLAGS) $^ -o $@

tema1_opt_m: solver_opt.c main.c utils.h
	$(CC) $(CFLAGS) $^ -o $@

compare: compare.c utils.h
	$(CC) $(OPT_CFLAGS) $^ -lm -o $@

clean:
	rm -rf compare tema1_blas tema1_neopt tema1_opt_m
