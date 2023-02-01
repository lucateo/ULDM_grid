# This is the compiler
CC=mpicxx
# LIBS=-fopenmp -lfftw3 -lfftw3f_threads -lfftw3f -lm
LIBS=-std=c++0x -O3 -march=native -fopenmp -L/opt/gridware/depots/e2b91392/el7/pkg/libs/fftw3_double/3.3.4/gcc-5.5.0+openmpi-1.10.7/lib -I/opt/gridware/depots/e2b91392/el7/pkg/libs/fftw3_double/3.3.4/gcc-5.5.0+openmpi-1.10.7/include -lfftw3_omp -lfftw3_mpi -lfftw3 -lm -lpthread
all: main_sim_mpi
main_sim_mpi: main_sim_mpi_2field.cpp uldm_mpi_2field.h
	$(CC) -o $@ $< $(LIBS)
