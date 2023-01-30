# This is the compiler
CC=mpic++
# LIBS=-fopenmp -lfftw3 -lfftw3f_threads -lfftw3f -lm
LIBS=-fopenmp -lfftw3_omp -lfftw3 -lfftw3_threads -lboost_system -lm -std=c++11 -O3
all: main_sim_nfw main_sim
main_sim: main_sim.cpp uldm_sim.h
	$(CC) -o $@ $< $(LIBS)
main_sim_nfw: main_nfw_new.cpp uldm_sim_nfw.h uldm_sim.h
	$(CC) -o $@ $< $(LIBS)

