# # This is the compiler
# CC=mpicxx
# # directory of possible libraries
# LDIR=/usr/local/lib
# # Option to put for compiling .o objects
# CLIB=-L$(LDIR)
# # flags, i.e. option to pass to the compiler (tell where to search for stuff)
# IDIR=/usr/local/include
# CFLAGS=-I$(IDIR)
# # directory of .o files
# ODIR=obj
# # possible local header file
# DEPS=uldm_mpi_2field.h
# # SOURCES=main_sim_mpi_2field.cpp uldm_mpi_2field_initial_cond.cpp utilities.cpp fourier.cpp
# SOURCES=$(shell find -name '*.cpp') # Trying to put wildcard
# _OBJ= $(patsubst %.cpp,%.o,$(SOURCES))
# # Retrieving the object files specified in _OBJ in the subdirectory ODIR (here obj)
# OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))
# LIBS=-std=c++0x -O3 -march=native -fopenmp -L/opt/gridware/depots/e2b91392/el7/pkg/libs/fftw3_double/3.3.4/gcc-5.5.0+openmpi-1.10.7/lib -I/opt/gridware/depots/e2b91392/el7/pkg/libs/fftw3_double/3.3.4/gcc-5.5.0+openmpi-1.10.7/include -lfftw3_omp -lfftw3_mpi -lfftw3 -lm -lpthread

# $(ODIR)/%.o: %.cpp $(DEPS)
# 	$(CC) -c -g -o $@ $< $(CFLAGS)
# # LIBS=-fopenmp -lfftw3 -lfftw3f_threads -lfftw3f -lm
# # $^ takes all prerequisites (the thing at the right of :)
# main_sim_mpi: $(OBJ)
# 	$(CC) -o $@ $^ $(CLIB) $(LIBS)

# .PHONY: clean
# # clean the directory from all .o objects
# clean:
# 	rm -f $(ODIR)/*.o
# #all: main_sim_mpi
# # main_sim_mpi: main_sim_mpi_2field.cpp uldm_mpi_2field_initial_cond.cpp uldm_mpi_2field.h
# # 	$(CC) -o $@ $< $(LIBS)
# This is the compiler
CC=mpicxx
SOURCES=domain3_main.cpp fourier.cpp utilities.cpp domain3_initial_cond.cpp output_domain3.cpp
# LIBS=-fopenmp -lfftw3 -lfftw3f_threads -lfftw3f -lm
LIBS=-std=c++0x -O3 -march=native -fopenmp -L/opt/gridware/depots/e2b91392/el7/pkg/libs/fftw3_double/3.3.4/gcc-5.5.0+openmpi-1.10.7/lib -I/opt/gridware/depots/e2b91392/el7/pkg/libs/fftw3_double/3.3.4/gcc-5.5.0+openmpi-1.10.7/include -lfftw3_omp -lfftw3_mpi -lfftw3 -lm -lpthread
all: main_sim_mpi
main_sim_mpi: main_sim_mpi_2field.cpp $(SOURCES) uldm_mpi_2field.h uldm_sim_nfw.h
	$(CC) -o $@ $< $(SOURCES) $(LIBS)