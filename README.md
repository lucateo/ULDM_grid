UltraLight Dark Matter code for multiple fields
==============================
This repository hosts a C++ mpi implementation of the Schr&ouml;dinger-Poisson equation for multiple fields. It relies on a pseudo-spectral solving algorithm mostly based on Levkov et al 2018.

There are 3 main programs:

* ``main_sim_mpi_2field.cpp``: The main program with the multiple fields implementation;
* ``main_sim_mpi_ext.cpp``: It solves the Schr&ouml;dinger-Poisson equations with an external potential;
* ``main_sim_mpi_stars.cpp``: It includes the presence of stars, considered to be subdominant with respect to the ULDM field. In particular, stars react only to the ULDM gravitational potential (and no backreaction on the ULDM field)

Running the code
-----------
You can use Makefile to compile (you need boost, openmp, mpi and fftw libraries). 
After compiling, to make it run, type for example

mpirun main_sim_mpi input_file.txt

where input_file.txt is the text file with the instructions. See the related file in this repository for explanations on the conventions for the input files.

initial_conditions.md has explanations of the initial conditions available.

The code comes with Doxygen documentation.

<!-- 0 = no mpi support (1 if you want mpi), 128 = number grid points, 20 = length of box, 400 = number of steps, 0.02 = time step, 2 = number of fields, 30= number of outputs (for animation), 60 = number of outputs with radial averaged profiles, 
NFW = string which tells which initial condition to use (in this case, an external NFW and 2 fields, initialized as Gaussian waves),
true = bool, false if you do not want to start from a backup run, true if you want to start from a backup run.
The others are parameters specific of the run (in this case, 0.57 = rho_s in grid units, 8=Rs in grid units, 140= number of particles for field 1 in grid units, 100 = number of particles for field 2 in grid units, 3 = ratio of mass between field 2 and 1)

The files are stored in a directory specific to the type of run, which should be created BEFORE running the code; 
just look at the output name in main_sim_mpi_2field.cpp (corresponding to the particular initial condition you are interested into). Backup files when mpi=true are stored in different files for different nodes. If you start a run using a certain number of cores, all the following runs (which uses the previous backup as initial conditions) have to use the same number of cores

Another example:

mpirun -np 16 main_sim_mpi 1 512 100 4000 0.1 1 100 400 levkov false 80 1

Simulations which uses levkov initial conditions, 16 cores, with mpi. Files are stored on the out_levkov/ directory, which should be created BEFORE running the program. -->

Details of the MPI implementation
- the grid is split in the z-direction onto different processes
- each of the psi grids has an additional two "ghost" layers both at the top and the bottom, which overlap wth the physical grid points on other processes (needed only for doing gradients in real space)
- Phi does not have ghost cells
- Fourier transforms basically handle themselves with the output grid split simmilarly in the k_z direction (the FT memory doesn't need ghost cells)

Details of the n fields implementation
- the array psi has 2*nfield components: Re(psi_i), Im(psi_i), all of them are evolved and contribute to the potential

Authors
-----
- Edward Hardy
- Marco Gorghetto
- Luca Teodori