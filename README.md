You can use Makefile to compile (you need boost, mpi and fftw libraries); main_sim_mpi_2field.cpp is the main file. 
After compiling, to make it run, type for example (for non mpi support)

./main_sim_mpi 0 128 20 400 0.02 2 30 60 NFW true 0.57 8 140 100 3

0 = no mpi support (1 if you want mpi), 128 = number grid points, 20 = length of box, 400 = number of steps, 0.02 = time step, 2 = number of fields, 30= number of outputs (for animation), 60 = number of outputs with radial averaged profiles, 
NFW = string which tells which initial condition to use (in this case, an external NFW and 2 fields, initialized as Gaussian waves),
true = bool, false if you do not want to start from a backup run, true if you want to start from a backup run.
The others are parameters specific of the run (in this case, 0.57 = rho_s in grid units, 8=Rs in grid units, 140= number of particles for field 1 in grid units, 100 = number of particles for field 2 in grid units, 3 = ratio of mass between field 2 and 1)

The files are stored in a directory specific to the type of run, which should be created BEFORE running the code; 
just look at the output name in main_sim_mpi_2field.cpp (corresponding to the particular initial condition you are interested into). Backup files when mpi=true are stored in different files for different nodes. If you start a run using a certain number of cores, all the following runs (which uses the previous backup as initial conditions) have to use the same number of cores

Another example:

mpirun -np 16 main_sim_mpi 1 512 100 4000 0.1 1 100 400 levkov false 80 1

Simulations which uses levkov initial conditions, 16 cores, with mpi. Files are stored on the out_levkov/ directory, which should be created BEFORE running the program.

Details of the MPI implementation
- the grid is split in the z-direction onto different processes
- each of the psi grids has an additional two "ghost" layers both at the top and the bottom, which overlap wth the physical grid points on other processes (needed only for doing gradients in real space)
- Phi does not have ghost cells
- Fourier transforms basically handle themselves with the output grid split simmilarly in the k_z direction (the FT memory doesn't need ghost cells)

Details of the n fields implementation
- the array psi now has 2*nfield components: Re(psi_i), Im(psi_i), all of them are evolved and contribute to the potential
