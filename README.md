You can use Makefile to compile (you need boost and fftw libraries); main_nfw.cpp is the main file with the NFW external potential attempt, the main_sim.cpp is the original only
ULDM simulation. After compiling, to make it run, type for example]

./main_sim 128 100 1000 1 1Sol false 4

128 = number grid points, 100 = length of box, 1000 = number of steps, 1 = time step, 1Sol = type of run, false = bool, false if you do not want to start from a backup run,
and the others are parameters specific of the run (in this case, 4 = core radius of the single soliton in grid units)

The files are stored in a directory specific to the type of run, which should be created BEFORE running the code (for example, for main_sim_nfw, you need to create the out_nfw/
directory first; just look at the cpp files themselves to understand).

Another example:

./main_sim 128 100 1000 1 Schive true 4 30 80

Simulations which uses multi soliton initial conditions a la Schive, starting from backup files of a run with the same parameters; the last three parameters are core radii of solitons, number of solitons and length of the box where the center of the solitons are randomply positioned. Files are stored on the out_schive/ directory, which should be created BEFORE running the program.
