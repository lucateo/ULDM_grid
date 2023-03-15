#include "uldm_mpi_2field.h"


//To make it run, an example: ./mpirun -np 4 main_sim 128 100 1000 1 1Sol false 4

// Remember to change string_outputname according to where you want to store output files, or
// create the directory BEFORE running the code
int main(int argc, char** argv){

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    fftw_init_threads();
    fftw_mpi_init();
    // Find out rank of the particular process and the overal number of processes being run
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if(world_rank==0){cout<<" world size is "<<world_size<<endl;}
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1); }


    int beginning=time(NULL);
    // srand(time(NULL));
    //srand(42);

    int Nx = atof(argv[1]);                       //number of gridpoints

    // keep all the boxes the same height for simplicity, so change if not divisible
    if(Nx%world_size!=0){
         if(world_rank==0){ cout<<"warning: space steps not divisible, adjusting"<<endl;}
         int temp= (int) Nx/world_size;
         Nx=temp*world_size;
    }
    // and as a result, points in the short direction
    int Nz= (Nz)/world_size;

    double Length= atof(argv[2]);                 //box Length in units of m
    int numsteps= atof(argv[3]);                   //Number of steps
    double dt= atof(argv[4]);                     //timeSpacing,  tf/Nt

    int outputnumb=10;//atof(argv[6]);;            //number of outputs
    int outputnumb_profile=10;//atof(argv[6]);;            //number of outputs

    double dx = Length/Nx;                            //latticeSpacing, dx=L/Nx, L in units of m
    int Pointsmax = Nx/2; //Number of maximum points which are plotted in profile function

    int nghost=2; // number of ghost cells on the psi grids, 2 is the usual and the code will probably break with any other value

    // This is to tell which initial condition you want to run
    string initial_cond = argv[5];

    string start_from_backup = argv[6]; // true or false, depending on whether you want to start from a backup or not
    bool backup_bool = false;
    if (start_from_backup == "true")
      backup_bool = true;


    if (initial_cond == "levkov" ) {// Levkov initial conditions
      if (argc > 8){
        double Npart = atof(argv[7]); // Number of particles
        int vw = atof(argv[8]); // Velocity of particles
        string outputname = "out/out_test/out_Levkov_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
          + "_Npart_" + to_string(Npart) + "_vw_" + to_string(vw)+ "_";
        domain3 D3(Nx,Nz,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost);
        D3.set_grid(false);
        D3.set_grid_phase(true); // It will output 2D slice of phase grid
        D3.set_waves_Levkov(Npart, vw);
       //  D3.set_backup_flag(backup_bool);
        D3.solveConvDif();
      }
      else
        cout<<"You need 8 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, Npart, vw" << endl;
    }
    else{
      cout<< "String in 5th position does not match any possible initial conditions; possible initial conditions are:" << endl;
      cout<< "Schive , Mocz , deterministic , levkov, 1Sol, Schive_random_vel" <<endl;
    }

    return 0;
}


