#include "uldm_mpi_2field.h"
#include <string>


//To make it run, an example: mpirun -np 4 main_sim_mpi 128 100 1000 1 1Sol false 4

// Remember to change string_outputname according to where you want to store output files, or
// create the directory BEFORE running the code
int main(int argc, char** argv){
  int provided;
  int world_rank;
  int world_size;
  int nghost=2; // number of ghost cells on the psi grids, 2 is the usual and the code will probably break with any other value
  bool mpirun_flag = false;
  if(mpirun_flag == true){
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    fftw_init_threads();
    fftw_mpi_init();
    // Find out rank of the particular process and the overal number of processes being run
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if(world_rank==0){cout<<" world size is "<<world_size<<endl;}
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1); }
  }
  else {
    world_rank = 0;
    nghost = 0;
    world_size=1;
  }

  int beginning=time(NULL);
  // srand(time(NULL));
  //srand(42);

  int Nx = atof(argv[1]);                       //number of gridpoints
  // keep all the boxes the same height for simplicity, so change if not divisible
  if(mpirun_flag==true && Nx%world_size!=0){
        if(world_rank==0){ cout<<"warning: space steps not divisible, adjusting"<<endl;}
        int temp= (int) Nx/world_size;
        Nx=temp*world_size;
  }
  // and as a result, points in the short direction
  int Nz= (Nx)/world_size;
  double Length= atof(argv[2]);                 //box Length in units of m
  int numsteps= atof(argv[3]);                   //Number of steps
  double dt= atof(argv[4]);                     //timeSpacing,  tf/Nt
  double ratio_m = atof(argv[5]); // ratio between the two masses

  int outputnumb=atof(argv[6]);// Number of outputs for the sliced (if Grid3D==True) or full 3D grid (if Grid3D==False) density profile (for animation)
  int outputnumb_profile=atof(argv[7]);//number of outputs for radial profiles

  double dx = Length/Nx;                            //latticeSpacing, dx=L/Nx, L in units of m
  int Pointsmax = Nx/2; //Number of maximum points which are plotted in profile function

  // This is to tell which initial condition you want to run
  string initial_cond = argv[8];

  string start_from_backup = argv[9]; // true or false, depending on whether you want to start from a backup or not
  bool backup_bool = false;
  if (start_from_backup == "true")
    backup_bool = true;

  if (initial_cond == "levkov" ) {// Levkov initial conditions
    if (argc > 12){
      double Npart = atof(argv[10]); // Number of particles
      double Npart1 = atof(argv[11]); // Number of particles
      int vw = atof(argv[12]); // Velocity of particles
      string outputname = "out_mpi/out_test/out_2fields_Levkov_nopsisqmean_Nx" + to_string(Nx) +"_rmasses_"+ to_string(ratio_m) + "_L_" + to_string(Length)
        + "_Npart_" + to_string(Npart) + "_Npart1_" + to_string(Npart1) + "_vw_" + to_string(vw)+ "_";
      domain3 D3(Nx,Nz,Length,ratio_m,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.set_waves_Levkov(Npart, Npart1, vw);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 12 arguments to pass to the code: Nx, Length, tf, dt, ratio_mass, output_profile, output_profile_radial, initial_cond, start_from_backup, Npart, Npart1, vw" << endl;
  }
  else if (initial_cond == "1Sol" ) {// 1 Soliton initial conditions
    if (argc > 10){
      double rc = atof(argv[10]); // radius of soliton
      int whichpsi = atof(argv[11]); // radius of soliton
      string outputname = "out_mpi/out_test/out_2fields_1Sol_nopsisqmean_Nx" + to_string(Nx) +"_rmasses_"+ to_string(ratio_m) + "_L_" + to_string(Length)
        + "_rc_" + to_string(rc)+ "_field_"+to_string(whichpsi) + "_";
      domain3 D3(Nx,Nz,Length,ratio_m,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setInitialSoliton_1(rc, whichpsi);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 10 arguments to pass to the code: Nx, Length, tf, dt, ratio_mass, output_profile, output_profile_radial, initial_cond, start_from_backup, rc" << endl;
  }
  else if (initial_cond == "Schive" ) {// Schive initial conditions
    if (argc > 12){
      double rc = atof(argv[10]); // radius of soliton
      int Nsol = atof(argv[11]); // Number of solitons
      double length_lim = atof(argv[12]); // Length lim of span of solitons
      string outputname = "out_mpi/out_test/out_2fields_Schive_nopsisqmean_Nx" + to_string(Nx) +"_rmasses_"+ to_string(ratio_m) + "_L_" + to_string(Length)
        + "_rc_" + to_string(rc)+ "_Nsol_" + to_string(Nsol)+ "_Llim_" + to_string(length_lim)+"_";
      domain3 D3(Nx,Nz,Length,ratio_m,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost,mpirun_flag);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_same_radius(Nsol,rc,length_lim);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 12 arguments to pass to the code: Nx, Length, tf, dt, ratio_mass, output_profile, output_profile_radial, initial_cond, start_from_backup, rc, Nsol, length_lim" << endl;
  }
  else if (initial_cond == "Mocz" ) {// Mocz initial conditions
    if (argc > 13){
      double min_radius = atof(argv[10]); //min radius of soliton
      double max_radius = atof(argv[11]); //max radius of soliton
      int Nsol = atof(argv[12]); // Number of solitons
      double length_lim = atof(argv[13]); // Length lim of span of solitons
      string outputname = "out_mpi/out_test/out_2fields_Mocz_nopsisqmean_Nx" + to_string(Nx) +"_rmasses_"+ to_string(ratio_m) + "_L_" + to_string(Length)
        + "_rmin_" + to_string(min_radius)+"_rmax_" +to_string(max_radius)+ "_Nsol_" + to_string(Nsol)+ "_Llim_" + to_string(length_lim)+"_";
      domain3 D3(Nx,Nz,Length,ratio_m,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost,mpirun_flag);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_random_radius(Nsol,min_radius,max_radius,length_lim);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 13 arguments to pass to the code: Nx, Length, tf, dt, ratio_mass, output_profile, output_profile_radial, initial_cond, start_from_backup, rmin, rmax, Nsol, length_lim" << endl;
  }
  else if (initial_cond == "deterministic" ) {// Mocz initial conditions
    if (argc > 11){
      double rc = atof(argv[10]); //radius of soliton
      int Nsol = atof(argv[11]); // Number of solitons, should not surpass 30
      string outputname = "out_mpi/out_test/out_2fields_deterministic_nopsisqmean_Nx" + to_string(Nx) +"_rmasses_"+ to_string(ratio_m) + "_L_" + to_string(Length)
        + "_rc_" + to_string(rc)+ "_Nsol_" + to_string(Nsol)+"_";
      domain3 D3(Nx,Nz,Length,ratio_m,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost,mpirun_flag);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_deterministic(rc,Nsol);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 11 arguments to pass to the code: Nx, Length, tf, dt, ratio_mass, output_profile, output_profile_radial, initial_cond, start_from_backup, rc, Nsol" << endl;
  }
  else if (world_rank==0){
    cout<< "String in 5th position does not match any possible initial conditions; possible initial conditions are:" << endl;
    cout<< "Schive , Mocz , deterministic , levkov, 1Sol" <<endl;
  }
  if(mpirun_flag==true){
    MPI_Finalize();
  }
  return 0;
}


