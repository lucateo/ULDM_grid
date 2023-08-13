#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include <cstdlib>
#include <string>


//To make it run, an example: mpirun -np 4 main_sim_mpi 128 100 1000 1 1Sol false 4

// Remember to change string_outputname according to where you want to store output files, or
// create the directory BEFORE running the code
int main(int argc, char** argv){
  // Inputs
  string mpi_string = argv[1]; // string to check if mpirun is used, put 1 to use mpi
  int Nx = atof(argv[2]);                       //number of gridpoints
  double Length= atof(argv[3]);                 //box Length in units of m
  int numsteps= atof(argv[4]);                   //Number of steps
  double dt= atof(argv[5]);                     //timeSpacing,  tf/Nt
  int num_fields = atof(argv[6]); // number of fields
  int outputnumb=atof(argv[7]);// Number of outputs for the sliced (if Grid3D==false) or full 3D grid (if Grid3D==true) density profile (for animation)
  int outputnumb_profile=atof(argv[8]);//number of outputs for radial profiles
  string initial_cond = argv[9];
  string start_from_backup = argv[10]; // true or false, depending on whether you want to start from a backup or not
  
  // mpi 
  int provided;
  int world_rank;
  int world_size;
  int nghost=2; // number of ghost cells on the psi grids, 2 is the usual and the code will probably break with any other value
  bool mpirun_flag;
  istringstream(mpi_string)>>mpirun_flag;
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

  // keep all the boxes the same height for simplicity, so change if not divisible
  if(mpirun_flag==true && Nx%world_size!=0){
        if(world_rank==0){ cout<<"warning: space steps not divisible, adjusting"<<endl;}
        int temp= (int) Nx/world_size;
        Nx=temp*world_size;
  }
  // and as a result, points in the short direction
  int Nz= (Nx)/world_size;
  double dx = Length/Nx;                            //latticeSpacing, dx=L/Nx, L in units of m
  int Pointsmax = Nx/2; //Number of maximum points which are plotted in profile function
  // This is to tell which initial condition you want to run
  bool backup_bool = false;
  if (start_from_backup == "true")
    backup_bool = true;
  multi_array<double, 1> ratio_mass(extents[num_fields]); // ratio of masses wrt field 0
  ratio_mass[0] = 1.0; // ratio of mass of field 0 is always 1

  // Apply initial conditions
  if (initial_cond == "levkov" ) {// Levkov initial conditions
    if (argc > 10 +2*num_fields-1){
      multi_array<double, 1> Nparts(extents[num_fields]);
      string outputname = "out_mpi/out_test/out_2fields_Levkov_nopsisqmean_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      for (int i=0; i<num_fields;i++){
        Nparts[i] = atof(argv[11+i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = atof(argv[10+num_fields-1+i]);
        }
      }
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.set_waves_Levkov(Nparts, num_fields);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 9 +2*num_fields arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, {Npart}, {ratio_masses}" << endl;
  }
  else if (initial_cond == "1Sol" ) {// 1 Soliton initial conditions
    if (argc > 13){
      double rc = atof(argv[11]); // radius of soliton
      int whichpsi = atof(argv[12]); // Of which field you put the soliton
      ratio_mass[whichpsi] = atof(argv[13]);
      string outputname = "out_mpi/out_test/out_2fields_1Sol_nopsisqmean_Nx" + to_string(Nx) +"_rmass_"
        + to_string(ratio_mass[whichpsi]) + "_L_" + to_string(Length)
        + "_rc_" + to_string(rc)+ "_field_"+to_string(whichpsi) + "_";
      domain3 D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
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
      cout<<"You need 13 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rc, which_field, ratio_mass[which_field]" << endl;
  }
  else if (initial_cond == "Schive" ) {// Schive initial conditions
    if (argc > 13){
      double rc = atof(argv[11]); // radius of soliton
      int Nsol = atof(argv[12]); // Number of solitons
      double length_lim = atof(argv[13]); // Length lim of span of solitons
      string outputname = "out_mpi/out_test/out_2fields_Schive_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rc_" + to_string(rc)+ "_Nsol_" + to_string(Nsol)+ "_Llim_" + to_string(length_lim)+"_";
      domain3 D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost,mpirun_flag);
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
      cout<<"You need 13 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rc, Num_sol, length_lim" << endl;
  }
  else if (initial_cond == "Mocz" ) {// Mocz initial conditions
    if (argc > 14){
      double min_radius = atof(argv[11]); //min radius of soliton
      double max_radius = atof(argv[12]); //max radius of soliton
      int Nsol = atof(argv[13]); // Number of solitons
      double length_lim = atof(argv[14]); // Length lim of span of solitons
      string outputname = "out_mpi/out_test/out_2fields_Mocz_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rmin_" + to_string(min_radius)+"_rmax_" +to_string(max_radius)+ "_Nsol_" + to_string(Nsol)+ "_Llim_" + to_string(length_lim)+"_";
      domain3 D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost,mpirun_flag);
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
      cout<<"You need 14 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rmin, rmax, Nsol, length_lim" << endl;
  }
  else if (initial_cond == "deterministic" ) {// Mocz initial conditions
    if (argc > 12){
      double rc = atof(argv[11]); //radius of soliton
      int Nsol = atof(argv[12]); // Number of solitons, should not surpass 30
      string outputname = "out_mpi/out_test/out_2fields_deterministic_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rc_" + to_string(rc)+ "_Nsol_" + to_string(Nsol)+"_";
      domain3 D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost,mpirun_flag);
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
      cout<<"You need 12 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rc, Nsol" << endl;
  }
  else if (world_rank==0){
    cout<< "String in 9th position does not match any possible initial conditions; possible initial conditions are:" << endl;
    cout<< "Schive , Mocz , deterministic , levkov, 1Sol" <<endl;
  }
  if(mpirun_flag==true){
    MPI_Finalize();
  }
  return 0;
}


