#include "uldm_mpi_2field.h"
#include "uldm_sim_nfw.h"
#include "uldm_stars.h"
#include <boost/multi_array.hpp>
#include <cmath>
#include <cstdlib>
#include <string>
#include "eddington.h"
//To make it run, an example: mpirun -np 4 main_sim_mpi 128 100 1000 1 1Sol false 4

// Remember to change string_outputname according to where you want to store output files, or
// create the directory BEFORE running the code
int main(int argc, char** argv){
  // Inputs
  string mpi_string = argv[1]; // string to check if mpirun is used, put 1 to use mpi
  int Nx = atof(argv[2]); //number of gridpoints
  double Length= atof(argv[3]); //box Length in units of m
  int numsteps= atof(argv[4]);//Number of steps
  double dt= atof(argv[5]);//timeSpacing,  tf/Nt
  int num_fields = atof(argv[6]); // number of fields
  int outputnumb=atof(argv[7]);// Number of steps before outputs the sliced (if Grid3D==false) 
                                    //or full 3D grid (if Grid3D==true) density profile (for animation)
                                    // Ensure it is a multiple of outputnumb_profile to avoid backup inconsistencies
  int outputnumb_profile=atof(argv[8]);//number of steps before outputs for radial profiles and stores backup
  string initial_cond = argv[9]; // String which specifies the initial condition
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
    int threads_ok = provided >= MPI_THREAD_FUNNELED;
    cout<< "threads ok? "<<threads_ok<<endl;
    fftw_init_threads();
    fftw_mpi_init();
    // Find out rank of the particular process and the overall number of processes being run
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if(world_rank==0){cout<<" world size is "<<world_size<<endl;}
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1); }
  }
  else { // If mpi is not true
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
  double dx = Length/Nx; //latticeSpacing, dx=L/Nx, L in units of m
  int Pointsmax = Nx/2; //Number of maximum points which are plotted in profile function
  bool backup_bool = false;
  if (start_from_backup == "true")
    backup_bool = true;
  multi_array<double, 1> ratio_mass(extents[num_fields]); // ratio of masses wrt field 0
  ratio_mass[0] = 1.0; // ratio of mass of field 0 is always 1

  // Apply initial conditions
  if (initial_cond == "levkov" ) {// Levkov initial conditions
    if (argc > 10 +2*num_fields-1){
      multi_array<double, 1> Nparts(extents[num_fields]); // Array of number of particls for the different fields
      string outputname;
      if (mpirun_flag==true)
        outputname = "out_levkov_new/Levkov_mpi_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      else
        outputname = "out_levkov_new/Levkov_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      // string outputname = "out_levkov/out_2fields_Levkov_nopsisqmean_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
      //   + "_";
      for (int i=0; i<num_fields;i++){
        Nparts[i] = atof(argv[11+i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = atof(argv[10+num_fields+i]);
          outputname = outputname + "rmass"+to_string(i) +"_"+ to_string(ratio_mass[i]) + "_";
        }
      }
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_waves_Levkov(Nparts[i], i);
        }
      }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 9 +2*num_fields arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, {Npart}, {ratio_masses}" << endl;
  }

  else if (initial_cond == "delta" ) {// Dirac delta on Fourier initial conditions
    if (argc > 10 +2*num_fields-1){
      multi_array<double, 1> Nparts(extents[num_fields]);
      string outputname;
      if (mpirun_flag==true)
        outputname = "out_delta/Delta_mpi_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      else
        outputname = "out_delta/Delta_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      // string outputname = "out_levkov/out_2fields_Levkov_nopsisqmean_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
      //   + "_";
      for (int i=0; i<num_fields;i++){
        Nparts[i] = atof(argv[11+i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = atof(argv[10+num_fields+i]);
          outputname = outputname + "rmass"+to_string(i) +"_"+ to_string(ratio_mass[i]) + "_";
        }
      }
      // Check that the inequality Nx > L^2r/4pi^2 is respected
      for (int i=0; i<num_fields;i++)
        if (ratio_mass[i]*Length*Length/(2*M_PI*M_PI) >= Nx){
          cout << "Error: parameters should respect the inequality Nx > L^2r/4pi^2" << endl;
          return 1;
        }
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_delta(Nparts[i], i);
        }
      }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 9 +2*num_fields arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, {Npart}, {ratio_masses}" << endl;
  }

  else if (initial_cond == "theta" ) {// Heaviside on Fourier initial conditions
    if (argc > 10 +2*num_fields-1){
      multi_array<double, 1> Nparts(extents[num_fields]);
      string outputname;
      if (mpirun_flag==true)
        outputname = "out_theta/Theta_mpi_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      else
        outputname = "out_theta/Theta_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      // string outputname = "out_levkov/out_2fields_Levkov_nopsisqmean_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
      //   + "_";
      for (int i=0; i<num_fields;i++){
        Nparts[i] = atof(argv[11+i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = atof(argv[10+num_fields+i]);
          outputname = outputname + "rmass"+to_string(i) +"_"+ to_string(ratio_mass[i]) + "_";
        }
      }
      // Check that the inequality Nx > L^2r/4pi^2 is respected
      for (int i=0; i<num_fields;i++)
        if (ratio_mass[i]*Length*Length/(2*M_PI*M_PI) >= Nx){
          cout << "Error: parameters should respect the inequality Nx > L^2r/4pi^2" << endl;
          return 1;
        }
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_theta(Nparts[i], i);
        }
      }
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
      for (int i = 0; i <num_fields; i++)
        ratio_mass[i] = 1; // Initialize all to one as a first step
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


  else if (initial_cond == "eddington_nfw_soliton" ) {// NFW Eddington initial conditions + soliton, for text purposes
    if (argc > 16){
      string outputname;
      if (mpirun_flag==true)
        outputname = "out_Eddington/NFW_Soliton_mpi_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      else
        outputname = "out_Eddington/NFW_Soliton_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      ratio_mass[1] = atof(argv[11]); // If nfields=1, this should be set to one
      double rs = atof(argv[12]); // NFW scale radius
      double rhos = atof(argv[13]); // NFW normalization
      double rc = atof(argv[14]); // Core radius
      outputname = outputname + "rmass_"+to_string(ratio_mass[1]) + "_rs_" + to_string(rs) + "_rhos_" + to_string(rhos)
        + "_rc_"+to_string(rc)+ "_";
      int num_k = atoi(argv[15]); // This will be the bool for the simplify_k in SetEddington function
      bool boolk = false; // simplify k sum always false
      
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
          NFW profile = NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
          Eddington eddington = Eddington(&profile);
          D3.setEddington(&eddington, 500, Length/Nx, Length, 1, ratio_mass[1], num_k,boolk); // The actual max radius is between Length and Length/2
          D3.setInitialSoliton_1(rc, 0);
        }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 15 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, ratio_mass, rs, rhos, rc, num_k" << endl;
  }


  else if (initial_cond == "eddington_nfw" ) {// NFW Eddington initial conditions
    if (argc > 14+3*(num_fields-1)){
      string outputname;
      double rs [num_fields];
      double rhos [num_fields];
      if (mpirun_flag==true)
        outputname = "out_Eddington/Eddington_NFW_mpi_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      else
        outputname = "out_Eddington/Eddington_NFW_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      for(int i = 0; i < num_fields; i++){
        ratio_mass[i] = atof(argv[11 +3*i]); // If nfields=1, this should be set to one
        rs[i] = atof(argv[12 + 3*i]); // NFW scale radius
        rhos[i] = atof(argv[13 + 3*i]); // NFW normalization
        outputname = outputname + "rmass_"+to_string(ratio_mass[i]) + "_rs_" + to_string(rs[i]) + "_rhos_" + to_string(rhos[i])+ "_";
      }
      int num_k = atoi(argv[14 + 3*(num_fields-1)]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i< num_fields; i++){
          NFW profile = NFW(rs[i], rhos[i], Length, true);// The actual max radius is between Length and Length/2
          Eddington eddington = Eddington(&profile);
          D3.setEddington(&eddington, 500, Length/Nx, Length, i, ratio_mass[i], num_k, boolk); // The actual max radius is between Length and Length/2
        }
      }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 14 + (3*nfields-1) arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, [ratio_mass[i], rs[i], rhos[i]], num_k" << endl;
  }


  else if (initial_cond == "eddington_plummer" ) {// Plummer Eddington initial conditions
    if (argc > 15){
      int field_id = atoi(argv[11]); // The field where to put the Eddington generated NFW profile
      ratio_mass[field_id] = atof(argv[12]); // If nfields=1, this should be set to one
      string outputname;
      if (mpirun_flag==true)
        outputname = "out_Eddington/Eddington_Plummer_mpi_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      else
        outputname = "out_Eddington/Eddington_Plummer_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_";
      double rs = atof(argv[13]); // Plummer scale radius
      double m0 = atof(argv[14]); // Plummer mass normalization
      outputname = outputname + "rs_" + to_string(rs) + "_M0_" + to_string(m0)+ "_";
      
      int num_k = atoi(argv[15]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      Plummer profile = Plummer(rs, m0, Length, true);// The actual max radius is between Length and Length/2
      Eddington eddington = Eddington(&profile);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setEddington(&eddington, 500, Length/Nx, Length, field_id, ratio_mass[field_id], num_k, boolk); // The actual max radius is between Length and Length/2
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 15 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, field_id, ratio_mass, rs, M0, simplify_ksum" << endl;
  }


  else if (initial_cond == "eddington_nfw_levkov" ) {// NFW Eddington initial conditions for field 1 plus levkov for field 0
    if (argc > 15){
      double Nparts = atof(argv[11]); // Levkov initial condition parameter
      ratio_mass[1] = atof(argv[12]); // If nfields=1, this should be set to one
      string outputname;
      if (mpirun_flag==true)
        outputname = "out_Eddington/NFW_Levkov_mpi_nfields_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_Nparts_" + to_string(Nparts) + "_ratiomass_"+ to_string(ratio_mass[1]) + "_";
      else
        outputname = "out_Eddington/NFW_Levkov_nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_Nparts_" + to_string(Nparts) + "_ratiomass_"+ to_string(ratio_mass[1]) + "_";
      double rs = atof(argv[13]); // NFW scale radius
      double rhos = atof(argv[14]); // NFW normalization
      // double rmax = atof(argv[15]); // NFW max radius
      outputname = outputname + "rs_" + to_string(rs) + "_rhos_" + to_string(rhos)+ "_";
      
      int num_k = atoi(argv[15]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      
      domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, world_rank,world_size,nghost, mpirun_flag);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      NFW profile = NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
      Eddington eddington = Eddington(&profile);
      
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        D3.setEddington(&eddington, 500, Length/Nx, Length, 1, ratio_mass[1], num_k, boolk); // The actual max radius is between Length and Length/2
        D3.set_waves_Levkov(Nparts, 0);
      }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 15 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, Npart, ratio_mass, rs, rhos, num_k" << endl;
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
        D3.setManySolitons_same_radius(Nsol,rc,length_lim, 0);
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
        D3.setManySolitons_random_radius(Nsol,min_radius,max_radius,length_lim, 0);
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
        D3.setManySolitons_deterministic(rc,Nsol, 0);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 12 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rc, Nsol" << endl;
  }
  

  else if (initial_cond == "NFW" ) {// External NFW initial conditions
    if (argc > 11 + 2*num_fields){
      multi_array<double, 1> Nparts(extents[num_fields]);
      double rho0_tilde = atof(argv[11]);//Adimensional rho_0 for the NFW external potential
      double rs_nfw = atof(argv[12]); // Number of particles
      string outputname;
      if (mpirun_flag == true)
        outputname = "out_ext_nfw/out_nfw_mpi_NoCenterAverage_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_Rs_" + to_string(rs_nfw) + "_rh0tilde_" + to_string(rho0_tilde)+ "_Npart_"
            +"_";
      else
        outputname = "out_ext_nfw/out_nfw_NoCenterAverage_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_Rs_" + to_string(rs_nfw) + "_rh0tilde_" + to_string(rho0_tilde)+ "_Npart_"
            +"_";
      for (int i=0; i<num_fields;i++){
        Nparts[i] = atof(argv[13+i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      NFW profile = NFW(rs_nfw, rho0_tilde, Length, true);
      domain_ext D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, 
        world_rank,world_size,nghost,mpirun_flag, &profile);
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = atof(argv[12+num_fields+i]);
        }
      }
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_waves_Levkov(Nparts[i], i);
        }
      }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 11 + 2*nfields arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rho0_tilde, Rs, {Npart}" << endl;
  }
  
  
  else if (initial_cond == "NFW_solitons" ) {// External NFW initial conditions with multi solitons
    if (argc > 13){
      double rho0_tilde = atof(argv[11]);//Adimensional rho_0 for the NFW external potential
      double rs_nfw = atof(argv[12]); // Number of particles
      int Nsol = atoi(argv[13]); // Number of solitons
      double rc = atof(argv[14]); // core radius of solitons
      string outputname;
      if (mpirun_flag == true)
        outputname = "out_ext_nfw/out_nfw_solitons_mpi_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_Rs_" + to_string(rs_nfw) + "_rh0tilde_" + to_string(rho0_tilde)
            +"_";
      else
        outputname = "out_ext_nfw/out_nfw_solitons_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_Rs_" + to_string(rs_nfw) + "_rh0tilde_" + to_string(rho0_tilde)
            +"_";
      outputname = outputname + "Nsol_"+ to_string(Nsol)+ "_rc_"+ to_string(rc) + "_";
      NFW profile = NFW(rs_nfw, rho0_tilde, Length, true);
      domain_ext D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, 
        world_rank,world_size,nghost,mpirun_flag, &profile);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_same_radius(Nsol, rc, Length/2, 0);
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 14 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rho0_tilde, Rs, Nsol, rc" << endl;
  }
  

  else if (initial_cond == "NFW_ext_Eddington" ) {// External NFW initial conditions with Eddington NFW profile
    if (argc > 14){
      double rho0_tilde = atof(argv[11]);//Adimensional rho_0 for the NFW external potential
      double rs_nfw = atof(argv[12]); // Adimensional rs
      double rho_edd = atof(argv[13]); // Eddington rhos
      double rs_edd = atof(argv[14]); // Eddington rs
      int num_k = atoi(argv[15]); // Number of k in Eddington k sum
      string outputname;
      if (mpirun_flag == true)
        outputname = "out_ext_nfw/out_nfwExt_Eddington_NFW_mpi_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_RsExt_" + to_string(rs_nfw) + "_rh0Ext_" + to_string(rho0_tilde)
            +"_Rs_" + to_string(rs_edd) + "_rh00_" + to_string(rho_edd) + "_";
      else
        outputname = "out_ext_nfw/out_nfwExt_Eddington_NFW_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_RsExt_" + to_string(rs_nfw) + "_rh0Ext_" + to_string(rho0_tilde)
            +"_Rs_" + to_string(rs_edd) + "_rh00_" + to_string(rho_edd) + "_";
      
      NFW profile_ext = NFW(rs_nfw, rho0_tilde, Length, true);
      NFW profile_edd = NFW(rs_edd, rho_edd, Length, true);
      Eddington eddington = Eddington(&profile_edd);
      domain_ext D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, 
        world_rank,world_size,nghost,mpirun_flag, &profile_ext);
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setEddington(&eddington, 500, Length/Nx, Length, 0, ratio_mass[0], num_k, false); // The actual max radius is between Length and Length/2
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 15 arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, rho0_tilde ext, Rs ext, rho0_tilde Eddington, Rs Eddington, num_k" << endl;
  }
  
  
  else if (initial_cond == "stars" ) {// Stars in ULDM background
    if (argc > 9 + 2*num_fields){
      multi_array<double, 1> Nparts(extents[num_fields]);
      string outputname;
      if (mpirun_flag == true)
        outputname = "out_stars/stars_levkov_mpi_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            +"_";
      else
        outputname = "out_stars/stars_levkov_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            +"_";
      for (int i=0; i<num_fields;i++){
        Nparts[i] = atof(argv[11+i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // Trial with 2 stars
      double stars[2][7] = {{1, 10,20,40, 0.01, -0.003, 0},
                            {1, 10,20,40, 0.01, -0.003, 0}};
      multi_array<double, 2> stars_arr(extents[2][7]);
      for (int i=0; i<stars_arr.shape()[0]; i++)
        for(int j=0; j<stars_arr.shape()[1]; j++){
          stars_arr[i][j] = stars[i][j];
      }
      domain_stars D3(Nx,Nz,Length,num_fields,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, 
        world_rank,world_size,nghost,mpirun_flag, 2);
      D3.put_initial_stars(stars_arr);
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = atof(argv[10+num_fields+i]);
        }
      }
      D3.set_ratio_masses(ratio_mass);
      D3.set_grid(false);
      D3.set_grid_phase(false); // It will output 2D slice of phase grid
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_waves_Levkov(Nparts[i], i);
        }
      }
      D3.set_backup_flag(backup_bool);
      D3.solveConvDif();
    }
    else if (world_rank==0)
      cout<<"You need 9*2*nfields arguments to pass to the code: mpi_bool, Nx, Length, numsteps, dt, num_fields, output_profile, output_profile_radial, initial_cond, start_from_backup string, {Npart}" << endl;
  }
  
  else if (world_rank==0){
    cout<< "String in 9th position does not match any possible initial conditions; possible initial conditions are:" << endl;
    cout<< "Schive , Mocz , deterministic , levkov, delta, theta, 1Sol, NFW, NFW_solitons, NFW_ext_Eddington, eddington_nfw, eddington_nfw_levkov, eddington_nfw_soliton, stars" <<endl;
  }
  if(mpirun_flag==true){
    MPI_Finalize();
  }
  return 0;
}


