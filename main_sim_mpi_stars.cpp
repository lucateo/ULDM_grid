
#include "uldm_mpi_2field.h"
#include "uldm_stars.h"
#include <boost/multi_array.hpp>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "eddington.h"
//To make it run, an example: mpirun -np 4 main_sim_mpi 128 100 1000 1 1Sol false 4

// Remember to change string_outputname according to where you want to store output files, or
// create the directory BEFORE running the code
int main(int argc, char** argv){
  // Inputs
  string name_input_code = argv[1];
  vector<string> params_sim; // Parameters of the run
  vector<string> params_initial_cond; // Parameters of the initial condition
  vector<string> hyperparams; // Other parameters of the code which usually are not touched
  ifstream file(name_input_code);
  int whichline = 0; // Helps in selecting the correct array to put input parameters on
  if (file.is_open()) {
    string line;
    while (std::getline(file, line)) {
      // using printf() in all tests for consistency
      if (line[0] !='#'){
        stringstream linestream; 
        linestream<<line ;
        string w;
        while (linestream >> w) {
          if(whichline==0){ 
            params_sim.push_back(w);
            // cout<< w << " " << whichline<< endl;
          }
          else if (whichline==1){ 
            params_initial_cond.push_back(w);
            // cout<< w << " " << whichline<< endl;
          }
          else if (whichline==2){ 
            // cout<< w << " " << whichline<< endl;
            hyperparams.push_back(w);
          }
        }
        whichline++;
      }
    }
    file.close();
  }

  string mpi_string = hyperparams[0]; // string to check if mpirun is used, put 1 to use mpi
  string grid3d_string = hyperparams[1]; // string to check if one should output 3d grid
  string phase_string = hyperparams[2]; // string to check if one should output phase of the field info
  int reduce_grid = stoi(hyperparams[3]); // Reduce resolution of the 3D output grid (needed if grid3D=true)
  string adaptive_string = hyperparams[4]; // Adaptive timestep flag; if 1, uses adaptive timestep
  string energy_spectrum = hyperparams[5]; // string to check if one wants to compute the energy spectrum

  bool mpirun_flag;
  istringstream(mpi_string)>>mpirun_flag;
  bool grid3d_flag;
  istringstream(grid3d_string)>>grid3d_flag;
  bool phase_flag;
  istringstream(phase_string)>>phase_flag;
  bool adaptive_flag;
  istringstream(adaptive_string)>>adaptive_flag;
  bool spectrum_flag;
  istringstream(energy_spectrum)>>spectrum_flag;

  cout<< mpirun_flag << " "<< grid3d_flag << " "<< phase_flag << " " << adaptive_flag <<endl;
  
  int num_fields = stod(params_sim[0]); // number of fields
  int Nx = stod(params_sim[1]); //number of gridpoints
  double Length= stod(params_sim[2]); //box Length in units of m
  double dt= stod(params_sim[3]);//timeSpacing,  tf/Nt
  int numsteps= stod(params_sim[4]);//Number of steps
  int outputnumb=stod(params_sim[5]);// Number of steps before outputs the sliced (if Grid3D==false) 
                                    //or full 3D grid (if Grid3D==true) density profile (for animation)
                                    // Ensure it is a multiple of outputnumb_profile to avoid backup inconsistencies
  int outputnumb_profile=stod(params_sim[6]);//number of steps before outputs for radial profiles and stores backup
  string initial_cond = params_sim[7]; // String which specifies the initial condition
  string start_from_backup = params_sim[8]; // true or false, depending on whether you want to start from a backup or not
  string directory_name = params_sim[9]; // Directory name
  int num_stars = stoi(params_sim[10]); // Directory name
  
  // Chek that parameters are loaded correctly
  cout<< num_fields << " " << Nx << " " << Length << " " << dt << " "<< numsteps << " " << outputnumb << " " << outputnumb_profile <<
  " "<< initial_cond << " " << start_from_backup<< endl;
  
  // mpi 
  int provided;
  int world_rank;
  int world_size;
  int nghost=2; // number of ghost cells on the psi grids, 2 is the usual and the code will probably break with any other value
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
        cout<< "World size must be greater than 1 for mpi";
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
  string outputname;
  if (mpirun_flag==true) outputname = "_mpi_";
  else outputname = "_";
  outputname = outputname+"nfields_"+to_string(num_fields)+"_Nx" + to_string(Nx) + "_L_" + to_string(Length)+ "_";
  
  domain_stars D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, Pointsmax, world_rank,world_size,nghost, mpirun_flag, num_stars);

  // Apply initial conditions
  bool run_ok = true;

  if (initial_cond == "stars" ) {// Stars in ULDM background
    if (params_initial_cond.size() > 2*num_fields-2){
      multi_array<double, 1> Nparts(extents[num_fields]);
      outputname=directory_name+"stars_levkov_mpi" + outputname;
      for (int i=0; i<num_fields;i++){
        Nparts[i] = stod(params_initial_cond[i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = stod(params_initial_cond[num_fields-1+i]);
        }
      }
      
      multi_array<double, 2> stars_arr(extents[2][num_stars]);
      ifstream infile = ifstream(outputname+"stars_backup.txt");
      int l = 0;
      int star_i = 0;
      string temp;
      while (std::getline(infile, temp, ' ')) {
        double num = stod(temp);
        if(l<7){ // loop over single star feature
          stars_arr[star_i][l] = num;
          cout<< stars_arr[star_i][l] << " " << star_i << " " << l << endl;
          l++;
        }
        if(l==7){ // loop over
          star_i++;
          l=0;
        }
      }

      D3.set_output_name(outputname);
      D3.put_initial_stars(stars_arr);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_waves_Levkov(Nparts[i], i);
        }
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2*nfields arguments to pass to the code: {Npart}, {ratio_mass except for field 0}" << endl;
    }
  }
  
  else if (initial_cond == "stars_soliton" ) {// Stars in ULDM background
    if (params_initial_cond.size() > 1){
      outputname= "out_stars/stars_soliton"+outputname;
      double rc = stod(params_initial_cond[0]); // core radius of soliton
      outputname = outputname + "rc_"+to_string(rc)+ "_Nstars_"+to_string(num_stars) + "_";
      
      multi_array<double, 2> stars_arr(extents[num_stars][7]);
      ifstream infile = ifstream(outputname+"stars_backup.txt");
      int l = 0;
      int star_i = 0;
      string temp;
      while (std::getline(infile, temp, ' ')) {
        double num = stod(temp);
        if(l<7){ // loop over single star feature
          stars_arr[star_i][l] = num;
          cout<< stars_arr[star_i][l] << " " << star_i << " " << l << endl;
          l++;
        }
        if(l==7){ // loop over
          star_i++;
          l=0;
        }
      }
      D3.put_initial_stars(stars_arr);
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
          D3.setInitialSoliton_1(rc, 0);
        }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2 arguments to pass to the code: rc, Nstars" << endl;
    }
  }

  else{
    run_ok=false; 
    if (world_rank==0){
      cout<< "String in 8th position does not match any possible initial conditions; possible initial conditions are:" << endl;
      cout<< "NFW, NFW_ext_Eddington, NFW_solitons"<<endl;;
    }
  }

  if(run_ok==true){
      D3.set_grid(grid3d_flag);
      D3.set_grid_phase(phase_flag); // It will output 2D slice of phase grid
      D3.set_backup_flag(backup_bool);
      D3.set_reduce_grid(reduce_grid);
      D3.set_adaptive_dt_flag(adaptive_flag);
      D3.set_spectrum_flag(spectrum_flag);
      D3.solveConvDif();
  }

  if(mpirun_flag==true){
    MPI_Finalize();
  }
  return 0;
}