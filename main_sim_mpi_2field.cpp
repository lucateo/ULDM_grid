#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include <cmath>
#include <cstdlib>
#include <ostream>
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
      if (line[0] !='#'){
        stringstream linestream; 
        linestream<<line ;
        string w;
        while (linestream >> w) {
          if(whichline==0){ 
            params_sim.push_back(w);
          }
          else if (whichline==1){ 
            params_initial_cond.push_back(w);
          }
          else if (whichline==2){ 
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
  
  // Chek that parameters are loaded correctly
  cout<< "Check parameters loaded correctly " << num_fields << " " << Nx << " " << Length << " " << dt << " "<< numsteps << " " << outputnumb << " " << outputnumb_profile <<
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
  srand(time(NULL));

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
  
  domain3 D3(Nx,Nz,Length, num_fields,numsteps,dt,outputnumb, outputnumb_profile, Pointsmax, world_rank,world_size,nghost, mpirun_flag);

  // Apply initial conditions
  bool run_ok = true;
  if (initial_cond == "levkov" ) {// Levkov initial conditions
    if (params_initial_cond.size() > 2*num_fields-1){
      multi_array<double, 1> Nparts(extents[num_fields]); // Array of number of particls for the different fields
      outputname = directory_name+"levkov" + outputname;
      for (int i=0; i<num_fields;i++){
        Nparts[i] = stod(params_initial_cond[i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      for(int i=0; i<num_fields;i++){
        ratio_mass[i] = stod(params_initial_cond[num_fields+i]);
        outputname = outputname + "rmass"+to_string(i) +"_"+ to_string(ratio_mass[i]) + "_";
      }
      D3.set_output_name(outputname);
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
      run_ok = false; 
      if (world_rank==0)
      cout<<"You need 2*num_fields arguments to pass to the code: {Npart}, {ratio_masses}" << endl;
    }
  }

  else if (initial_cond == "delta" ) {// Dirac delta on Fourier initial conditions
    if (params_initial_cond.size() > 2*num_fields-2){
      multi_array<double, 1> Nparts(extents[num_fields]);
      outputname = directory_name+"Delta" + outputname;
      for (int i=0; i<num_fields;i++){
        Nparts[i] = stod(params_initial_cond[i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = stod(params_initial_cond[num_fields+i-1]);
          outputname = outputname + "rmass"+to_string(i) +"_"+ to_string(ratio_mass[i]) + "_";
        }
      }
      // Check that the inequality Nx > L^2r/4pi^2 is respected
      for (int i=0; i<num_fields;i++)
        if (ratio_mass[i]*Length*Length/(2*M_PI*M_PI) >= Nx){
          cout << "Error: parameters should respect the inequality Nx > L^2r/4pi^2" << endl;
          return 1;
        }
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_delta(Nparts[i], i);
        }
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2*num_fields-1 arguments to pass to the code: {Npart}, {ratio_masses}" << endl;
    }
  }


  else if (initial_cond == "theta" ) {// Heaviside on Fourier initial conditions
    if (params_initial_cond.size() > 2*num_fields-2){
      multi_array<double, 1> Nparts(extents[num_fields]);
      outputname=directory_name+"Theta" + outputname;
      for (int i=0; i<num_fields;i++){
        Nparts[i] = stod(params_initial_cond[i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      // 
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = stod(params_initial_cond[num_fields+i-1]);
          outputname = outputname + "rmass"+to_string(i) +"_"+ to_string(ratio_mass[i]) + "_";
        }
      }
      // Check that the inequality Nx > L^2r/4pi^2 is respected
      for (int i=0; i<num_fields;i++)
        if (ratio_mass[i]*Length*Length/(2*M_PI*M_PI) >= Nx){
          cout << "Error: parameters should respect the inequality Nx > L^2r/4pi^2" << endl;
          return 1;
        }
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_theta(Nparts[i], i);
        }
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2*num_fields-1 arguments to pass to the code: {Npart}, {ratio_masses}" << endl;
    }
  }


  else if (initial_cond == "1Sol" ) {// 1 Soliton initial conditions
    if (params_initial_cond.size() > 2){
      double rc = stod(params_initial_cond[0]); // radius of soliton
      int whichpsi = stod(params_initial_cond[1]); // Of which field you put the soliton
      for (int i = 0; i <num_fields; i++)
        ratio_mass[i] = 1; // Initialize all to one as a first step
      ratio_mass[whichpsi] = stod(params_initial_cond[2]);
      outputname = directory_name+"out_1Sol" + outputname +"ratiomass_"+ to_string(ratio_mass[whichpsi])+ "_rc_" 
        + to_string(rc)+ "_field_"+to_string(whichpsi) + "_";
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setInitialSoliton_1(rc, whichpsi);
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 3 arguments to pass to the code: rc, which_field, ratio_mass[which_field]" << endl;
    }
  }


  else if (initial_cond == "eddington_nfw_soliton" ) {// NFW Eddington initial conditions + soliton, for text purposes
    if (params_initial_cond.size() > 4){
      outputname = directory_name+"NFW_Soliton" + outputname;
      ratio_mass[1] = stod(params_initial_cond[0]); // If nfields=1, this should be set to one
      double rs = stod(params_initial_cond[1]); // NFW scale radius
      double rhos = stod(params_initial_cond[2]); // NFW normalization
      double rc = stod(params_initial_cond[3]); // Core radius
      outputname = outputname + "rmass_"+to_string(ratio_mass[1]) + "_rs_" + to_string(rs) + "_rhos_" + to_string(rhos)
        + "_rc_"+to_string(rc)+ "_";
      int num_k = stoi(params_initial_cond[4]); // This will be the bool for the simplify_k in SetEddington function
      bool boolk = false; // simplify k sum always false
      int center = int(Nx/2);
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        NFW *profile = new NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
        Eddington eddington = Eddington(true);
        eddington.set_profile_den(profile);
        eddington.set_profile_pot(profile);
        D3.setEddington(&eddington, 500, Length/Nx, Length, 1, ratio_mass[1], num_k,boolk,center,center,center); // The actual max radius is between Length and Length/2
        D3.setInitialSoliton_1(rc, 0);
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 5 arguments to pass to the code: ratio_mass, rs, rhos, rc, num_k" << endl;
    }
  }


  else if (initial_cond == "eddington_nfw" ) {// NFW Eddington initial conditions
    outputname =directory_name+"Eddington_NFW" + outputname ;
    int nprofile_per_mass [num_fields]; 
    for(int i = 0; i < num_fields; i++){
      ratio_mass[i] = stod(params_initial_cond[2*i]);
      nprofile_per_mass[i] =stod(params_initial_cond[2*i+1]); 
    }
    int total_nprofiles = 0;
    for(int i = 0; i < num_fields; i++){
      total_nprofiles=total_nprofiles+nprofile_per_mass[i];
    }
    if (params_initial_cond.size() > 2*num_fields+ 5*total_nprofiles-1){
      double rs [total_nprofiles];
      double rhos [total_nprofiles];
      int center_x [total_nprofiles];
      int center_y [total_nprofiles];
      int center_z [total_nprofiles];
      for(int i = 0; i < num_fields; i++){
        for(int j=0; j<nprofile_per_mass[i];j++){
          int index_field;
          if(i!=0) index_field =j+ i*nprofile_per_mass[i-1];
          else  index_field =j;
          rs[index_field] = stod(params_initial_cond[2*num_fields+5*index_field]); // NFW scale radius
          rhos[index_field] = stod(params_initial_cond[2*num_fields+1+5*index_field]); // NFW normalization
          center_x[index_field] = stoi(params_initial_cond[2*num_fields+2+5*index_field]); 
          center_y[index_field] = stoi(params_initial_cond[2*num_fields+3+5*index_field]); 
          center_z[index_field] = stoi(params_initial_cond[2*num_fields+4+5*index_field]); 
          outputname = outputname + "rmass_"+to_string(ratio_mass[i]) + "_rs_" + to_string(rs[index_field]) 
            + "_rhos_" + to_string(rhos[index_field])+ "_";
          cout<<"check " << rs[index_field] << " " << rhos[index_field] << " " << center_x[index_field] << " "<< center_y[index_field] << " "
              << center_z[index_field] << endl;
        }
      }
      int num_k = stoi(params_initial_cond[2*num_fields+ 5*total_nprofiles]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for(int i = 0; i < num_fields; i++){
          for(int j=0; j<nprofile_per_mass[i];j++){
            int index_field;
            if(i!=0) index_field =j+ i*nprofile_per_mass[i-1];
            else  index_field =j;
            NFW *profile = new NFW(rs[index_field], rhos[index_field], Length, true);// The actual max radius is between Length and Length/2
            Eddington eddington = Eddington(true);
            eddington.set_profile_den(profile);
            eddington.set_profile_pot(profile);
            D3.setEddington(&eddington, 500, Length/Nx, Length, i, ratio_mass[i], num_k, boolk,
                center_x[index_field], center_y[index_field], center_z[index_field]); // The actual max radius is between Length and Length/2
          }
        }
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need (2*nfields+5*nprofiles) arguments to pass to the code: [ratio_mass[i], nprofile_per_mass]; [rs[i], rhos[i], center_x[i], center_y[i], center_z[i]], num_k" << endl;
    }
  }


  else if (initial_cond == "eddington_nfw_halos" ) {// NFW Eddington initial conditions
    outputname =directory_name+"Eddington_halos_NFW" + outputname ;
    // Total density percentage going to field with ratio mass
    double density_percentage [num_fields];
    for(int i = 0; i < num_fields; i++){
      ratio_mass[i] = stod(params_initial_cond[2*i]);
      density_percentage[i] = stod(params_initial_cond[2*i+1]);
      outputname = outputname + "rmass_"+to_string(ratio_mass[i]) + "_rho_percent_" + to_string(density_percentage[i]) + "_"; 
      cout<<"check ratio mass, density percentage " << ratio_mass[i] << " " << density_percentage[i]  << endl;
    }
    // How many halos, which have num_fields profiles in it, sharing the same potential but different density normalizations
    int num_halos = stod(params_initial_cond[2*num_fields]);
    if (params_initial_cond.size() > 2*num_fields+ 5*num_halos+1){
      double rs [num_halos];
      double rhos [num_halos];
      int center_x [num_halos];
      int center_y [num_halos];
      int center_z [num_halos];
      for(int i = 0; i < num_halos; i++){
        rs[i] = stod(params_initial_cond[2*num_fields+1+5*i]); // NFW scale radius
        rhos[i] = stod(params_initial_cond[2*num_fields+2+5*i]); // NFW normalization
        center_x[i] = stoi(params_initial_cond[2*num_fields+3+5*i]); 
        center_y[i] = stoi(params_initial_cond[2*num_fields+4+5*i]); 
        center_z[i] = stoi(params_initial_cond[2*num_fields+5+5*i]); 
        outputname = outputname + "rs_" + to_string(rs[i]) 
          + "_rhos_" + to_string(rhos[i])+ "_";
        cout<<"check " << rs[i] << " " << rhos[i] << " " << center_x[i] << " "<< center_y[i] << " "
            << center_z[i] << endl;
        }
      
      int num_k = stoi(params_initial_cond[2*num_fields+ 5*num_halos+1]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        for(int i = 0; i < num_halos; i++){
          for(int j=0; j<num_fields;j++){
            NFW *profile_pot = new NFW(rs[i], rhos[i], Length, true);// The actual max radius is between Length and Length/2
            NFW *profile_den = new NFW(rs[i], density_percentage[j]*rhos[i], Length, true);// The actual max radius is between Length and Length/2
            Eddington eddington = Eddington(false);
            eddington.set_profile_den(profile_den);
            eddington.set_profile_pot(profile_pot);
            D3.setEddington(&eddington, 500, Length/Nx, Length, j, ratio_mass[j], num_k, boolk,
                center_x[i], center_y[i], center_z[i]); // The actual max radius is between Length and Length/2
          }
        }
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need (2*nfields+1+5*num_halos+1) arguments to pass to the code: [ratio_mass[i], density_percentage[i]]; num_halos, [rs[i], rhos[i], center_x[i], center_y[i], center_z[i]], num_k" << endl;
    }
  }


  else if (initial_cond == "eddington_plummer" ) {// Plummer Eddington initial conditions
    if (params_initial_cond.size() > 5){
      int field_id = stoi(params_initial_cond[0]); // The field where to put the Eddington generated NFW profile
      ratio_mass[field_id] = stod(params_initial_cond[1]); // If nfields=1, this should be set to one
      outputname = directory_name+"Eddington_Plummer" + outputname;
      double rs = stod(params_initial_cond[2]); // Plummer scale radius
      double m0 = stod(params_initial_cond[3]); // Plummer mass normalization
      int num_k = stoi(params_initial_cond[4]); // Number of maximum k points in nested loop
      outputname = outputname + "rs_" + to_string(rs) + "_M0_" + to_string(m0)+ "_";
      bool boolk = false; // simplify k sum always false

      int center = int(Nx/2);
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      Plummer *profile = new Plummer(rs, m0, Length, true);// The actual max radius is between Length and Length/2
      Eddington eddington = Eddington(true);
      eddington.set_profile_den(profile);
      eddington.set_profile_pot(profile);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setEddington(&eddington, 500, Length/Nx, Length, field_id, 
          ratio_mass[field_id], num_k, boolk,center,center,center); // The actual max radius is between Length and Length/2
    }
    else{
      run_ok =false;
      if (world_rank==0)
        cout<<"You need 5 arguments to pass to the code: field_id, ratio_mass, rs, M0, simplify_ksum" << endl;
    }
  }


  else if (initial_cond == "eddington_nfw_levkov" ) {// NFW Eddington initial conditions for field 1 plus levkov for field 0
    if (params_initial_cond.size() > 4){
      double Nparts = stod(params_initial_cond[0]); // Levkov initial condition parameter
      ratio_mass[1] = stod(params_initial_cond[1]); 
      outputname =directory_name+"NFW_Levkov" + outputname ;
      double rs = stod(params_initial_cond[2]); // NFW scale radius
      double rhos = stod(params_initial_cond[3]); // NFW normalization
      // double rmax = stod(params[15]); // NFW max radius
      outputname = outputname +"rmass_"+to_string(ratio_mass[1])+ "_rs_" + to_string(rs) + "_rhos_" + to_string(rhos)+ "_";
      
      int num_k = stoi(params_initial_cond[4]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      int center = int(Nx/2);
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      NFW *profile = new NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
      Eddington eddington = Eddington(true);
      eddington.set_profile_den(profile);
      eddington.set_profile_pot(profile);
      
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else{
        D3.setEddington(&eddington, 500, Length/Nx, Length, 1, ratio_mass[1], num_k, boolk,center,center,center); // The actual max radius is between Length and Length/2
        D3.set_waves_Levkov(Nparts, 0);
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 5 arguments to pass to the code: Npart, ratio_mass, rs, rhos, num_k" << endl;
    }
  }


  else if (initial_cond == "Schive" ) {// Schive initial conditions
    if (params_initial_cond.size() > 3){
      double rc = stod(params_initial_cond[0]); // radius of soliton
      int Nsol = stod(params_initial_cond[1]); // Number of solitons
      double length_lim = stod(params_initial_cond[2]); // Length lim of span of solitons
      double rmass = stod(params_initial_cond[3]); //
      ratio_mass[0] = rmass;
      outputname = directory_name+"out_Schive"+outputname
        + "rc_" + to_string(rc)+ "_Nsol_" + to_string(Nsol)+ "_Llim_" + to_string(length_lim)+"_";
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_same_radius(Nsol,rc,length_lim, 0);
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 4 arguments to pass to the code: rc, Num_sol, length_lim, ratio_mass" << endl;
    }
  }


  else if (initial_cond == "Mocz" ) {// Mocz initial conditions
    if (params_initial_cond.size() > 3){
      double min_radius = stod(params_initial_cond[0]); //min radius of soliton
      double max_radius = stod(params_initial_cond[1]); //max radius of soliton
      int Nsol = stod(params_initial_cond[2]); // Number of solitons
      double length_lim = stod(params_initial_cond[3]); // Length lim of span of solitons
      outputname = directory_name+"out_Mocz"+outputname + "_rmin_" + to_string(min_radius)+"_rmax_" 
        +to_string(max_radius)+ "_Nsol_" + to_string(Nsol)+ "_Llim_" + to_string(length_lim)+"_";
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_random_radius(Nsol,min_radius,max_radius,length_lim, 0);
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 4 arguments to pass to the code: rmin, rmax, Nsol, length_lim" << endl;
    }
  }


  else if (initial_cond == "deterministic" ) {// deterministic initial conditions for tests
    if (params_initial_cond.size() > 2){
      double rc = stod(params_initial_cond[0]); //radius of soliton
      int Nsol = stoi(params_initial_cond[1]); // Number of solitons, should not surpass 30
      ratio_mass[0] =stod(params_initial_cond[2]); 
      outputname = directory_name+"out_deterministic"+ outputname + "ratiomass_" + to_string(ratio_mass[0]) +"_rc_" + to_string(rc)+ "_Nsol_" + to_string(Nsol)+"_";
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setManySolitons_deterministic(rc,Nsol, 0);
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 3 arguments to pass to the code:rc, Nsol, ratio_mass " << endl;
    }
  }
  


  else if (initial_cond == "elliptCollapse" ) {// Elliptical collapse initial conditions
    if (params_initial_cond.size() > 7){
      double Norm = stod(params_initial_cond[0]); // Normalization of profile
      double a_e = stod(params_initial_cond[1]); // parameter elliptical
      double b_e = stod(params_initial_cond[2]); // parameter elliptical
      double c_e = stod(params_initial_cond[3]); // parameter elliptical
      ratio_mass[0] = stod(params_initial_cond[4]); // Of which ratio mass you put the initial condition;
      bool rand_phases;
      istringstream(params_initial_cond[5])>>rand_phases; //If Acorr !=0, if true, use random phases, otherwise uses random field
      double A_corr = stod(params_initial_cond[6]); // Acorr; if Acorr==0, avoid the random procedure completely;
      double lcorr = stod(params_initial_cond[7]); // Correlation length, effective if random==1;
      outputname = directory_name+"out_EllitpCollapse" + outputname + "ratio_mass_"+ to_string(ratio_mass[0])+ "_Norm_" 
        + to_string(Norm)+ "_ae_"+to_string(a_e) +"_be_"+to_string(b_e) +"_ce_"+to_string(c_e) + "_" ;
      if(A_corr != 0){
        outputname = outputname +"rand_phases_"+ to_string(rand_phases) +"_Acorr_"+to_string(A_corr) 
          + "_lcorr_" + to_string(lcorr) + "_";
      }
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.setEllitpicCollapse(Norm,a_e,b_e,c_e,0, rand_phases, A_corr,lcorr);
    }
    else{
      run_ok=false; 
      if (world_rank==0){
        cout<<"You need 8 arguments to pass to the code: Norm, a_e, b_e, c_e, ratio_mass[0], random_phases_int, ";
        cout<< "Acorr, lcorr" <<endl;
      }
    }
  }

  // Static NFW initial conditon, psi = sqrt(rho)
  else if (initial_cond == "staticProfile_NFW" ) {
    if (params_initial_cond.size() > 1){
      double rs = stod(params_initial_cond[0]); // NFW scale radius
      double rhos = stod(params_initial_cond[1]); // NFW normalization
      outputname = directory_name+"out_static_NFW" + outputname+ "rs_" 
        + to_string(rs) + "_rhos_" + to_string(rhos) + "_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      // Set velocity of center of mass to zero
      vector<double> vcm = {0,0,0};
      NFW *nfw_profile = new NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
      if(start_from_backup=="true")
        D3.initial_cond_from_backup();
      else
        D3.set_static_profile(nfw_profile,0, vcm);
    }
    else{
      run_ok=false; 
      if (world_rank==0){
        cout<<"You need 2 arguments to pass to the code: rs, rhos" <<endl;
      }
    }
  }



  else{
    run_ok=false; 
    if (world_rank==0){
      cout<< "String in 8th position does not match any possible initial conditions; possible initial conditions are:" << endl;
      cout<< "Schive , Mocz , deterministic , levkov, delta, theta, 1Sol, NFW, NFW_solitons, NFW_ext_Eddington, eddington_nfw," <<endl;
      cout<<" eddington_nfw_halos, eddington_nfw_levkov, eddington_nfw_soliton, stars, stars_soliton, elliptCollapse," <<endl;
      cout<<" staticProfile_NFW" <<endl;
    }
  }

  if(run_ok==true){
      cout<<"File name of the run: "<< outputname <<endl;
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


  
  
