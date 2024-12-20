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
  int num_stars = stoi(params_sim[10]); // Number of stars
  
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
  D3.set_grid(grid3d_flag);
  D3.set_grid_phase(phase_flag); // It will output 2D slice of phase grid
  D3.set_backup_flag(backup_bool);
  D3.set_reduce_grid(reduce_grid);
  D3.set_adaptive_dt_flag(adaptive_flag);
  D3.set_spectrum_flag(spectrum_flag);

  // Apply initial conditions
  bool run_ok = true;

  if (initial_cond == "stars" ) {// Stars in ULDM background
    if (params_initial_cond.size() > 2*num_fields-2){
      multi_array<double, 1> Nparts(extents[num_fields]);
      outputname=directory_name+"stars_levkov" + outputname;
      for (int i=0; i<num_fields;i++){
        Nparts[i] = stod(params_initial_cond[i]); // Number of particles
        outputname = outputname + "Npart"+to_string(i) +"_"+ to_string(Nparts[i]) + "_";
      }
      if(num_fields > 1){
        for(int i=1; i<num_fields;i++){
          ratio_mass[i] = stod(params_initial_cond[num_fields-1+i]);
        }
      }
      
      multi_array<double, 2> stars_arr = D3.generate_random_stars(1/(2*sqrt(3)));

      D3.set_output_name(outputname);
      D3.put_initial_stars(stars_arr);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        for (int i=0; i<num_fields; i++){
          D3.set_waves_Levkov(Nparts[i], i);
        }
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2*nfields-1 arguments to pass to the code: {Npart}, {ratio_mass except for field 0}" << endl;
    }
  }
  

  else if (initial_cond == "stars_noise" ) {// Stars in Gaussian noise
    if (params_initial_cond.size() > 1){
      outputname=directory_name+"stars_gauss_noise" + outputname;
      double Acorr = stod(params_initial_cond[0]); // Amplitude of the correlation function
      double Lcorr = stod(params_initial_cond[1]); // Length of the correlation function
      multi_array<double, 2> stars_arr = D3.generate_random_stars(1./(2*sqrt(3)*Lcorr));
      outputname = outputname + "Acorr_"+to_string(Acorr)+ "_Lcorr_"+to_string(Lcorr) 
        + "_Nstars_"+to_string(num_stars) + "_";
      D3.set_output_name(outputname);
      D3.put_initial_stars(stars_arr);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true"){
        D3.get_star_backup();
        D3.initial_cond_from_backup();
      }
      else{
        D3.set_Gauss_noise(Acorr, Lcorr,0);
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2 arguments to pass to the code: Acorr, Lcorr " << endl;
    }
  }
  
  else if (initial_cond == "stars_soliton" ) {// Stars in ULDM background
    if (params_initial_cond.size() > 1){
      outputname= directory_name+"stars_soliton"+outputname;
      double rc = stod(params_initial_cond[0]); // core radius of soliton
      double c_pert = stod(params_initial_cond[1]); // perturbation parameter
      outputname = outputname + "rc_"+to_string(rc)+ "_Nstars_"+to_string(num_stars) 
          + "_pert_" + to_string(c_pert) + "_";
      
      multi_array<double, 2> stars_arr(extents[num_stars][8]);
      if(start_from_backup=="false"){
        ifstream infile = ifstream("stars_input.txt");
        string line;
        int star_i = 0;
        // Read the file line by line
        while (getline(infile, line) && star_i < num_stars) {
            stringstream ss(line);
            double num;
            int l = 0;

            // Read each value separated by spaces
            while (ss >> num && l < 8) {
                stars_arr[star_i][l] = num;
                cout << stars_arr[star_i][l] << " " << star_i << " " << l << endl;
                l++;
            }

            star_i++;
        }
        infile.close();
        D3.put_initial_stars(stars_arr);
      }
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
          D3.set1Sol_perturbed(rc, c_pert);
        }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2 arguments to pass to the code: rc, c_perturbation" << endl;
    }
  }

  else if (initial_cond == "uniform_sphere_test" ) {// Uniform sphere for test purposes
    if (params_initial_cond.size() > 0){
      outputname= directory_name+"stars_sphere"+outputname;
      double rho0 = stod(params_initial_cond[0]); // central density
      double rad = stod(params_initial_cond[1]); // radius uniform sphere
      outputname = outputname + "rho0_"+to_string(rho0)+ "_rad_"+to_string(rad) +"_Nstars_"+to_string(num_stars) + "_";
      
      multi_array<double, 2> stars_arr(extents[num_stars][8]);
      if(start_from_backup=="false"){
        ifstream infile = ifstream("stars_input.txt");
        int l = 0;
        int star_i = 0;
        string temp;
        while (std::getline(infile, temp, ' ')) {
          double num = stod(temp);
          if(l<8){ // loop over single star feature; 8-th entry is potential energy, which I set to zero in initial conditions
            stars_arr[star_i][l] = num;
            cout<< stars_arr[star_i][l] << " " << star_i << " " << l << endl;
            l++;
          }
          if(l==8){ // loop over
            star_i++;
            l=0;
          }
        }
        D3.put_initial_stars(stars_arr);
      }
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
          D3.uniform_sphere(rho0, rad);
        }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 2 arguments to pass to the code: rho0, rc" << endl;
    }
  }


  // Static NFW for test purposes
  else if (initial_cond == "uniform_NFW_test" ) {
    if (params_initial_cond.size() > 2){
      outputname= directory_name+"uniform_NFW_test"+outputname;
      double rhos = stod(params_initial_cond[0]); // rhos
      double rs = stod(params_initial_cond[1]); // rs
      double r_plummer = stod(params_initial_cond[2]); // r_plummer
      outputname = outputname + "rho0_"+to_string(rhos)+ "_rs_"+to_string(rs) +"_Nstars_"+to_string(num_stars) + "_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        NFW *profile_nfw = new NFW(rs, rhos, 3*Length, true);// The actual max radius is between Length and Length/2
        // For stars, there is no meaning to put Length as rmax
        // setEddington computes fE_func, so you should run it before
        vector<double> vcm(3,0);
        D3.set_static_profile(profile_nfw,0, vcm);
        D3.set_output_name(outputname);
        D3.set_ratio_masses(ratio_mass);
        // Then generate stars
        if(world_rank==0){
          Eddington eddington_stars = Eddington(false);
          Plummer *profile_plummer = new Plummer(r_plummer, 1, 3*Length, true);
          // If you define a different max length, you should define a new Profile 
          // with that max length
          NFW *profile_nfw_stars = new NFW(rs, rhos, 3*Length, true);
          eddington_stars.set_profile_den(profile_plummer);
          eddington_stars.set_profile_pot(profile_nfw_stars);
          // Stars gravity is not taken into account, so the potential 
          // should be just nfw
          // eddington_stars.set_profile_pot(profile_plummer);
          eddington_stars.generate_fE_arr(1000, Length/Nx, 3*Length, outputname+"stars_");
          vector<double> xmax;
          vector<double> v_cm;
          for (int i=0; i<3; i++){
            xmax.push_back(Length/2);
            v_cm.push_back(D3.v_center_mass(i,0));
            // v_cm.push_back(0);
          }
          if(num_stars==1){
            multi_array<double, 2> stars_arr(extents[num_stars][8]);
            stars_arr[0][0] = 1; stars_arr[0][1] = 15 + Length/2;
            stars_arr[0][2] = -35 + Length/2; stars_arr[0][3] = 3+ Length/2;
            stars_arr[0][4] = -1; stars_arr[0][5] = -0.9; stars_arr[0][6] = 0.66;
            D3.put_initial_stars(stars_arr);
          }
          else
          D3.generate_stars(&eddington_stars,
            Length/Nx, Length/3, xmax, v_cm,0,num_stars);
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup(); 
        }
        D3.get_star_backup();
        }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 3 arguments to pass to the code: rho0, rs, r_plummer" << endl;
    }
  }

  // Static Burkert for test purposes
  else if (initial_cond == "uniform_Burkert_test" ) {
    if (params_initial_cond.size() > 2){
      outputname= directory_name+"uniform_Burkert_test"+outputname;
      double rhos = stod(params_initial_cond[0]); // rhos
      double rs = stod(params_initial_cond[1]); // rs
      double r_plummer = stod(params_initial_cond[2]); // r_plummer
      outputname = outputname + "rho0_"+to_string(rhos)+ "_rs_"+to_string(rs) +"_Nstars_"+to_string(num_stars) + "_";
      
      multi_array<double, 2> stars_arr(extents[num_stars][8]);
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        Burkert *profile_burk = new Burkert(rs, rhos, 3*Length, true);// The actual max radius is between Length and Length/2
        // For stars, there is no meaning to put Length as rmax
        // setEddington computes fE_func, so you should run it before
        vector<double> vcm(3,0);
        D3.set_static_profile(profile_burk,0, vcm);
        D3.set_output_name(outputname);
        D3.set_ratio_masses(ratio_mass);
        // Then generate stars
        if(world_rank==0){
          double length_max_stars= 10*Length;
          D3.put_initial_stars(stars_arr);
          Eddington eddington_stars = Eddington(false);
          Plummer *profile_plummer = new Plummer(r_plummer, 1, length_max_stars, true);
          // If you define a different max length, you should define a new Profile 
          // with that max length
          Burkert *profile_burk_stars = new Burkert(rs, rhos, length_max_stars, true);
          eddington_stars.set_profile_den(profile_plummer);
          eddington_stars.set_profile_pot(profile_burk_stars);
          // Stars gravity is not taken into account, so the potential 
          // should be just nfw
          // eddington_stars.set_profile_pot(profile_plummer);
          eddington_stars.generate_fE_arr(1000, Length/Nx, length_max_stars, outputname+"stars_",Length/3);
          vector<double> xmax;
          vector<double> v_cm;
          for (int i=0; i<3; i++){
            xmax.push_back(Length/2);
            v_cm.push_back(D3.v_center_mass(i,0));
            // v_cm.push_back(0);
          }
          if(num_stars==1){
            multi_array<double, 2> stars_arr(extents[num_stars][8]);
            stars_arr[0][0] = 1; stars_arr[0][1] = 10 + Length/2;
            stars_arr[0][2] = -10 + Length/2; stars_arr[0][3] = 10+ Length/2;
            stars_arr[0][4] = 1; stars_arr[0][5] = 0.2; stars_arr[0][6] = -0.02;
            D3.put_initial_stars(stars_arr);
          }
          else
            D3.generate_stars(&eddington_stars,
              Length/Nx, Length/3, xmax, v_cm,0,num_stars);
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup(); 
        }
        D3.get_star_backup();
        }
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 3 arguments to pass to the code: rho0, rs, r_plummer" << endl;
    }
  }

  // Testing external potential, here I change the step_fourier function to put all the time an external potential
  else if (initial_cond == "test" ) { 
      outputname= directory_name+"test"+outputname;
      outputname = outputname + "_Nstars_"+to_string(num_stars) + "_";
      
      multi_array<double, 2> stars_arr(extents[num_stars][8]);
      ifstream infile = ifstream("stars_input.txt");
      int l = 0;
      int star_i = 0;
      string temp;
      while (std::getline(infile, temp, ' ')) {
        double num = stod(temp);
        if(l<8){ // loop over single star feature; 8-th entry is potential energy, which I set to zero in initial conditions
          stars_arr[star_i][l] = num;
          cout<< stars_arr[star_i][l] << " " << star_i << " " << l << endl;
          l++;
        }
        if(l==8){ // loop over
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
          D3.setTest();
        }
    }

  else if (initial_cond == "stars_eddington_NFW" ) {// Stars in NFW background
    if (params_initial_cond.size() > 2){
      outputname= directory_name+"stars_eddington_NFW"+outputname;
      double rs = stod(params_initial_cond[0]);
      double rhos= stod(params_initial_cond[1]);
      int num_k = stoi(params_initial_cond[2]); // Number of maximum k points in nested loop
      bool boolk = false; // simplify k sum always false
      outputname = outputname+"rs_"+to_string(rs)+"_rhos_"+to_string(rhos)+"_num_stars_"+to_string(num_stars)+"_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        multi_array<double, 2> stars_arr(extents[num_stars][8]);
        D3.put_initial_stars(stars_arr);
        NFW *profile = new NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
        Eddington eddington = Eddington(true);
        eddington.set_profile_den(profile);
        eddington.set_profile_pot(profile);
        // setEddington computes fE_func, so you should run it before
        D3.setEddington(&eddington, 500, Length/Nx, Length, 0, ratio_mass[0], num_k, boolk,
            int(Nx/2),int(Nx/2),int(Nx/2) ); // The actual max radius is between Length and Length/2
        
        // Generate stars
        if(world_rank==0){
          vector<double> xmax = {Length/2, Length/2, Length/2};
          vector<double> v_cm = {0, 0, 0};
          D3.generate_stars(&eddington,Length/Nx, Length/2, xmax, v_cm,0,num_stars);
          D3.put_initial_stars(stars_arr);
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup(); 
        }
      }
      D3.get_star_backup();
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 3 arguments to pass to the code: rs, rhos, num_k" << endl;
    }
  }
  
  // Set an initial halo, let it relax, then put stars
  else if (initial_cond == "halo_stars_delayed_NFW" ) {
    outputname= directory_name+"halo_stars_NFW"+outputname;
    int nhalos = stoi(params_initial_cond[0]);
    if (params_initial_cond.size()> 2 + 4*nhalos) {
      vector<double> rs;
      vector<double> rhos;
      vector<double> r_plummer;
      vector<double> M_plummer;
      vector<int> num_stars_halo;
      
      // Divide the number of stars in the halos
      int num_stars_acc = 0;
      for(int i=0; i<nhalos-1;i++){
        // Divide the number of stars equally in the halos
        int num_stars_here = int(num_stars/nhalos);
        num_stars_acc += num_stars_here;
        // num_stars_halo contains the final index of the stars in each halo
        num_stars_halo.push_back(num_stars_acc);
      }
      num_stars_halo.push_back(num_stars);

      for (int halo_num =0; halo_num<nhalos;halo_num++){
        // NFW parameters
        rs.push_back(stod(params_initial_cond[1+4*halo_num]));
        rhos.push_back(stod(params_initial_cond[2+4*halo_num]));
        // Plummer parameters
        r_plummer.push_back(stod(params_initial_cond[3+4*halo_num]));
        M_plummer.push_back(stod(params_initial_cond[4+4*halo_num]));
      }
      int num_k = stoi(params_initial_cond[5+4*(nhalos-1)]); // Number of maximum k points in nested loop
      int numstep_relax = stoi(params_initial_cond[6 + 4*(nhalos-1)]); // Number of steps to relax the halo
      double vcm_halo = 0;
      bool boolk = false; // simplify k sum always false
      for(int i=0; i<nhalos;i++){
        outputname = outputname+"rs_"+to_string(rs[i])+"_rhos_"+to_string(rhos[i])+
          "_r_plummer_"+to_string(r_plummer[i])+"_M_plummer_"+to_string(M_plummer[i])
          +"_";
      }
      // If it is there on the input parameters, use vcm_halo to set the relative halos velocities
      if(params_initial_cond.size() > 7 + 4*(nhalos-1)){
        vcm_halo = stod(params_initial_cond[7 + 4*(nhalos-1)]); // Center of mass velocity of the halo
        outputname = outputname + "vcm_halo_"+to_string(vcm_halo)+"_";
      }
      outputname = outputname+"num_stars_"+to_string(num_stars)+"_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        multi_array<double, 2> stars_arr(extents[num_stars][8]);
        D3.put_initial_stars(stars_arr);
        double length_max_halo = 4*Length;
        // Initialize vectors to zero with specified size
        vector<int> x_center(nhalos, 0), y_center(nhalos, 0), z_center(nhalos, 0);
        vector<double> vcm_halox(nhalos, 0.0), vcm_haloy(nhalos, 0.0), vcm_haloz(nhalos, 0.0);
        for(int i=0;i<nhalos;i++){
          NFW *profile_nfw = new NFW(rs[i], rhos[i], length_max_halo, true);// The actual max radius is between Length and Length/2
          // For stars, there is no meaning to put Length as rmax
          Eddington eddington_nfw = Eddington(true);
          eddington_nfw.set_profile_den(profile_nfw);
          eddington_nfw.set_profile_pot(profile_nfw);
          // Just sets the halo in the center of the box if there is only one halo
          if(nhalos ==1) {
            x_center[i] =int(Nx/2); 
            y_center[i] =int(Nx/2); 
            z_center[i] =int(Nx/2);
            vcm_halox[i]=0; vcm_haloy[i]=0; vcm_haloz[i]=0;
          }
          else{
            // Random velocity angle in the xy plane, leave the z component to zero
            double rand_phi = fRand(0,2*M_PI);
            x_center[i]=int(fRand(Nx/4,3*Nx/4)); 
            y_center[i]=int(fRand(Nx/4,3*Nx/4)); 
            z_center[i]=int(fRand(Nx/4,3*Nx/4));
            vcm_halox[i]=vcm_halo*cos(rand_phi);
            vcm_haloy[i]=vcm_halo*sin(rand_phi);
            vcm_haloz[i]=0;
            cout<<"vcm_halo "<<vcm_halo<<" "<<vcm_halox[i]<<" "<<vcm_haloy[i]<<" "<<vcm_haloz[i]<<endl;
          }
          // Random direction of the vcm_halo on the xy plane
          vector<double> vcm_halo_vect = {vcm_halox[i], vcm_haloy[i], vcm_haloz[i]};
          // setEddington computes fE_func, so you should run it before
          D3.setEddington(&eddington_nfw, 1000, Length/Nx, length_max_halo, 0, ratio_mass[0], num_k, boolk,
              x_center[i],y_center[i],z_center[i], vcm_halo_vect ); // The actual max radius is between Length and Length/2
        }
        // Print the full initial snapshot, it prints in a snapshot file and not
        // in the backup file
        ofstream psi_snapshot;
        D3.outputfullPsi(psi_snapshot,false,1);
        // if numstep_relax==0, do not do evolution without stars
        // Also if number of halos is greater than one, do not relax the halo
        if(numstep_relax>0 && nhalos==1){
          // Sets the number of steps to relax the halo
          D3.set_numsteps(numstep_relax);
          D3.put_numstar_eff(0);
          // Run until relaxing time
          D3.solveConvDif();
        }
        D3.set_numsteps(numsteps);
        D3.put_numstar_eff(num_stars);
        // Then generate stars
        if(world_rank==0){
          double Length_max_stars= 10*Length;
          for (int id_halo =0; id_halo< nhalos;id_halo++){
            Eddington eddington_stars = Eddington(false);
            Plummer *profile_plummer = new Plummer(r_plummer[id_halo], 
                M_plummer[id_halo], Length_max_stars, true);
            // // Compute the actual density profile
            // multi_array<double,2> density_profile = D3.profile_density(0);
            // vector<double> radius_profile;
            // vector<double> density_profile_vec;
            // for(int i=0; i<density_profile.shape()[1]; i++){
            //   radius_profile.push_back(density_profile[0][i]);
            //   density_profile_vec.push_back(density_profile[1][i]);
            //   // cout<<radius_profile[i]<<" "<<density_profile_vec[i]<<endl;
            // }
            // Generic profile has still issues from usual non-monotonous density profiles
            // Generic *profile_ext_stars = new Generic(density_profile_vec,radius_profile, Length/3, true);
            // If you define a different max length, you should define a new Profile 
            // with that max length
            NFW *profile_nfw_stars = new NFW(rs[id_halo], rhos[id_halo], Length_max_stars, true);
            eddington_stars.set_profile_den(profile_plummer);
            eddington_stars.set_profile_pot(profile_nfw_stars);
            // Stars gravity is not taken into account, so the potential 
            // should be just nfw
            // eddington_stars.set_profile_pot(profile_plummer);
            eddington_stars.generate_fE_arr(1000, Length/Nx, Length_max_stars,outputname+"stars_");
            vector<double> xmax;
            // Call this function to set the maximum density location in D3
            double max_density = D3.find_maximum(0);
            vector<double> v_cm(3,0);
            if(nhalos==1){
              for (int i=0; i<3; i++){
                  xmax.push_back(D3.get_maxx(0,i)*Length/Nx);
                  v_cm.push_back(D3.v_center_mass(i,0));
                  // v_cm.push_back(0);
                }
            }
              // Put just the center of the halos and vcm =vcm_halo if nhalos>1
            else{
              xmax.push_back(x_center[id_halo]*Length/Nx);
              xmax.push_back(y_center[id_halo]*Length/Nx);
              xmax.push_back(z_center[id_halo]*Length/Nx);
              v_cm[0]=vcm_halox[id_halo]; 
              v_cm[1]=vcm_haloy[id_halo]; 
              v_cm[2]=vcm_haloz[id_halo];
            }
            // Maximum length in profiles and here does not need to be the same
            // multi_array<double,2> stars_arr = D3.generate_stars(&eddington_stars,
            //   Length/Nx, Length/3, xmax, v_cm);
            int start_star;
            if(id_halo==0) start_star = 0;
            else start_star = num_stars_halo[id_halo-1];
            int end_star = num_stars_halo[id_halo];
            D3.generate_stars(&eddington_stars,
              Length/Nx, Length/3, xmax, v_cm, start_star, end_star);
          }
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup();
        }
        D3.get_star_backup();
      }
      // After, if start_from_backup is false, it will not save the relaxing of halo part
    }
    else{
      run_ok=false; 
      if (world_rank==0){
        cout<<"You need 3+ 4*n_halo arguments to pass to the code: n_halo, [rs, rhos, r_plummer, M_plummer], num_k, numstep_relax" << endl;
        cout<<"If you want to set the velocity of the halo, you need to pass vcm_halo as the last argument" << endl;
      }
    }
  }


  // Set an initial halo with anisotropy, let it relax, then put stars
  else if (initial_cond == "halo_stars_anisotropy_NFW" ) {
    outputname= directory_name+"halo_stars_anisotropy_NFW"+outputname;
    if (params_initial_cond.size()> 6) {
      // NFW parameters
      double rs = stod(params_initial_cond[0]);
      double rhos =stod(params_initial_cond[1]);
      // Plummer parameters
      double r_plummer =stod(params_initial_cond[2]);
      double M_plummer=stod(params_initial_cond[3]);
      double beta_ani = stod(params_initial_cond[4]);
      int num_k = stoi(params_initial_cond[5]); // Number of maximum k points in nested loop
      int numstep_relax = stoi(params_initial_cond[6]); // Number of steps to relax the halo
      bool boolk = false; // simplify k sum always false
      outputname = outputname+"rs_"+to_string(rs)+"_rhos_"+to_string(rhos)+
        "_r_plummer_"+to_string(r_plummer)+"_M_plummer_"+to_string(M_plummer)
        +"_beta_"+to_string(beta_ani)+"_num_stars_"+to_string(num_stars)+"_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        multi_array<double, 2> stars_arr(extents[num_stars][8]);
        D3.put_initial_stars(stars_arr);
        double length_max_halo = 4*Length;
        NFW *profile_nfw = new NFW(rs, rhos, length_max_halo, true);// The actual max radius is between Length and Length/2
        // For stars, there is no meaning to put Length as rmax
        Eddington eddington_nfw = Eddington(true, beta_ani);
        eddington_nfw.set_profile_den(profile_nfw);
        eddington_nfw.set_profile_pot(profile_nfw);
        // setEddington computes fE_func, so you should run it before
        D3.setEddington(&eddington_nfw, 1000, Length/Nx, length_max_halo, 0, ratio_mass[0], num_k, boolk,
            int(Nx/2),int(Nx/2), int(Nx/2)); // The actual max radius is between Length and Length/2
        // Print the full initial snapshot, it prints in a snapshot file and not
        // in the backup file
        ofstream psi_snapshot;
        D3.outputfullPsi(psi_snapshot,false,1);
        // if numstep_relax==0, do not do evolution without stars
        // Also if number of halos is greater than one, do not relax the halo
        if(numstep_relax>0 ){
          // Sets the number of steps to relax the halo
          D3.set_numsteps(numstep_relax);
          D3.put_numstar_eff(0);
          // Run until relaxing time
          D3.solveConvDif();
        }
        D3.set_numsteps(numsteps);
        D3.put_numstar_eff(num_stars);
        // Then generate stars
        if(world_rank==0){
          double Length_max_stars= 10*Length;
          Eddington eddington_stars = Eddington(false, beta_ani);
          Plummer *profile_plummer = new Plummer(r_plummer, 
              M_plummer, Length_max_stars, true);
          NFW *profile_nfw_stars = new NFW(rs, rhos, Length_max_stars, true);
          eddington_stars.set_profile_den(profile_plummer);
          eddington_stars.set_profile_pot(profile_nfw_stars);
          // Stars gravity is not taken into account, so the potential 
          // should be just nfw
          // eddington_stars.set_profile_pot(profile_plummer);
          // Stars could be more concentrated than the resolution of the grid spacing
          double len_min_stars = 0.1*Length/Nx;
          eddington_stars.generate_fE_arr(1000, len_min_stars, Length_max_stars,outputname+"stars_");
          vector<double> xmax;
          // Call this function to set the maximum density location in D3
          double max_density = D3.find_maximum(0);
          vector<double> v_cm(3,0);
          for (int i=0; i<3; i++){
              xmax.push_back(D3.get_maxx(0,i)*Length/Nx);
              v_cm.push_back(D3.v_center_mass(i,0));
          }
            // Put just the center of the halos and vcm =vcm_halo if nhalos>1
          // Maximum length in profiles and here does not need to be the same
          cout<<"num_stars "<<num_stars<<endl;
          D3.generate_stars(&eddington_stars,
            len_min_stars, Length/3, xmax, v_cm, 0, num_stars);
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup();
        }
        D3.get_star_backup();
      }
      // After, if start_from_backup is false, it will not save the relaxing of halo part
    }
    else{
      run_ok=false; 
      if (world_rank==0){
        cout<<"You need 7 arguments to pass to the code: rs, rhos, r_plummer, M_plummer, beta_anisotropy, num_k, numstep_relax" << endl;
      }
    }
  }

  
  // Set 2 initial halos
  else if (initial_cond == "2halos_stars_NFW" ) {
    outputname= directory_name+"halo_stars_NFW"+outputname;
    if (params_initial_cond.size()> 10) {
      vector<double> rs;
      vector<double> rhos;
      vector<double> r_plummer;
      vector<double> M_plummer;
      int nhalos = 2;
      for (int halo_num =0; halo_num<nhalos;halo_num++){
        // NFW parameters
        rs.push_back(stod(params_initial_cond[4*halo_num]));
        rhos.push_back(stod(params_initial_cond[1+4*halo_num]));
        // Plummer parameters
        r_plummer.push_back(stod(params_initial_cond[2+4*halo_num]));
        M_plummer.push_back(stod(params_initial_cond[3+4*halo_num]));
      }
      int num_k = stoi(params_initial_cond[8]); // Number of maximum k points in nested loop
      double vcm_halo = stod(params_initial_cond[9]);
      double dist_between_halos = stod(params_initial_cond[10]); 
      bool boolk = false; // simplify k sum always false
      for(int i=0; i<nhalos;i++){
        outputname = outputname+"rs_"+to_string(rs[i])+"_rhos_"+to_string(rhos[i])+
          "_r_plummer_"+to_string(r_plummer[i])+"_M_plummer_"+to_string(M_plummer[i])
          +"_";
      }
      outputname = outputname+"vcm_"+to_string(vcm_halo)+"_dist_halos_"+to_string(dist_between_halos)
        +"_num_stars_"+to_string(num_stars)+"_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        multi_array<double, 2> stars_arr(extents[num_stars][8]);
        D3.put_initial_stars(stars_arr);
        double length_max_halo = 4*Length;
        // Initialize vectors to zero with specified size
        vector<int> x_center(nhalos, 0), y_center(2, 0), z_center(nhalos, 0);
        vector<double> vcm_halox(nhalos, 0.0), vcm_haloy(nhalos, 0.0), vcm_haloz(nhalos, 0.0);
        for(int i=0;i<nhalos;i++){
          NFW *profile_nfw = new NFW(rs[i], rhos[i], length_max_halo, true);// The actual max radius is between Length and Length/2
          // For stars, there is no meaning to put Length as rmax
          Eddington eddington_nfw = Eddington(true);
          eddington_nfw.set_profile_den(profile_nfw);
          eddington_nfw.set_profile_pot(profile_nfw);
          // Random velocity angle in the xy plane, leave the z component to zero
          double rand_phi = fRand(0,2*M_PI);
          x_center[i]=int(Nx/2. + dist_between_halos*Nx/Length*(-1/2.+i)); 
          y_center[i]=int(Nx/2); 
          z_center[i]=int(Nx/2);
          vcm_halox[i]=vcm_halo*cos(rand_phi);
          vcm_haloy[i]=vcm_halo*sin(rand_phi);
          vcm_haloz[i]=0;
          cout<<"vcm_halo "<<vcm_halo<<" "<<vcm_halox[i]<<" "<<vcm_haloy[i]<<" "<<vcm_haloz[i]<<endl;
          // Random direction of the vcm_halo on the xy plane
          vector<double> vcm_halo_vect = {vcm_halox[i], vcm_haloy[i], vcm_haloz[i]};
          // setEddington computes fE_func, so you should run it before
          D3.setEddington(&eddington_nfw, 1000, Length/Nx, length_max_halo, 0, ratio_mass[0], num_k, boolk,
              x_center[i],y_center[i],z_center[i], vcm_halo_vect ); // The actual max radius is between Length and Length/2
        }
        // Print the full initial snapshot, it prints in a snapshot file and not
        // in the backup file
        ofstream psi_snapshot;
        D3.outputfullPsi(psi_snapshot,false,1);
        D3.put_numstar_eff(num_stars);
        // Then generate stars
        if(world_rank==0){
          double Length_max_stars= 10*Length;
          for (int id_halo =0; id_halo< nhalos;id_halo++){
            Eddington eddington_stars = Eddington(false);
            Plummer *profile_plummer = new Plummer(r_plummer[id_halo], 
                M_plummer[id_halo], Length_max_stars, true);
            NFW *profile_nfw_stars = new NFW(rs[id_halo], rhos[id_halo], Length_max_stars, true);
            eddington_stars.set_profile_den(profile_plummer);
            eddington_stars.set_profile_pot(profile_nfw_stars);
            eddington_stars.generate_fE_arr(1000, Length/Nx, Length_max_stars,outputname+"stars_");
            vector<double> xmax;
            // Call this function to set the maximum density location in D3
            double max_density = D3.find_maximum(0);
            vector<double> v_cm(3,0);
            xmax.push_back(x_center[id_halo]*Length/Nx);
            xmax.push_back(y_center[id_halo]*Length/Nx);
            xmax.push_back(z_center[id_halo]*Length/Nx);
            v_cm[0]=vcm_halox[id_halo]; 
            v_cm[1]=vcm_haloy[id_halo]; 
            v_cm[2]=vcm_haloz[id_halo];
            D3.generate_stars(&eddington_stars,
              Length/Nx, Length/3, xmax, v_cm, int(id_halo*num_stars/2), int((id_halo+1)*num_stars/2));
          }
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup();
        }
        D3.get_star_backup();
      }
    }
    else{
      run_ok=false; 
      if (world_rank==0){
        cout<<"You need 3+ 4*2 arguments to pass to the code: [rs, rhos, r_plummer, M_plummer], num_k, vcm, dist_halos" << endl;
      }
    }
  }


  // Set an initial Burkert halo, let it relax, then put stars
  else if (initial_cond == "halo_stars_delayed_Burkert" ) {
    if (params_initial_cond.size() > 5){
      outputname= directory_name+"halo_stars_Burkert"+outputname;
      // Burkert parameters
      double r0 = stod(params_initial_cond[0]);
      double rho0= stod(params_initial_cond[1]);
      // Plummer parameters
      double r_plummer = stod(params_initial_cond[2]);
      double M_plummer= stod(params_initial_cond[3]);
      int num_k = stoi(params_initial_cond[4]); // Number of maximum k points in nested loop
      int numstep_relax = stoi(params_initial_cond[5]); // Number of steps to relax the halo
      bool boolk = false; // simplify k sum always false
      outputname = outputname+"r0_"+to_string(r0)+"_rho0_"+to_string(rho0)+
          "_r_plummer_"+to_string(r_plummer)+"_M_plummer_"+to_string(M_plummer)+
          "_num_stars_"+to_string(num_stars)+"_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        multi_array<double, 2> stars_arr(extents[num_stars][8]);
        D3.put_initial_stars(stars_arr);
        double length_max_halo = 4*Length;
        Burkert *profile_burk = new Burkert(r0, rho0, length_max_halo, true);// The actual max radius is between Length and Length/2
        // For stars, there is no meaning to put Length as rmax
        Eddington eddington_burk = Eddington(true);
        eddington_burk.set_profile_den(profile_burk);
        eddington_burk.set_profile_pot(profile_burk);
        // setEddington computes fE_func, so you should run it before
        D3.setEddington(&eddington_burk, 1000, Length/Nx, length_max_halo, 0, ratio_mass[0], num_k, boolk,
            int(Nx/2),int(Nx/2),int(Nx/2) ); // The actual max radius is between Length and Length/2
        
        // Print the full initial snapshot, it prints in a snapshot file and not
        // in the backup file
        ofstream psi_snapshot;
        D3.outputfullPsi(psi_snapshot,false,1);
        // if numstep_relax==0, do not do evolution without stars
        if(numstep_relax>0){
          // Sets the number of steps to relax the halo
          D3.set_numsteps(numstep_relax);
          D3.put_numstar_eff(0);
          // Run until relaxing time
          D3.solveConvDif();
        }
        D3.set_numsteps(numsteps);
        D3.put_numstar_eff(num_stars);
        // Then generate stars
        if(world_rank==0){
          double Length_max_stars= 10*Length;
          Eddington eddington_stars = Eddington(false);
          Plummer *profile_plummer = new Plummer(r_plummer, M_plummer, Length_max_stars, true);
          // If you define a different max length, you should define a new Profile 
          // with that max length
          Burkert *profile_burk_stars = new Burkert(r0, rho0, Length_max_stars, true);
          eddington_stars.set_profile_den(profile_plummer);
          eddington_stars.set_profile_pot(profile_burk_stars);
          // Stars gravity is not taken into account, so the potential 
          // should be just burkert
          eddington_stars.generate_fE_arr(1000, Length/Nx, Length_max_stars,outputname+"stars_");
          vector<double> xmax;
          // Call this function to set the maximum density location in D3,
          // needed if numstep_relax==0
          double max_density = D3.find_maximum(0);
          vector<double> v_cm;
          for (int i=0; i<3; i++){
            xmax.push_back(D3.get_maxx(0,i)*Length/Nx);
            v_cm.push_back(D3.v_center_mass(i,0));
            // v_cm.push_back(0);
          }
          D3.generate_stars(&eddington_stars,
            Length/Nx, Length/3, xmax, v_cm,0,num_stars);
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup();
        }
        D3.get_star_backup();
      }
      // After, if start_from_backup is false, it will not save the relaxing of halo part
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 6 arguments to pass to the code: r0, rho0, r_plummer, M_plummer, num_k, numstep_relax" << endl;
    }
  }


  // Set an initial halo, let it relax, then put stars in a disk
  else if (initial_cond == "halo_stars_disk" ) {
    if (params_initial_cond.size() > 5){
      outputname= directory_name+"halo_stars_disk"+outputname;
      // NFW parameters
      double rs = stod(params_initial_cond[0]);
      double rhos= stod(params_initial_cond[1]);
      // Plummer parameters
      double r_plummer = stod(params_initial_cond[2]);
      double M_plummer= stod(params_initial_cond[3]);
      int num_k = stoi(params_initial_cond[4]); // Number of maximum k points in nested loop
      int numstep_relax = stoi(params_initial_cond[5]); // Number of steps to relax the halo
      bool boolk = false; // simplify k sum always false
      outputname = outputname+"rs_"+to_string(rs)+"_rhos_"+to_string(rhos)+
          "_r_plummer_"+to_string(r_plummer)+"_M_plummer_"+to_string(M_plummer)+
          "_num_stars_"+to_string(num_stars)+"_";
      
      D3.set_output_name(outputname);
      D3.set_ratio_masses(ratio_mass);
      
      if(start_from_backup=="true"){
        D3.initial_cond_from_backup();
        D3.get_star_backup();
      }
      else{
        // Sets the number of steps to relax the halo
        D3.set_numsteps(numstep_relax);
        NFW *profile_nfw = new NFW(rs, rhos, Length, true);// The actual max radius is between Length and Length/2
        // For stars, there is no meaning to put Length as rmax
        Eddington eddington_nfw = Eddington(true);
        eddington_nfw.set_profile_den(profile_nfw);
        eddington_nfw.set_profile_pot(profile_nfw);
        // setEddington computes fE_func, so you should run it before
        D3.setEddington(&eddington_nfw, 500, Length/Nx, Length, 0, ratio_mass[0], num_k, boolk,
            int(Nx/2),int(Nx/2),int(Nx/2) ); // The actual max radius is between Length and Length/2
        D3.set_output_name(outputname);
        D3.set_ratio_masses(ratio_mass);
        D3.put_numstar_eff(0);
        // Run until relaxing time
        D3.solveConvDif();
        D3.set_numsteps(numsteps);
        D3.put_numstar_eff(num_stars);
        // Then generate stars
        if(world_rank==0){
          Eddington eddington_stars = Eddington(false);
          Plummer *profile_plummer = new Plummer(r_plummer, M_plummer, Length/3, true);
          // If you define a different max length, you should define a new Profile 
          // with that max length
          NFW *profile_nfw_stars = new NFW(rs, rhos, Length/3, true);// The actual max radius is between Length and Length/2
          eddington_stars.set_profile_den(profile_plummer);
          eddington_stars.set_profile_pot(profile_nfw_stars);
          // Stars gravity is not taken into account, so the potential 
          // should be just nfw
          // eddington_stars.set_profile_pot(profile_plummer);
          eddington_stars.generate_fE_arr(500, Length/Nx, Length/3);
          multi_array<double,2> stars_arr = D3.generate_stars_disk(&eddington_stars, Length/Nx, Length/3);
          D3.put_initial_stars(stars_arr);
          // World rank 0 creates the star backup, then all the other ranks will take from this backup
          D3.out_star_backup(); 
        }
        D3.get_star_backup();
      }
      // After, if start_from_backup is false, it will not save the relaxing of halo part
    }
    else{
      run_ok=false; 
      if (world_rank==0)
        cout<<"You need 6 arguments to pass to the code: rs, rhos, r_plummer, M_plummer, num_k, numstep_relax" << endl;
    }
  }

  else{
    run_ok=false; 
    if (world_rank==0){
      cout<< "String in 8th position does not match any possible initial conditions; possible initial conditions are:" << endl;
      cout<< "NFW, NFW_ext_Eddington, NFW_solitons, halo_stars_delayed_NFW, halo_stars_disk"<<endl;
      cout<< "uniform_NFW_test, halo_stars_delayed_Burkert " << endl;
    }
  }

  if(run_ok==true){
      cout<<"File name of the run: "<< outputname <<endl;
      D3.solveConvDif();
  }

  if(mpirun_flag==true){
    MPI_Finalize();
  }
  return 0;
}