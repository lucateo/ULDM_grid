#include "uldm_sim.h"

//To make it run, an example: ./main_sim 128 100 1000 1 1Sol false 4

// Remember to change string_outputname according to where you want to store output files, or
// create the directory BEFORE running the code
int main(int argc, char** argv){
    int beginning=time(NULL);
    // srand(time(NULL));
    //srand(42);

    int Nx = atof(argv[1]);                       //number of gridpoints
    double Length= atof(argv[2]);                 //box Length in units of m
    int numsteps= atof(argv[3]);                   //Number of steps
    double dt= atof(argv[4]);                     //timeSpacing,  tf/Nt

    int outputnumb=10;//atof(argv[6]);;            //number of outputs
    int outputnumb_profile=10;//atof(argv[6]);;            //number of outputs

    double dx = Length/Nx;                            //latticeSpacing, dx=L/Nx, L in units of m
    int Pointsmax = Nx/2; //Number of maximum points which are plotted in profile function

    // This is to tell which initial condition you want to run
    string initial_cond = argv[5];

    string start_from_backup = argv[6]; // true or false, depending on whether you want to start from a backup or not
    bool backup_bool = false;
    if (start_from_backup == "true")
      backup_bool = true;

    if (initial_cond == "1Sol" ) {// Initial condition with one soliton on the center
      if (argc > 7){
      double r_c = atof(argv[7]); // Core radius for initial conditions
      string outputname = "out/out_test/out_Nsol1_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rc_" + to_string(r_c) + "_";
      domain3 D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax);

    D3.set_grid(false);// It will output 2D projected density grid and not 3D
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.setInitialSoliton_1(r_c); // Set the initial condition
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
    cout<< "Average of Phi "<< D3.phi_average() <<endl;
      }
      else
        cout<<"You need 7 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, rc" << endl;
    }
    else if (initial_cond == "levkov" ) {// Levkov initial conditions
      if (argc > 8){
      double Npart = atof(argv[7]); // Number of particles
      int vw = atof(argv[8]); // Velocity of particles
      string outputname = "out/out_test/out_Levkov_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_Npart_" + to_string(Npart) + "_vw_" + to_string(vw)+ "_";
      domain3 D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax);
    D3.set_grid(false);
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.set_waves_Levkov(Npart, vw);
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
      }
      else
        cout<<"You need 8 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, Npart, vw" << endl;
    }

    else if (initial_cond == "Schive_random_vel" ) { // Schive-like (multi soliton with same core radius) with random velocity, just a toy
      if (argc > 8){
      double r_c = atof(argv[7]); // Core radius for initial conditions
      int num_Sol = atof(argv[8]); // number of solitons in simulations with many solitons initial conditions
      string outputname = "out_schive/out_Schive_random_vel_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rc_" + to_string(r_c) + "_Nsol_" + to_string(num_Sol)+ "_";
      domain3 D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax);
    D3.set_grid(false);
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.setManySolitons_same_radius_random_vel(num_Sol, r_c);
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
      }
      else
        cout<<"You need 8 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, rc, num_Sol" << endl;
    }

    else if (initial_cond == "Schive" ) {// Schive-like (multi soliton with same core radius)
      if (argc > 9){
      double r_c = atof(argv[7]); // Core radius for initial conditions
      int num_Sol = atof(argv[8]); // number of solitons in simulations with many solitons initial conditions
      double length_lim = atof(argv[9]); // Limit the span of the center of solitons inside a smaller box
      string outputname = "out_schive/out_Schive_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rc_" + to_string(r_c) + "_Nsol_" + to_string(num_Sol)+ "_length_lim_" + to_string(length_lim)+ "_";
      domain3 D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax);
    D3.set_grid(false);
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.setManySolitons_same_radius(num_Sol, r_c, length_lim);
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
      }
      else
        cout<<"You need 9 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, rc, num_Sol, length_lim" << endl;
    }

    else if (initial_cond == "Mocz" ) {// Mocz-like (multi soliton with random core radius)
      if (argc > 10){
      double min_radius = atof(argv[7]); // Minimal core radius for initial conditions
      double max_radius = atof(argv[8]); // Maximal core radius for initial conditions
      int num_Sol = atof(argv[9]); // number of solitons in simulation
      double length_lim = atof(argv[10]); // Limit the span of the center of solitons inside a smaller box
      string outputname = "out_mocz/out_Mocz_nopsisqmean_Nx_" + to_string(Nx) + "_L_"
        + to_string(dt) + "_radius_range_" + to_string(min_radius) +"_" + to_string(max_radius) + "_Nsol_" + to_string(num_Sol)
        + "_length_lim_" + to_string(length_lim)+ "_";
      domain3 D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax);
    D3.set_grid(false);
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.setManySolitons_random_radius(num_Sol, min_radius, max_radius, length_lim);
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
      }
      else
        cout<<"You need 10 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, min_rad, max_rad, num_Sol, length_lim" << endl;
    }

    else if (initial_cond == "deterministic" ) {//The many soliton, with fixed position, for testing purposes
      if (argc > 8){
      double r_c = atof(argv[7]); // Core radius for initial conditions
      double num_Sol = atof(argv[8]); // Number of solitons; maximum allowed is 30
      string outputname = "out_deterministic/outdeterministic_nopsisqmean_Nx_" + to_string(Nx) + "_L_" + to_string(Length)
        + "_rc_" + to_string(r_c) + "_Nsol_" + to_string(num_Sol) +  "_";
      domain3 D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax);
    D3.set_grid(false);
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.setManySolitons_deterministic(r_c, num_Sol);
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
    cout<< "Average of Phi "<< D3.phi_average() <<endl;
      }
      else
        cout<<"You need 8 arguments to pass to the code: Nx, Length, tf, dt, initial_cond, start_from_backup, rc, num_Sol" << endl;
    }

    else{
      cout<< "String in 5th position does not match any possible initial conditions; possible initial conditions are:" << endl;
      cout<< "Schive , Mocz , deterministic , levkov, 1Sol, Schive_random_vel" <<endl;
    }

    return 0;
}


