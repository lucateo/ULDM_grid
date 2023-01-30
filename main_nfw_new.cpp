// #include "uldm_sim.h"
#include "uldm_sim_nfw.h"
//To make it run, an example: ./main_sim_nfw 128 100 1000 1 Simple 10 0.1 20
int main(int argc, char** argv){

    int beginning=time(NULL);
    srand(time(NULL));
    // srand(42);

    int Nx = atof(argv[1]);                       //number of gridpoints
    double Length= atof(argv[2]);                 //box Length in units of m
    int numsteps= atof(argv[3]);                   //Number of steps
    double dt= atof(argv[4]);                     //Initial timeSpacing

    int outputnumb=10;//atof(argv[6]);;            //number of outputs
    int outputnumb_profile=10;//atof(argv[6]);;            //number of outputs

    double dx = Length/Nx;                            //latticeSpacing, dx=L/Nx, L in units of m
    int Pointsmax = Nx/2; //Number of maximum points which are plotted in profile function

    // Reference values for conversion to physical values
    double mass_particle = 1e-24; // in eV
    double v0_phys = 200; // in km/s
    double clight = 3e8; // in m/s
    double hbar_eV = 6.582119568038699e-16; // in eV*s
    double mplanck = 1.2e19; // in GeV
    double eV2Joule = 1.6021e-19;
    double msun = 1.989e30; // in Kg
    double pc2meter = 3.086e16;

    //To allow for more "settings", right now the supported ones are "NFW",
    // i.e. Levkov waves with NFW external potential, and other test ones.
    string initial_cond = argv[5];
    string start_from_backup = argv[6]; // true or false, depending on whether you want to start from a backup or not
    bool backup_bool = false;
    if (start_from_backup == "true")
      backup_bool = true;

    double rs_nfw = atof(argv[7]); //NFW radius in grid units

    if (initial_cond == "1Sol" ) { // Initial condition with one soliton on the center
      if (argc > 9){
      double rho0_tilde = atof(argv[8]); //Adimensional rho_0 for the NFW external potential
      double const_nfw = rho0_tilde/(4*M_PI); // The constant which enters domain_ext_nfw class
      double r_c = atof(argv[9]); // Core radius for initial conditions
      string outputname = "out_test_nfw/out_nwf_new_Nsol1_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length) +
         "_Rs_" + to_string(rs_nfw) + "_rho_tilde_" + to_string(rho0_tilde)+ "_rc_" + to_string(r_c) + "_";
      domain_ext_nfw D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, rs_nfw, const_nfw);
      D3.setInitialSoliton_1(r_c);

    D3.set_grid(false); // It will output 2D projected density grid and not 3D
    D3.set_grid_phase(true); // It will output 2D slice of phase grid
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
    cout<< "Average of Phi "<< D3.phi_average() <<endl;
      }
      else
        cout<<"You need 9 arguments to pass to the code: Nx, Length, Nsteps, dt, initial_cond, start_from_backup, rs_nfw, alpha, rc" << endl;
    }

    else if (initial_cond == "NFW" ) {
      if (argc > 9){
          double rho0_tilde = atof(argv[8]);//Adimensional rho_0 for the NFW external potential
          double Npart = atof(argv[9]); // Number of particles
          double const_nfw = rho0_tilde/(4*M_PI);// The constant which enters domain_ext_nfw class
          double M_uldm_phys = v0_phys*1e3/clight * Npart * pow(1e9*mplanck,2)/(mass_particle *4*M_PI) * eV2Joule/pow(clight,2) / msun; // in msun
          double grid_unit_space = hbar_eV/mass_particle * clight/(1e3*v0_phys/clight) / (1e3*pc2meter); // in kpc
          double grid_unit_time = hbar_eV/mass_particle /pow(1e3*v0_phys/clight,2) / (3600*24*365.25*1e9); // in Gyrs

          string outputname = "out_nfw/out_nfw2_new_nopsisqmean_averages_center_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_Rs_" + to_string(rs_nfw) + "_rh0tilde_" + to_string(rho0_tilde)+ "_Npart_"
            + to_string(Npart) +"_";

          domain_ext_nfw D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, rs_nfw, const_nfw);
          D3.set_grid(false); // It will output 2D projected density grid and not 3D
          D3.set_grid_phase(true); // It will output 2D slice of phase grid
          D3.set_backup_flag(backup_bool);
          if (start_from_backup == "true")
            D3.initial_cond_from_backup();
          else
            D3.set_waves_Levkov(Npart, 1);
          cout<<"M uldm "<<M_uldm_phys<<" Msun, unit space "<< grid_unit_space<<" kpc, unit time "<<grid_unit_time<<" Gyrs"<<endl;
          D3.solveConvDif();
      }
      else
        cout<<"You need 9 arguments to pass to the code: Nx, Length, Nsteps, dt, initial_cond, start_from_backup, rs_nfw, alpha, Npart" << endl;
    }
    else if (initial_cond == "Simple" ) {
      if (argc > 9){
          double alpha = atof(argv[8]); // Ratio between ULDM and other dark matter
          double Npart = atof(argv[9]); // Number of particles
          double const_nfw = Npart * (1 - alpha)/(16*M_PI *M_PI * alpha*pow(rs_nfw, 3)* (log(1+Length/(2*rs_nfw)) + rs_nfw/(rs_nfw + Length/2) -1  ) );
          double M_uldm_phys = v0_phys*1e3/clight * Npart * pow(1e9*mplanck,2)/(mass_particle *4*M_PI) * eV2Joule/pow(clight,2) / msun; // in msun
          double M_nfw = (1 - alpha)/alpha * M_uldm_phys;
          double grid_unit_space = hbar_eV/mass_particle * clight/(1e3*v0_phys/clight) / (1e3*pc2meter); // in kpc
          double grid_unit_time = hbar_eV/mass_particle /pow(1e3*v0_phys/clight,2) / (3600*24*365.25*1e9); // in Gyrs

          string outputname = "out_nfw/out_nfw_nopsisqmean_averages_center_Nx" + to_string(Nx) + "_L_" + to_string(Length)
            + "_Rs_" + to_string(rs_nfw) + "_alpha_" + to_string(alpha)+ "_Npart_"
            + to_string(Npart) +"_";

          domain_ext_nfw D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, rs_nfw, const_nfw);
          D3.set_grid(false);
          D3.set_grid_phase(true);
          D3.set_backup_flag(backup_bool);
          if (start_from_backup == "true")
            D3.initial_cond_from_backup();
          else
            D3.set_waves_Levkov(Npart, 1);
          cout<<"M uldm "<<M_uldm_phys<<" Msun, M_nfw "<<M_nfw<<" M_sun, unit space "<< grid_unit_space<<" kpc, unit time "<<grid_unit_time<<" Gyrs"<<endl;
          D3.solveConvDif();
      }
      else
        cout<<"You need 9 arguments to pass to the code: Nx, Length, Nsteps, dt, initial_cond, start_from_backup, rs_nfw, alpha, Npart" << endl;
    }

    else if (initial_cond == "Schive" ) {
      if (argc > 11){
      double alpha = atof(argv[8]); // Ratio between ULDM and other dark matter
      double r_c = atof(argv[9]); // Core radius for initial conditions
      int num_Sol = atof(argv[10]); // number of solitons in simulations with many solitons initial conditions
      double length_lim = atof(argv[11]); // Limit the span of the center of solitons inside a smaller box
      string outputname = "out_test_nfw/out_nfw_Schive_averages_center_nopsisqmean_Nx" + to_string(Nx) + "_L_" + to_string(Length)
        + "_Rs_" + to_string(rs_nfw) + "_alpha_"+ to_string(dt) + "_rc_" + to_string(r_c) + "_Nsol_"
        + to_string(num_Sol)+ "_length_lim_" + to_string(length_lim)+ "_";
      domain_ext_nfw D3(Nx,Length,numsteps,dt,outputnumb, outputnumb_profile, outputname, Pointsmax, 0, 0);
          if (start_from_backup == "true")
            D3.initial_cond_from_backup();
          else
            D3.setManySolitons_same_radius(num_Sol, r_c, length_lim);
      double mass_uldm = D3.total_mass();
      double const_nfw = mass_uldm * (1 - alpha)/(16*M_PI *M_PI * alpha*pow(rs_nfw, 3)* (log(1+Length/(2*rs_nfw)) + rs_nfw/(rs_nfw + Length/2) -1  ) );
      D3.set_nfw_params(rs_nfw, const_nfw);
    D3.set_grid(false);
    D3.set_grid_phase(true);
    D3.set_backup_flag(backup_bool);
    D3.solveConvDif();
      }
      else
        cout<<"You need 11 arguments to pass to the code: Nx, Length, Nsteps, dt, initial_cond, start_from_backup, rs_nfw, alpha, rc, num_Sol, length_lim" << endl;
    }

    else{
      cout<< "String in 5th position does not match any possible initial conditions; possible initial conditions are:" << endl;
      cout<< "Simple NFW 1Sol Schive" <<endl;
    }
    return 0;
}

