#ifndef ULDM_MPI_2FIELDS_H
#define ULDM_MPI_2FIELDS_H

#include<iostream>
#include<stdio.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <unistd.h>
#include <limits>
#include <random>
#include <omp.h>
#include "boost/multi_array.hpp"
#include <fftw3-mpi.h>
#include <mpi.h>


using namespace std;
using namespace boost;

// extern tells the compiler that the function is definedsomewhere else (in this case, utilities.cpp),
// to avoid redefinition errors; couple this with ifndef statement
//random double betwenn fMin and fMax
extern double fRand(double fMin, double fMax); 
// shift function for fourier transform conventions, this one has minus signs
extern double shift(float i, float N);
//For Levkov waves initial conditions, Np is the number of particles and k2 the squared momentum
extern double P_spec(double k2, double Np);
// cyclic boundary conditions
extern int cyc(int ar, int le);
// 3 point order derivative, mostly for computing kinetic energy
extern double derivative_3point(double f1_plus, double f1_minus, double f2_plus, 
                                double f2_minus, double f3_plus, double f3_minus);
  
// Soliton profile in grid units
extern double psi_soliton(double r_c, double r);

/////////////////////  stuff for the ghosts ///////////////////////////////
// New function to send
extern void sendg( multi_array<double,4>  &grid,int ii, int world_rank, int world_size, int dir, int nghost);
// New function to receive
extern void receiveg(multi_array<double,4> &grid,int ii,  int world_rank, int world_size, int dir, int nghost);
// MPI stuff to sort out the ghosts
extern void transferghosts( multi_array<double,4> &gr,int ii, int world_rank, int world_size, int nghost);// Class for the Fourier Transform

// File printing functions; template class should be defined in the header
template<class T1>
inline int dimension(const multi_array<T1,3> & arr){ return arr.shape()[arr.num_dimensions()-1]; }
template<class T, class U>
void print2(multi_array<T,2> & v1,  U & filename){ //It prints a 2D array in the output file filename
    int Nx=v1.shape()[0];
    int Ny=v1.shape()[1];
        filename<<"{";
            for(int i = 0; i < Nx; i++){
            filename<<"{";
                for(int j = 0; j < Ny; j++){
                      filename<<v1[i][j];
                      if(j!=(Ny-1)){filename<<",";}
            }
        filename<<"}";
        if(i!=(Nx-1)){filename<<",";}
      }
    filename<< " } ";
}

template<class T, class U>
void print3(multi_array<T,3> & v1,  U & filename){ //It prints a 3D array in the output file filename
  int Nx=v1.shape()[0];
  int Ny=v1.shape()[1];
  int Nz=v1.shape()[2];
  filename<<"{";
  for(int i = 0;i < Nx; i++){
    filename<<"{";
    for(int j = 0;j < Ny; j++){
      filename<<"{";
      for(int k = 0;k < Nz; k++){
        filename<<v1[i][j][k];
        if(k!=(Nz-1)){filename<<",";}
      }
      filename<<"}";
      if(j!=(Ny-1)){filename<<",";}
    }
    filename<<"}";
      if(i!=(Nx-1)){filename<<",";}
  }
  filename<< " } ";
}

template<class T, class U>
// Class to print 3D arrays which are easy to read (load) with c++; it is just a string
// of numbers, arranged in a 3*Nx array such that v1[i][j][k] goes to the i + Nx*j + Nx*Nx*k position
// The numbers are separated by a space, " ", except for the last one
void print3_cpp(multi_array<T,3> & v1,  U & filename){
    int Nx=v1.shape()[0];
    int Ny=v1.shape()[1];
    int Nz=v1.shape()[2];
    for(int k = 0;k < Nz; k++){
      for(int j = 0;j < Ny; j++){
        for(int i = 0;i < Nx; i++){
          filename<<v1[i][j][k];
          if(k!=(Nx-1) || j!=(Nx-1) || i!=(Nx-1)) //If it is not the last one
            filename<< " ";
        }
      }
    }
}
template<class T, class U> //Outputs the full psi with the n fields
void print4_cpp(multi_array<T,4> & v1,  U & filename, int nghost){
  int Nl = v1.shape()[0]; //Dimension along the first argument (2*nfields)
  int Nx=v1.shape()[1];
  int Ny=v1.shape()[2];
  int Nz=v1.shape()[3]; //Nz contains the 2*nghost cells
  for(int l=0;l<Nl;l++){
    for(int k = nghost;k < Nz-nghost; k++){
      for(int j = 0;j < Nx; j++){
        for(int i = 0;i < Nx; i++){
          filename<<v1[l][i][j][k];
          if(l!=(Nl-1) || k!=(Nz-nghost-1) || j!=(Ny-1) || i!=(Nx-1)) //If it is not the last one
            filename<< " ";
        }
      }
    }
  }
}

// No ghosts in the z-direction on these grids
class Fourier{
  size_t Nx; // Number of grid points in linear dimension
  size_t Nz; // Number of points in the short grid dimension
  // variables for fftw_
  fftw_plan plan;
  fftw_plan planback;
  fftw_complex *rin;
  ptrdiff_t alloc_local, local_n0, local_0_start;
  int world_rank;
  int world_size;
  bool mpi_bool;

  public:
    ////////////////// Everything is defined in fourier.cpp ///////////////////////////////////
    Fourier(); //default constructor
    Fourier(size_t PS, size_t PSS, int WR, int WS, bool mpi_flag); //constructor
    ~Fourier(); //destructor
    //calculate the FT, more precisely: \tilde{A}_k = \Sum_{n1,n2,n3=0}^{Nx-1} A_n e^(-2\pi i k\cdot n/Nx)
    void calculateFT();
    // Calculate the unnormalized inverse FT, remember to divide the output by Nx^3
    void calculateIFT();
    // A sample grid for testing purposes
    void inputsamplegrid();
    // Insert on initial conditions the Levkov waves
    void inputSpectrum(double Length, double Npart, double r);
    //functions needed for the evolution
    //input psi on the array for fftw, note the shift due to the ghost cells
    // whichPsi should be 0 or 1 depending on which field
    void inputpsi(multi_array<double,4> &psi, int nghost, int whichPsi);
    //function for putting -k^2 factor; more precisely, psi->exp(-i c_alpha dt k^2/2 ) psi
    void kfactorpsi(double tstep, double Length, double calpha, int whichPsi, multi_array<double, 1> ratio_mass);
    // Put on psi the result of Fourier transforms
    void transferpsi(multi_array<double,4> &psi, double factor, int nghost, int whichPsi);  
    // input |psi|^2 to the memory that will be FTed
    // virtual, for inheritance (in case one wants to use an external potential
    virtual void inputPhi(multi_array<double,4> &psi_in, int nghost, int nfields);
    //function for putting -1/k^2 factor on the output of the FT
    // needed for solving the Poisson equation to get Phi
    void kfactorPhi(double Length);  
    // take the result of calculating Phi from the Poisson equation and put it onto a grid
    // Note the normalisation factor here from the inverse FT
    void transferPhi(multi_array<double,3> &Phi, double factor);
    //function for the total kinetic energy using the FT
    // note this does not sum up the values on different nodes
    double e_kin_FT(multi_array<double,4> &psi, double Length,int nghost, int whichPsi);
};


//////////////////////////////////////////////////////////////////////////////////////////
// Class for the domain to be solved in: builds the grid with the correct parameters    //
//////////////////////////////////////////////////////////////////////////////////////////
class domain3{
  protected:
    multi_array<double,4> psi;  // contains the n fields
    int nghost;                 // number of ghost layers above and below in the z direction
    bool mpi_bool; // If true, the domain is distributed over MPI
    int nfields;                 // number of fields
    multi_array<double,3> Phi;
    multi_array<double,1> ratio_mass; // Ratio between the masses of the ULDM wrt field 0
    size_t PointsS; // number of points in the long direction
    size_t PointsSS; // number of points in the short direction, not including ghost cells
    double Length; // Physical length of the domain
    int numsteps; // Number of steps, adaptive timestep case
    double dt; // Initial time step
    double deltaX; // Physical grid spacing
    int numoutputs; // Number of outputs for the sliced (if Grid3D==True) or full 3D grid (if Grid3D==False) density profile (for animation)
    int numoutputs_profile; //number of outputs for the radial profiles

    string outputname; // Name of the output file
    Fourier fgrid;               //for the FT

    multi_array<double,1> ca;     //vectors that store the numerical values of the coefficients to step forward
    multi_array<double,1> da;

    ofstream runinfo; // Output file for the run information (grid points, length etc.)
    ofstream profilefile; // Output file for radial profiles of density, energies etc.
    ofstream profile_sliced; // Output file for 2D projected profile
    ofstream phase_slice; // Slice for the phase
    ofstream timesfile_grid;  //output file for grid
    ofstream timesfile_profile;  //output file for useful information (total energies etc.)
    ofstream info_initial_cond;  //output file for initial condition details
    int snapshotcount=0;          //variable to number the snapshots
    bool Grid3D = false; // If true, it outputs the full density on the 3D grid; if false (recommended), it outputs the 2D projection of the density profile
    bool phaseGrid = false; // If true, it outputs the phase slice passing on the center
    bool start_from_backup = false; // If true, starts from the backup files
    int pointsmax =0;
    multi_array<int, 2> maxx;       //location x,y,z of the max density in the grid (for a certain field)
    int maxNode=0;    // node that the maximum value is on
    multi_array<double, 1>  maxdensity; // Max density on the grid

    // Variables useful for backup
    double tcurrent; // Current time of simulation
    double E_tot_initial; // Stores the initial total energy to implement the adaptive time step

    int world_rank;
    int world_size;

    public:
      /////////////////// The following defined in domain3_main.cpp /////////////////////
      domain3(size_t PS,size_t PSS, double L, int nfields, int Numsteps, double DT, int Nout, int Nout_profile, 
          string Outputname, int pointsm, int WR, int WS, int Nghost, bool mpi_flag);
      domain3 (); //default constructor
      ~domain3(); //destructor
      long double psisqmean(int whichPsi);
      double total_mass(int whichPsi);
      // note k is counted including ghosts
      double energy_kin(const int & i, const int & j, const int & k, int whichPsi);
      // it computes the potential energy density at grid point (i,j,k)
      // virtual for inheritance (external potential)
      // note that Psi fields have ghost points Phi doesn't, k includes ghosts
      virtual double energy_pot(const int & i, const int & j, const int & k, int whichPsi);
      double e_kin_full1(int whichPsi);
      long double full_energy_kin(int whichPsi);
      long double full_energy_pot(int whichPsi);
      double find_maximum(int whichPsi);
      // update the ghosts on the psi grids, need to run this before the timeshots
      void sortGhosts();
      // psi -> exp(-i tstep d_alpha Phi) psi; does a step forward or with the opposite sign by changing the sign of tstep
      virtual void expiPhi(double tstep, double da, int whichPsi);
      void set_grid(bool grid_bool);//If false, domain3 outputs only the 2D sliced density profile
      void set_grid_phase(bool bool_phase);//If false, domain3 does not output the phase slice
      void set_backup_flag(bool bool_backup);//If false, no backup
      void set_ratio_masses(multi_array<double,1> ratio_mass);//Sets the ratio between the masses of the ULDM wrt field 0
      virtual void makestep(double stepCurrent, double tstep);
      void solveConvDif();
      // Notice that you should call the backup from a run which uses the SAME number of cores in mpi processes
      void initial_cond_from_backup(); //Sets the initial conditions from the backup files, if backup_flag is true

      ////////////// The following are all defined in output_domain3.cpp /////////////////////////
      void openfiles();
      void openfiles_backup();
      // Outputs the full density
      void outputfulldensity(ofstream& fileout,int whichPsi);
      // Outputs the full 3D phi, for backup purposes
      void outputfullPhi(ofstream& fileout);
      // Outputs the full 4D psi, every field, for backup purposes
      void outputfullPsi(ofstream& fileout);
      // Virtual because for the NFW case (for example) you want to compute radial functions starting from the center of the box and not the maximum
      virtual multi_array<double,2> profile_density(int whichPsi);
      void snapshot(double stepCurrent);
      void snapshot_profile(double stepCurrent);
      // Outputs the projected 2D column density, every field
      void outputSlicedDensity(ofstream& fileout);
      // Outputs a 2D slice of the phase
      void outputPhaseSlice(ofstream& fileout);
      // Considering I am implementing the possbility to use backups, this function closes files
      // WITHOUT putting a final }; if one wants to read these arrays with a program (Mathematica etc.),
      // one should load those arrays and put a final }
      void closefiles();
      // It stores run output info
      virtual void exportValues();
      
      //////////////// Initial conditions functions, defined in domain3_initial_cond.cpp ////////////////
      // Initial condition with waves, test purposes
      void initial_waves();
      // Sets one soliton in the center of the box
      void setInitialSoliton_1(double r_c, int whichPsi);
      // sets many solitons as initial condition, with random core radius whose centers are confined in a box of length length_lim
      void setManySolitons_random_radius(int num_Sol, double min_radius, double max_radius, double length_lim);
      // sets many solitons as initial condition, with same core radius whose centers are confined in a box of length length_lim
      void setManySolitons_same_radius(int num_Sol, double r_c, double length_lim);
      // Deterministic initial condition
      void setManySolitons_deterministic(double r_c, int num_sol);
      // Levkov like initial conditions
      void set_waves_Levkov(multi_array<double, 1> Npart);

       // functions below not adapted for MPI yet
 /*
        double phi_max(){ // Finds the maximum of the potential Phi
          double phi_maximum = 0;
          for(int i=0;i<PointsS;i++)
                for(int j=0; j<PointsS;j++)
                    for(int k=0; k<PointsSS;k++){
                      double phi_temp = Phi[i][j][k];
                      if (phi_temp > phi_maximum)
                      {phi_maximum =phi_temp; }
                    }
          return phi_maximum;
        }
        // Computes the Phi average, for testing purposes (it should be zero)
        double phi_average(){
          double average = 0;
          for(int i=0;i<PointsS;i++)
                for(int j=0; j<PointsS;j++)
                    for(int k=0; k<PointsS;k++){
                      average = average + Phi[i][j][k];
                    }
          return average;
        }
*/

};

#endif


