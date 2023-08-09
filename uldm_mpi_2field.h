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
extern double fRand(double fMin, double fMax); //random double betwenn fMin and fMax
// shift function for fourier transform conventions, this one has minus signs
extern double shift(float i, float N);

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
    int Nx=dimension(v1);
        filename<<"{";
            for(int i = 0;i < Nx; i++){
        filename<<"{";
        for(int j = 0;j < Nx; j++){
        filename<<"{";
            for(int k = 0;k < Nx; k++){
                      filename<<v1[i][j][k];
              if(k!=(Nx-1)){filename<<",";}
        }
        filename<<"}";
        if(j!=(Nx-1)){filename<<",";}
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
    int Nx=dimension(v1);
    for(int k = 0;k < Nx; k++){
      for(int j = 0;j < Nx; j++){
        for(int i = 0;i < Nx; i++){
          filename<<v1[i][j][k];
          if(k!=(Nx-1) || j!=(Nx-1) || i!=(Nx-1)) //If it is not the last one
            filename<< " ";
        }
    }
    }
}
template<class T, class U> //Outputs the full psi with the to fields
void print4_cpp(multi_array<T,4> & v1,  U & filename, int nghost){
  int Nl = v1.shape()[0]; //Dimension along the first argument
  int Nx=v1.shape()[1];
  int Nz=v1.shape()[3]; //Nz contains the 2*nghost cells
  for(int l=0;l<Nl;l++){
    for(int k = nghost;k < Nz-nghost; k++){
      for(int j = 0;j < Nx; j++){
        for(int i = 0;i < Nx; i++){
          filename<<v1[l][i][j][k];
          if(l!=(Nl-1) || k!=(Nx-1) || j!=(Nx-1) || i!=(Nx-1)) //If it is not the last one
            filename<< " ";
        }
      }
    }
  }
}
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
void transferghosts( multi_array<double,4> &gr,int ii, int world_rank, int world_size, int nghost);// Class for the Fourier Transform

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

  public:
    Fourier(size_t PS, size_t PSS, int WR, int WS);
    //calculate the FT, more precisely: \tilde{A}_k = \Sum_{n1,n2,n3=0}^{Nx-1} A_n e^(-2\pi i k\cdot n/Nx)
    void calculateFT();
    // Calculate the unnormalized inverse FT, remember to divide the output by Nx^3
    void calculateIFT();

  // A sample grid for testing purposes
  void inputsamplegrid();
  // Insert on initial conditions the Levkov waves
  void inputSpectrum(double Length, double Npart, double vw);
  //functions needed for the evolution
  //input psi on the array for fftw, note the shift due to the ghost cells
  // whichPsi should be 0 or 1 depending on which field
  void inputpsi(multi_array<double,4> &psi, int nghost, int whichPsi);
  //function for putting -k^2 factor; more precisely, psi->exp(-i c_alpha dt k^2/2 ) psi
  void kfactorpsi(double tstep, double Length, double calpha, int whichPsi, double r);
  // Put on psi the result of Fourier transforms
  void transferpsi(multi_array<double,4> &psi, double factor, int nghost, int whichPsi);  
  // input |psi|^2 to the memory that will be FTed
  // virtual, for inheritance (in case one wants to use an external potential
  virtual void inputPhi(multi_array<double,4> &psi_in, int nghost);
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
    multi_array<double,4> psi;  // contains the two fields
    int nghost;                 // number of ghost layers above and below in the z direction
    multi_array<double,3> Phi;
    double ratio_mass; // Ratio between the two masses of ULDM
    size_t PointsS;
    size_t PointsSS; // number of points in the short direction, not including ghost cells
    double Length;
    // double tf;
    int numsteps; // Number of steps, adaptive timestep case
    double dt;
    double deltaX;
    int numoutputs; // Number of outputs for the sliced (if Grid3D==True) or full 3D grid (if Grid3D==False) density profile (for animation)
    int numoutputs_profile; //number of outputs for the radial profiles

    string outputname;
    Fourier fgrid;               //for the FT

    multi_array<double,1> ca;     //vectors that store the numerical values of the coefficients to step forward
    multi_array<double,1> da;

    multi_array<double,1> jumps;  //output timeshots contained in 'jumps'
    multi_array<double,1> jumps_profile;  //output timeshots contained in 'jumps_profile'

    ofstream runinfo;
    ofstream profilefile; // Output file for radial profiles of density, energies etc.
    ofstream profile_sliced; // Output file for 2D projected profile
    ofstream phase_slice; // Slice for the phase
    ofstream profilefile1; // Output file for radial profiles of density, energies etc. of field 1
    ofstream profile_sliced1; // Output file for 2D projected profile of field 1
    ofstream phase_slice1; // Slice for the phase of field 1
    ofstream timesfile_grid;  //output file for grid
    ofstream timesfile_profile;  //output file for useful information (total energies etc.)
    ofstream info_initial_cond;  //output file for initial condition details
    int snapshotcount=0;          //variable to number the snapshots
    bool Grid3D = false; // If true, it outputs the full density on the 3D grid; if false (recommended), it outputs the 2D projection of the density profile
    bool phaseGrid = false; // If true, it outputs the phase slice passing on the center
    bool start_from_backup = false; // If true, starts from the backup files, not adapted for MPI yet
    int pointsmax =0;
    int maxx;       //location x,y,z of the max density in the grid
    int maxy;
    int maxz;
    int maxx1;       //location x,y,z of the max density in the grid of field 1
    int maxy1;
    int maxz1;
    int maxNode;    // node that the maximum value is on
    double maxdensity; // Max density on the grid

    // Variables useful for backup
    double tcurrent; // Current time of simulation
    double E_tot_initial; // Stores the initial total energy to implement the adaptive time step

    int world_rank;
    int world_size;

    public:
        domain3(size_t PS,size_t PSS, double L, double r_m, int Numsteps, double DT, int Nout, int Nout_profile, 
            string Outputname, int pointsm, int WR, int WS, int Nghost):
          nghost(Nghost),
          psi(extents[4][PS][PS][PSS+2*Nghost]), //real and imaginary parts are stored consequently, i.e. psi[0]=real part psi1 and psi[1]=imaginary part psi1, then the same for psi2
          Phi(extents[PS][PS][PSS]),
          ratio_mass(r_m),
          fgrid(PS,PSS,WR,WS), // class for Fourier trasform, defined above
          ca(extents[8]),
          da(extents[8]),
          PointsS(PS),
          PointsSS(PSS),
          Length(L),
          dt(DT),
          // tf(Tf),
          numsteps(Numsteps),
          pointsmax(pointsm),
          numoutputs(Nout),
          numoutputs_profile(Nout_profile),
          outputname(Outputname),
          jumps(extents[Nout+1]), // vector whose length corresponds to the outputs in time
          jumps_profile(extents[Nout_profile+1]), // vector whose length corresponds to the outputs in time for profile
          world_rank(WR),
          world_size(WS)
          {
            deltaX=Length/PointsS;
            // stepping numbers, as defined in axionyx documentation
            ca[0]=0.39225680523878;   ca[1]=0.51004341191846;   ca[2] =-0.47105338540976; ca[3]=0.06875316825251;    ca[4]=0.06875316825251;    ca[5]=-0.47105338540976;  ca[6]=0.51004341191846;   ca[7]=0.39225680523878;
            da[0]=0.784513610477560;  da[1]=0.235573213359359;  da[2]=-1.17767998417887;  da[3]=1.3151863206839023;  da[4]=-1.17767998417887;   da[5]=0.235573213359359;  da[6]=0.784513610477560;  da[7]=0;
          }; // constructor

        ~domain3() {};
        void openfiles();
        void openfiles_backup();
        // Considering I am implementing the possbility to use backups, this function closes files
        // WITHOUT putting a final }; if one wants to read these arrays with a program (Mathematica etc.),
        // one should load those arrays and put a final }
        void closefiles();
        // It stores run output info
        virtual void exportValues();
        void setoutputs(double t_ini=0){// Set the indices of the steps when output file function will be called
            for(int inde=0; inde<numoutputs+1; inde++){
                jumps[inde]=int(numsteps/numoutputs*inde);
            }
            for(int inde=0; inde<numoutputs_profile+1; inde++){
                jumps_profile[inde]=int(numsteps/numoutputs_profile*inde);
            }
        }
        // fileout should have a different name on each node
        void outputfulldensity(ofstream& fileout,int whichPsi);
        void outputfullPhi(ofstream& fileout);
        // Outputs the full 3D psi, both fields, for backup purposes
        void outputfullPsi(ofstream& fileout);

        void outputSlicedDensity(ofstream& fileout, int whichPsi);
        // Outputs a 2D slice of the phase
        void outputPhaseSlice(ofstream& fileout, int whichPsi);
    
    long double psisqmean(int whichPsi){// Computes the mean |psi|^2 of field i=0,1
      long double totV=0;
        #pragma omp parallel for collapse(3) reduction(+:totV)
        for(size_t i=0;i<PointsS;i++)
          for(size_t j=0;j<PointsS;j++)
            for(size_t k=0;k<PointsSS;k++)
              totV=totV+pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi+1][i][j][k],2);
        #pragma omp barrier

	    long double totVshared; // total summed up across all nodes
	    MPI_Allreduce(&totV, &totVshared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        totVshared=totVshared/(PointsS*PointsS*PointsS);
        return totVshared;
      }
      double total_mass(int whichPsi){ // Computes the total mass
        return psisqmean(whichPsi) * pow(Length, 3);
    }

	// cycs not needed in the z direction, but do no harm
        // note k is counted including ghosts
        inline double energy_kin(const int & i, const int & j, const int & k, int whichPsi){// it computes the kinetic energy density at grid point (i,j,k)

         int find=2*whichPsi; // to save typing

      	 double der_psi1re = derivative_3point(psi[find][cyc(i+1, PointsS)][j][k], psi[find][cyc(i-1, PointsS)][j][k], psi[find][cyc(i+2, PointsS)][j][k],
              psi[find][cyc(i-2, PointsS)][j][k], psi[find][cyc(i+3, PointsS)][j][k], psi[find][cyc(i-3, PointsS)][j][k])/deltaX;
          double der_psi1im = derivative_3point(psi[find+1][cyc(i+1, PointsS)][j][k], psi[find+1][cyc(i-1, PointsS)][j][k], psi[find+1][cyc(i+2, PointsS)][j][k],
              psi[find+1][cyc(i-2, PointsS)][j][k],psi[find+1][cyc(i+3, PointsS)][j][k], psi[find+1][cyc(i-3, PointsS)][j][k])/deltaX;
          double der_psi2re = derivative_3point(psi[find][i][cyc(j+1, PointsS)][k], psi[find][i][cyc(j-1, PointsS)][k], psi[find][i][cyc(j+2, PointsS)][k],
              psi[find][i][cyc(j-2, PointsS)][k], psi[find][i][cyc(j+3, PointsS)][k], psi[find][i][cyc(j-3, PointsS)][k])/deltaX;
          double der_psi2im = derivative_3point(psi[1][i][cyc(j+1, PointsS)][k], psi[1][i][cyc(j-1, PointsS)][k], psi[1][i][cyc(j+2, PointsS)][k],
              psi[find+1][i][cyc(j-2, PointsS)][k], psi[find+1][i][cyc(j+3, PointsS)][k], psi[find+1][i][cyc(j-3, PointsS)][k])/deltaX;
          double der_psi3re = derivative_3point(psi[find][i][j][cyc(k+1, PointsSS)], psi[find][i][j][cyc(k-1, PointsSS)], psi[find][i][j][cyc(k+2, PointsSS)],
              psi[find][i][j][cyc(k-2, PointsSS)], psi[find][i][j][cyc(k+3, PointsSS)], psi[find][i][j][cyc(k-3, PointsSS)])/deltaX;
          double der_psi3im = derivative_3point(psi[find+1][i][j][cyc(k+1, PointsSS)], psi[find+1][i][j][cyc(k-1, PointsSS)], psi[find+1][i][j][cyc(k+2, PointsSS)],
              psi[find+1][i][j][cyc(k-2, PointsSS)], psi[find+1][i][j][cyc(k+3, PointsSS)], psi[find+1][i][j][cyc(k-3, PointsSS)])/deltaX;
          return 0.5* (pow(der_psi1im,2) + pow(der_psi1re,2) + pow(der_psi2im,2) + pow(der_psi2re,2)+ pow(der_psi3im,2) + pow(der_psi3re,2));
        }
        
        // it computes the potential energy density at grid point (i,j,k)
        // virtual for inheritance (external potential)
        // note that Psi fields have ghost points Phi doesn't, k includes ghosts
        virtual inline double energy_pot(const int & i, const int & j, const int & k, int whichPsi){
            return 0.5*(pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2))*Phi[i][j][k-nghost];
        }

        double e_kin_full1(int whichPsi){//Full kinetic energy with Fourier
          double locV= pow(Length,3)*fgrid.e_kin_FT(psi,Length,nghost,whichPsi);
          double totV;
          // gather all the locV for all processes, sums them (MPI_SUM) and returns totV, but is the order correct? 
          // TODO, I changed the order
          MPI_Allreduce(&locV, &totV, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
          return totV;
        }

      long double full_energy_kin(int whichPsi){// it computes the kinetic energy of the whole box with derivatives
        long double total_energy = 0;
        #pragma omp parallel for collapse (3) reduction(+:total_energy)
        for(int i=0;i<PointsS;i++)
          for(int j=0; j<PointsS;j++)
            for(int k=nghost; k<PointsSS+nghost;k++){
              total_energy = total_energy + energy_kin(i,j,k,whichPsi);
            }
        long double total_energy_shared; // total summed up across all nodes
        MPI_Allreduce(&total_energy, &total_energy_shared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        return total_energy_shared*pow(Length/PointsS,3);
      }

      long double full_energy_pot(int whichPsi){// it computes the potential energy of the whole box
        long double total_energy = 0;
        #pragma omp parallel for collapse (3) reduction(+:total_energy)
        for(int i=0;i<PointsS;i++)
          for(int j=0; j<PointsS;j++)
            for(int k=nghost; k<PointsSS+nghost;k++){
              total_energy = total_energy + energy_pot(i,j,k,whichPsi);
            }
        long double total_energy_shared; // total summed up across all nodes
        MPI_Allreduce(&total_energy, &total_energy_shared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        return total_energy_shared*pow(Length/PointsS,3);
      }


	double find_maximum(int whichPsi){ // Sets maxx, maxy, maxz equal to the maximum, it just checks for one global maximum
    double maxdensity = 0;
    for(int i=0;i<PointsS;i++)
      for(int j=0; j<PointsS;j++)
        for(int k=nghost; k<PointsSS+nghost;k++){
          double density_current = pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2);
          if (density_current > maxdensity)
          {maxx=i; maxy=j; maxz=k; maxdensity =density_current;}       // convention is that maxz counts inluding the ghost points
        }

    // now compare across nodes (there's probably a better way to do this, but it's ok for now)
    maxNode=0;
    if(world_rank!=0){  // collect onto node 0
      vector<float> sendup {(float) 1.0000001*maxx,(float)1.00000001* maxy,(float)1.000001*maxz,(float) maxdensity };
      MPI_Send(&sendup.front(), sendup.size(), MPI_FLOAT, 0, 9090, MPI_COMM_WORLD);
    }
    if(world_rank==0){
        for(int lpb=1;lpb<world_size;lpb++){ // recieve from every other node
            vector<float> reci(4,0); //vector to recieve data into
            MPI_Recv(&reci.front(),reci.size(), MPI_FLOAT, lpb, 9090, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            if(reci[3]> maxdensity){maxdensity=reci[3]; maxx=reci[0];  maxy=reci[1];  maxz=reci[2];maxNode=lpb; }
        }
    }

    // now correct on node 0, but now we need to tell all the nodes
    MPI_Bcast(&maxNode, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxx, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxy, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxz, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxdensity, 1, MPI_DOUBLE,0, MPI_COMM_WORLD);
    return maxdensity;
  }


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

        // update the ghosts on the psi grids, need to run this before the timeshots
        void sortGhosts(){
             for(int ii=0;ii<4;ii++){
                transferghosts(psi,ii, world_rank, world_size, nghost);
             }
        }

        // Virtual because for the NFW case (for example) you want to compute radial functions starting from the center of the box and not the maximum
        virtual multi_array<double,2> profile_density(double density_max, int whichPsi);

        // psi -> exp(-i tstep d_alpha Phi) psi; does a step forward or with the opposite sign by changing the sign of tstep
        virtual void expiPhi(double tstep, double da, int whichPsi, double r){
          // Insert the ratio of mass for the second field
          if(whichPsi==0){r=1;}
          #pragma omp parallel for collapse(3)
          for(size_t i=0;i<PointsS;i++)
            for(size_t j=0;j<PointsS;j++)
              for(size_t k=nghost;k<PointsSS+nghost;k++){
                double Repsi=psi[2*whichPsi][i][j][k];
                double Impsi=psi[2*whichPsi+1][i][j][k]; // caution: Phi doesn't have ghost cells so there's a relative shift in k
                psi[2*whichPsi][i][j][k]=cos(-tstep*da*r*Phi[i][j][k-nghost])*Repsi - sin(-tstep*da*r*Phi[i][j][k-nghost])*Impsi;   //real part
                psi[2*whichPsi+1][i][j][k]=sin(-tstep*da*r*Phi[i][j][k-nghost])*Repsi + cos(-tstep*da*r*Phi[i][j][k-nghost])*Impsi;   //im part
          }
          #pragma omp barrier
        }


        void snapshot(double stepCurrent);
        void set_grid(bool grid_bool){//If false, domain3 outputs only the 2D sliced density profile
          Grid3D = grid_bool;
        }
        void set_grid_phase(bool bool_phase){//If false, domain3 does not output the phase slice
          phaseGrid = bool_phase;
        }
        void set_backup_flag(bool bool_backup){//If false, no backup
          start_from_backup = bool_backup;
        }

        void snapshot_profile(double stepCurrent);

        virtual void makestep(double stepCurrent, double tstep){ // makes a step in a dt
          // loop over the 8 values of alpha
          for(int alpha=0;alpha<8;alpha++){
            //1: For any fixed value of alpha, perform the operation exp(-i c_\alpha dt k^2/2)\psi(t) in Fourier space
            for(int whichF=0;whichF<2;whichF++){
              fgrid.inputpsi(psi,nghost,whichF);
              fgrid.calculateFT();                                                  //calculates the FT
              fgrid.kfactorpsi(tstep,Length,ca[alpha],whichF,ratio_mass);                             //multiples the psi by exp(-i c_\alpha dt k^2/2)
              fgrid.calculateIFT();                                                 //calculates the inverse FT
              fgrid.transferpsi(psi,1./pow(PointsS,3),nghost,whichF);                             //divides it by 1/PS^3 (to get the correct normalizaiton of the FT)
            }
            //2: Then perform the operation exp(-i d_\alpha dt \Phi(x)) applied to the previous output
            if(alpha!=7){  //da[7]=0 so no evolution here anyway
              //first, calculate Phi(x) by solving nabla^2Phi = |psi^2|-<|psi^2|> in Fourier space
//                    double psisqavg= psisqmean();                   //calculates the number |psi^2| (it should be conserved)
              fgrid.inputPhi(psi,nghost);                                     //inputs |psi^2|
              fgrid.calculateFT();                                              //calculates its FT, FT(|psi^2|)
              fgrid.kfactorPhi(Length);                                         //calculates -1/k^2 FT(|psi^2|)
              fgrid.calculateIFT();                                             //calculates the inverse FT
              fgrid.transferPhi(Phi,1./pow(PointsS,3));                     //transfers the result into the xytzgrid Phi and multiplies it by 1/PS^3
              //second, multiply the previous output of psi by exp(-i d_\alpha dt \Phi(x)+Phi_ext), in coordinate space
              expiPhi(dt, da[alpha],0,ratio_mass);
              expiPhi(dt, da[alpha],1,ratio_mass);
            }
          }
          long double psisqmean0 = psisqmean(0);
          long double psisqmean1 = psisqmean(1);
          if(world_rank==0){ cout<<"mean value of field 0 "<<psisqmean0<<"  and field 1 "<<psisqmean1<<endl;}
        }

        void solveConvDif(){
          int beginning=time(NULL);
          if (start_from_backup == true)
            openfiles_backup();
          else
            openfiles();
          setoutputs();
          int stepCurrent=0;
          if (start_from_backup == false){
            tcurrent = 0;
            
            snapshot(stepCurrent); // I want the snapshot of the initial conditions
            snapshot_profile(stepCurrent); 
            
            // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
            // (Phi is not computed yet), so store the initial energy after one step
            if(world_rank==0){
              cout<<"current time = "<< tcurrent << " step " << stepCurrent <<endl;
              cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
            }
            makestep(stepCurrent,dt);
            tcurrent=tcurrent+dt;
            stepCurrent=stepCurrent+1;
            E_tot_initial = e_kin_full1(0) + full_energy_pot(0) + e_kin_full1(1) + full_energy_pot(1);
          }
          else if (start_from_backup == true){
            ifstream infile(outputname+"runinfo.txt");
            string temp;
            size_t i = 0;
            while (std::getline(infile, temp, ' ') && i<1){ // convoluted way to read just the first character
              tcurrent = stod(temp);
              i++;
            }
            E_tot_initial = e_kin_full1(0) + full_energy_pot(0)+ e_kin_full1(1) + full_energy_pot(1);
          }

        while(stepCurrent<=numsteps){
          if(world_rank==0){
            cout<<"current time = "<< tcurrent  << " step " << stepCurrent <<endl;
            cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
          }
          makestep(stepCurrent,dt);
          tcurrent=tcurrent+dt;
          stepCurrent=stepCurrent+1;
          for(int index=0;index<=numoutputs;index++)
              if(stepCurrent==jumps[index]) { snapshot(stepCurrent); }
          for(int index=0;index<=numoutputs_profile;index++)
            if(stepCurrent==jumps_profile[index]) {
              snapshot_profile(stepCurrent);
              double etot_current = e_kin_full1(0) + full_energy_pot(0)+ e_kin_full1(1) + full_energy_pot(1);
              // Criterium for dropping by half the time step if energy is not conserved well enough
              if(abs(etot_current-E_tot_initial)/abs(etot_current + E_tot_initial) > 0.001 ){
                dt = dt/2;
                E_tot_initial = etot_current;
              }
              exportValues(); // for backup purposes
              ofstream phi_final;
              outputfullPhi(phi_final);
              ofstream psi_final;
              outputfullPsi(psi_final);
            }
          }
          closefiles();
          cout<<"end"<<endl;
       }

        void initial_cond_from_backup(){
          multi_array<double,1> Arr1D(extents[PointsS*PointsS*PointsSS]);
          ifstream infile(outputname+"phi_final_"+to_string(world_rank)+".txt");
          string temp;
          double num;
          size_t l = 0;
          // Get the input from the input file
          while (std::getline(infile, temp, ' ')) {
          // Add to the list of output strings
          num = stod(temp);
          Arr1D[l] = num;
          l++;
          }
          for(size_t i =0; i<PointsS; i++)
            for(size_t j =0; j<PointsS; j++)
              for(size_t k =0; k<PointsS; k++){
                // cout<<i<<endl;
                Phi[i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS];
              }
          infile.close();
          infile = ifstream(outputname+"psi_final_"+to_string(world_rank)+".txt");
          l = 0;
          multi_array<double,1> Arr1Dpsi(extents[PointsS*PointsS*PointsSS*4]);
          while (std::getline(infile, temp, ' ')) {
            // Add to the list of output strings
            num = stod(temp);
            Arr1Dpsi[l] = num;
            l++;
          }
          for(size_t m =0; m<4; m++)
            for(size_t i =0; i<PointsS; i++)
              for(size_t j =0; j<PointsS; j++)
                for(size_t k =0; k<PointsS; k++){
                  // cout<<i<<endl;
                  psi[m][i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS+m*PointsS*PointsS*PointsSS];
                }
          infile.close();
        }

        // Initial conditions functions
        // Initial condition with waves, test purposes
        void initial_waves();
        void setInitialSoliton_1(double r_c);
        // sets many solitons as initial condition, with random core radius whose centers are confined in a box of length length_lim
        void setManySolitons_random_radius(int num_Sol, double min_radius, double max_radius, double length_lim);
        // sets many solitons as initial condition, with same core radius whose centers are confined in a box of length length_lim
        void setManySolitons_same_radius(int num_Sol, double r_c, double length_lim);
        void setManySolitons_deterministic(double r_c, int num_sol);
        void set_waves_Levkov(double Npart, double Npart1, double vw);
};

#endif


