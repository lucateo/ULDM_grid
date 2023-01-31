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
int beginning=time(NULL);

double fRand(double fMin, double fMax) //random double betwenn fMin and fMax
{
    double f=static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    return fMin + f * (fMax - fMin);
}

// shift function for fourier transform conventions, this one has minus signs
double shift(float i, float N){
    return i<N/2.? i: i-N;
}

// File printing functions
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

//For Levkov waves initial conditions, Np is the number of particles and k2 the squared momentum
inline double P_spec(double k2, double Np){
        return  pow(8*Np,0.5)*exp(-k2/2.)*pow(M_PI,3./4);
}

// cyclic boundary conditions
int cyc(int ar, int le){ // ar is the index where you compute something, le is the size of the grid.
    if(ar>= le){ // if ar > le, meaning you overshoot the grid dimensions, you go (for periodic boundary
        ar=ar-le; // conditions) to the other side of the grid
    }
    else if(ar< 0){
        ar=ar+le;
    }
return ar;
}

// 3 point order derivative, mostly for computing kinetic energy
double derivative_3point(double f1_plus, double f1_minus, double f2_plus, double f2_minus, double f3_plus, double f3_minus){
  double m1 = (f1_plus - f1_minus)/2.0;
  double m2 = (f2_plus - f2_minus)/4.0;
  double m3 = (f3_plus - f3_minus)/6.0;
  return (15*m1 - 6*m2 +m3)/10.0;
}

// Soliton profile in grid units
double psi_soliton(double r_c, double r){
    // lambda^2 of soliton solution with r_c in units of 1/m
    double a = 0.228;
    double b = 4.071;
    double factor_half = pow(2,(double)1.0/(2*b)) -1; // factor which defines r_c, the radius where density drops by half
    double lambda2 = factor_half/(a*a*r_c*r_c);
    return lambda2 / pow(1 + factor_half*pow(r/r_c,2), b);
}

// Class for the Fourier Transform
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
        Fourier(size_t PS, size_t PSS, int WR, int WS){
            int fftw_init_threads(void);
            Nx=PS; Nz=PSS;
	    world_rank=WR; world_size=WS;

            size_t totalValues=Nx*Nx*Nz;

            alloc_local = fftw_mpi_local_size_3d(PS,PS,PS, MPI_COMM_WORLD,&local_n0, &local_0_start);
            rin = fftw_alloc_complex(alloc_local); // memory for input/ output                
            fftw_plan_with_nthreads(omp_get_max_threads());
            plan = fftw_mpi_plan_dft_3d(Nx, Nx, Nx, rin , rin, MPI_COMM_WORLD,FFTW_FORWARD, FFTW_MEASURE);
            planback = fftw_mpi_plan_dft_3d(Nx, Nx, Nx, rin , rin, MPI_COMM_WORLD,FFTW_BACKWARD, FFTW_MEASURE);

            };

        //calculate the FT, more precisely: \tilde{A}_k = \Sum_{n1,n2,n3=0}^{Nx-1} A_n e^(-2\pi i k\cdot n/Nx)
        void calculateFT(){
           fftw_execute(plan);
        }
        // Calculate the unnormalized inverse FT, remember to divide the output by Nx^3
        void calculateIFT(){
           fftw_execute(planback);
        }

        // A sample grid for testing purposes
        void inputsamplegrid(){
            size_t i,j,k;
            #pragma omp parallel for collapse(3)
     	    for(i=0;i<Nx;i++){
                    for(j=0; j<Nx;j++){
                        for(k=0; k<Nz;k++){
			    size_t realk= local_0_start+k;
                            rin[i+Nx*j+Nx*Nx*k][1]=cos(i);
                            rin[i+Nx*j+Nx*Nx*k][0]=sin(realk);
                            }
		         }
            }
	    #pragma omp barrier
	}

        // Insert on initial conditions the Levkov waves
        void inputSpectrum(double Length, double Npart, double vw){
                //size_t i,j,k;
                #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                    for(size_t i=0;i<Nx;i++){
                        for(size_t j=0; j<Nx;j++){
                            for(size_t k=0; k<Nz;k++){
				size_t realk=local_0_start+k;
                                double phase=fRand(0,2*M_PI);
                                double momentum2=
                                    pow(2*M_PI/Length,2)*(pow(shift(i,Nx),2) + pow(shift(j,Nx),2) + pow(shift(realk,Nx),2));
                                double amplit=P_spec( momentum2  , Npart);

                                rin[i+Nx*j+Nx*Nx*k][0]=amplit*cos(phase); //fill in the real part
                                rin[i+Nx*j+Nx*Nx*k][1]=amplit*sin(phase); //fill in the imaginary part
                                }
                        }
                    }
                    #pragma omp barrier
            }

//functions needed for the evolution
//input psi on the array for fftw, note the shift due to the ghost cells
        void inputpsi(multi_array<double,4> &psi, int nghost, int whichPsi){           // whichPsi should be 0 or 1 depending on which field
            size_t i,j,k;
            #pragma omp parallel for collapse(3)
            for(i=0;i<Nx;i++){
                for(j=0; j<Nx;j++){
                    for(k=0; k<Nz;k++){
                        rin[i+Nx*j+Nx*Nx*k][0]= psi[2*whichPsi][i][j][k+nghost];
                        rin[i+Nx*j+Nx*Nx*k][1]= psi[2*whichPsi+1][i][j][k+nghost];
                    }
                }
            }
            #pragma omp barrier
        }

        //function for putting -k^2 factor; more precisely, psi->exp(-i c_alpha dt k^2/2 ) psi
        void kfactorpsi(double tstep, double Length, double calpha){
            size_t i,j,k;
            #pragma omp parallel for collapse(3)
            for(i=0;i<Nx;i++)
              for(j=0; j<Nx;j++)
                 for(k=0; k<Nz;k++){
	            size_t ktrue=local_0_start+k;
                    double ksq= -0.5*tstep*calpha*(pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
                    double repart=rin[i+Nx*j+Nx*Nx*k][0];
                    double impart=rin[i+Nx*j+Nx*Nx*k][1];
                       rin[i+Nx*j+Nx*Nx*k][0]= (cos(ksq)*repart-sin(ksq)*impart);
                       rin[i+Nx*j+Nx*Nx*k][1]= (sin(ksq)*repart+cos(ksq)*impart);
                 }
            #pragma omp barrier
       }

        // Put on psi the result of Fourier transforms
        void transferpsi(multi_array<double,4> &psi, double factor, int nghost, int whichPsi){
            size_t i,j,k;
            #pragma omp parallel for collapse(3)
            for(i=0;i<Nx;i++)
              for(j=0; j<Nx;j++)
                 for(k=0; k<Nz;k++){
     		    psi[2*whichPsi][i][j][k+nghost]=rin[i+Nx*j+Nx*Nx*k][0]*factor;
                    psi[2*whichPsi+1][i][j][k+nghost]=rin[i+Nx*j+Nx*Nx*k][1]*factor;
                    }
            #pragma omp barrier
        }

       // input |psi|^2 to the memory that will be FTed
       // virtual, for inheritance (in case one wants to use an external potential
       virtual void inputPhi(multi_array<double,4> &psi_in, int nghost){ // Luca: I removed the psisqmean, since it is useless
            #pragma omp parallel for collapse(3)
               for(size_t i=0;i<Nx;i++)
                  for(size_t j=0; j<Nx;j++)
                     for(size_t k=0; k<Nz;k++){
                        rin[i+Nx*j+Nx*Nx*k][0]=pow(psi_in[0][i][j][k+nghost],2)+pow(psi_in[1][i][j][k+nghost],2)+pow(psi_in[2][i][j][k+nghost],2)+pow(psi_in[3][i][j][k+nghost],2);
                        rin[i+Nx*j+Nx*Nx*k][1]=0;
                }
            #pragma omp barrier
        };

        //function for putting -1/k^2 factor on the output of the FT
       // needed for solving the Poisson equation to get Phi
       void kfactorPhi(double Length){
            //size_t i,j,k;
            #pragma omp parallel for collapse(3)
            for(size_t i=0;i<Nx;i++)
              for(size_t j=0; j<Nx;j++)
                 for(size_t k=0; k<Nz;k++){
	            size_t ktrue=local_0_start+k;
     		    double ksq=(pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
                    if(ksq==0){
                        rin[i+Nx*j+Nx*Nx*k][1]= 0;
                        rin[i+Nx*j+Nx*Nx*k][0]= 0;
                    }
                    else{
                        rin[i+Nx*j+Nx*Nx*k][1]= (-1./ksq)*rin[i+Nx*j+Nx*Nx*k][1];
                        rin[i+Nx*j+Nx*Nx*k][0]= (-1./ksq)*rin[i+Nx*j+Nx*Nx*k][0];
                    }
               }
            #pragma omp barrier
       }

       // take the result of calculating Phi from the Poisson equation and put it onto a grid
       // Note the normalisation factor here from the inverse FT
        void transferPhi(multi_array<double,3> &Phi, double factor){
            #pragma omp parallel for collapse(3)
            for(size_t i=0;i<Nx;i++)
                for(size_t j=0; j<Nx;j++)
                    for(size_t k=0; k<Nz;k++)
                        Phi[i][j][k]=rin[i+Nx*j+Nx*Nx*k][0]*factor;
            #pragma omp barrier
        }

        //function for the total kinetic energy using the FT
        // note this does not sum up the values on different nodes
       double e_kin_FT(multi_array<double,4> &psi, double Length,int nghost, int whichPsi){
            inputpsi(psi,nghost,whichPsi);
            calculateFT();
            long double tot_en=0;
            #pragma omp parallel for collapse(3) reduction(+:tot_en)
            for(size_t i=0;i<Nx;i++)
              for(size_t j=0; j<Nx;j++)
                 for(size_t k=0; k<Nz;k++){
	            size_t ktrue=local_0_start+k;
     		    double ksq=(pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
                    tot_en=tot_en+ksq*(pow(rin[i+Nx*j+Nx*Nx*k][0],2)+ pow(rin[i+Nx*j+Nx*Nx*k][1],2));
                }

        return 0.5*tot_en/pow(Nx,6);
    }
};



/////////////////////  stuff for the ghosts ///////////////////////////////

// New function to send
void sendg( multi_array<double,4>  &grid,int ii, int world_rank, int world_size, int dir, int nghost){

        // first package the data that needs to be sent into a vector since MPI knows how to deal with this already
        size_t nelements=size_t(grid.shape()[1])*size_t(grid.shape()[1])*nghost;
        vector<double> send(nelements);
        
        size_t track=0;
        size_t j=0;
        size_t k=0;

        size_t ml=grid.shape()[1];
        size_t md=grid.shape()[3];

        for(size_t i=0; i<ml; i++){ 
            for(j=0; j<ml; j++){
		for(k=0; k<nghost; k++){  //convention is to fill from top and bottom as k increases    
                  track = k+j*nghost+i*ml*nghost;
                  if(dir==0){ send[track]=grid[ii][i][j][md-k-1-nghost];} // just filling in the vectors: note we need to send the physical points
                  else{ send[track]=grid[ii][i][j][k+nghost];}                    // not the ghosts
                }
            }
        }

      if(dir==0){MPI_Send(&send.front(), send.size(), MPI_DOUBLE , cyc(world_rank+1,world_size), 20, MPI_COMM_WORLD);}                                                   
      else {MPI_Send(&send.front(), send.size(), MPI_DOUBLE , cyc(world_rank-1,world_size), 21, MPI_COMM_WORLD);}     
}

// New function to receive
void receiveg(multi_array<double,4> &grid,int ii,  int world_rank, int world_size, int dir, int nghost){
        
        size_t nelements=size_t(grid.shape()[1])*size_t(grid.shape()[1])*nghost;     
        vector<double> received(nelements);

        // convention is send up to process +1 with tag +t, and down to process -1 with tag 1000000-t
        // so recieve at bottom from process -1 with tag +t
        if(dir==0){MPI_Recv(&received.front(), received.size(), MPI_DOUBLE, int(cyc(world_rank-1,world_size)), 20, MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
        else {MPI_Recv(&received.front(), received.size(), MPI_DOUBLE, int(cyc(world_rank+1,world_size)), 21, MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
        
        // populate the grid with the transfered data
        size_t track=0;
        size_t j=0;
        size_t k=0;

        size_t ml=grid.shape()[1];
        size_t md=grid.shape()[3];

        #pragma omp parallel for collapse(3) private (track)            
        for(size_t i=0; i<ml; i++){ 
            for(j=0; j<ml; j++){
		for(k=0; k<nghost; k++){  // need to be a bit careful with conventions for filling                                   
                    track = k+size_t(j)*nghost+size_t(i)*ml*nghost;
                    if(dir==0){grid[ii][i][j][md-k-1]=received[track];}
                    else{grid[ii][i][j][k]= received[track];}
                }
            }
        }
       

    }


// MPI stuff to sort out the ghosts
void transferghosts( multi_array<double,4> &gr,int ii, int world_rank, int world_size, int nghost){

    int dir;
    for(dir=0;dir<2;dir++){
    if(world_rank%2==1){ // this trick prevents different processes both waiting for each other to send
              sendg(gr,ii, world_rank,world_size,dir,nghost);    
              receiveg(gr,ii, world_rank,world_size,dir,nghost);
          }
          else{
              receiveg(gr,ii,world_rank,world_size,dir,nghost);
              sendg(gr,ii,world_rank,world_size,dir,nghost);    
           }
    }
}



//////////////////////////////////////////////////////////////////////////////////////////
// Class for the domain to be solved in: builds the grid with the correct parameters    //
//////////////////////////////////////////////////////////////////////////////////////////
class domain3{
  protected:
    multi_array<double,4> psi;  // contains the two fields
    int nghost;                 // number of ghost layers above and below in the z direction 
    multi_array<double,3> Phi;
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
    ofstream timesfile_grid;  //output file for grid
    ofstream timesfile_profile;  //output file for useful information (total energies etc.)
    ofstream info_initial_cond;  //output file for initial condition details
    int snapshotcount=0;          //variable to number the snapshots
    bool Grid3D = false; // If true, it outputs the full density on the 3D grid; if false (recommended), it outputs the 2D projection of the density profile
    bool phaseGrid = false; // If true, it outputs the phase slice passing on the center
//    bool start_from_backup = false; // If true, starts from the backup files, not adapted for MPI yet
    int pointsmax =0;
    int maxx;       //location x,y,z of the max density in the grid
    int maxy;
    int maxz;
    int maxNode;    // node that the maximum value is on
    double maxdensity; // Max density on the grid

    // Variables useful for backup
    double tcurrent; // Current time of simulation
    double E_tot_initial; // Stores the initial total energy to implement the adaptive time step

    int world_rank;
    int world_size;


    public:
        domain3(size_t PS,size_t PSS, double L, int Numsteps, double DT, int Nout, int Nout_profile, string Outputname, int pointsm, int WR, int WS, int Nghost):
            nghost(Nghost),
            psi(extents[4][PS][PS][PSS+2*Nghost]),     //real and imaginary parts are stored consequently, i.e. psi[0]=real part psi1 and psi[1]=imaginary part psi1, then the same for psi2
	    Phi(extents[PS][PS][PSS]),
            fgrid(PS,PSS,WR,WS), // class for Fourier trasform, defined above
            ca(extents[8]),
            da(extents[8]),
            PointsS(PS),
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

        void openfiles(){ // It opens all the output files, and it inserts an initial {
            timesfile_grid.open(outputname+"times_grid.txt");       timesfile_grid<<"{";   timesfile_grid.setf(ios_base::fixed);
            timesfile_profile.open(outputname+"times_profile.txt");       timesfile_profile<<"{";   timesfile_profile.setf(ios_base::fixed);
            profilefile.open(outputname+"profiles.txt");  profilefile<<"{"; profilefile.setf(ios_base::fixed);
            profile_sliced.open(outputname+"profile_sliced.txt");  profile_sliced<<"{"; profile_sliced.setf(ios_base::fixed);
            phase_slice.open(outputname+"phase_slice.txt");  phase_slice<<"{"; phase_slice.setf(ios_base::fixed);
        }

        void openfiles_backup(){ //It opens all the output files in append mode, with no initial insertion of { (for backup mode)
            timesfile_grid.open(outputname+"times_grid.txt", ios_base::app); timesfile_grid.setf(ios_base::fixed);
            timesfile_profile.open(outputname+"times_profile.txt", ios_base::app);  timesfile_profile.setf(ios_base::fixed);
            profilefile.open(outputname+"profiles.txt", ios_base::app); profilefile.setf(ios_base::fixed);
            profile_sliced.open(outputname+"profile_sliced.txt", ios_base::app);   profile_sliced.setf(ios_base::fixed);
            phase_slice.open(outputname+"phase_slice.txt", ios_base::app);   phase_slice.setf(ios_base::fixed);
        }

//         void closefiles(){
        //     timesfile_grid<<"\n}";     timesfile_grid.close();
        //     timesfile_profile<<"\n}";     timesfile_profile.close();
        //     profilefile<<"\n}";   profilefile.close();
        //     profile_sliced<<"\n}";   profile_sliced.close();
        //     phase_slice<<"\n}";   phase_slice.close();
        // }

        // Considering I am implementing the possbility to use backups, this function closes files
        // WITHOUT putting a final }; if one wants to read these arrays with a program (Mathematica etc.),
        // one should load those arrays and put a final }
        void closefiles(){
            timesfile_grid.close();
            timesfile_profile.close();
            profilefile.close();
            profile_sliced.close();
            phase_slice.close();
        }

        // It stores run output info
        virtual void exportValues(){
            runinfo.open(outputname+"runinfo.txt");
            runinfo.setf(ios_base::fixed);
            runinfo<<tcurrent<<" "<<E_tot_initial<<" "<< Length<<" "<<numsteps<<" "<<PointsS<<" "<<numoutputs<<" "<<numoutputs_profile<<endl;
            runinfo.close();
        }

        void setoutputs(double t_ini=0){// Set the indices of the steps when output file function will be called
            for(int inde=0; inde<numoutputs+1; inde++){
                jumps[inde]=int(numsteps/numoutputs*inde);
            }
            for(int inde=0; inde<numoutputs_profile+1; inde++){
                jumps_profile[inde]=int(numsteps/numoutputs_profile*inde);
            }
        }

        // fileout should have a different name on each node
        void outputfulldensity(ofstream& fileout,int whichPsi){// Outputs the full 3D density profile
                multi_array<double,3> density(extents[PointsS][PointsS][PointsSS]);
                #pragma omp parallel for collapse(3)
                    for(size_t i=0;i<PointsS;i++)
                        for(size_t j=0; j<PointsS;j++)
                            for(size_t k=0; k<PointsS;k++){
                                density[i][j][k]= pow(psi[2*whichPsi][i][j][k+nghost],2)+pow(psi[2*whichPsi+1][i][j][k+nghost],2);
                            }
                #pragma omp barrier
                print3(density,fileout);
    }

/*
	// XXXXXXXXXXXXX  Not adapted for MPI yet
        void outputfullPhi(ofstream& fileout){// Outputs the full 3D Phi potential, for backup purposes
                fileout.open(outputname+"phi_final.txt"); fileout.setf(ios_base::fixed);
                print3_cpp(Phi,fileout);
                fileout.close();
    }
        void outputfullPsi_Re(ofstream& fileout){// Outputs the full 3D real psi, for backup purposes
                multi_array<double,3> psi_store(extents[PointsS][PointsS][PointsS]);
                #pragma omp parallel for collapse(3)
                    for(size_t i=0;i<PointsS;i++)
                        for(size_t j=0; j<PointsS;j++)
                            for(size_t k=0; k<PointsS;k++){
                                psi_store[i][j][k]= psi[0][i][j][k];
                            }
                #pragma omp barrier
                fileout.open(outputname+"psiRe_final.txt"); fileout.setf(ios_base::fixed);
                print3_cpp(psi_store,fileout);
                fileout.close();
    }
        void outputfullPsi_Im(ofstream& fileout){// Outputs the full 3D imaginary psi, for backup purposes
                multi_array<double,3> psi_store(extents[PointsS][PointsS][PointsS]);
                #pragma omp parallel for collapse(3)
                    for(size_t i=0;i<PointsS;i++)
                        for(size_t j=0; j<PointsS;j++)
                            for(size_t k=0; k<PointsS;k++){
                                psi_store[i][j][k]= psi[1][i][j][k];
                            }
                #pragma omp barrier
                fileout.open(outputname+"psiIm_final.txt"); fileout.setf(ios_base::fixed);
                print3_cpp(psi_store,fileout);
                fileout.close();
    }
*/


        void outputSlicedDensity(ofstream& fileout, int whichPsi){ // Outputs the projected 2D density profile
                multi_array<double,2> density_sliced(extents[PointsS][PointsS]);
                #pragma omp parallel for collapse(3)
                    for(size_t i=0;i<PointsS;i++)
                        for(size_t j=0; j<PointsS;j++)
                            for(size_t k=nghost; k<PointsSS+nghost;k++){
                                density_sliced[i][j]= density_sliced[i][j] + pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi][i][j][k],2);
                            }
                #pragma omp barrier
                print2(density_sliced,fileout);
    }

	
        void outputPhaseSlice(ofstream& fileout, int whichPsi){ // Outputs a 2D slice of the phase
                multi_array<double,2> phase_sliced(extents[PointsS][PointsS]);
                #pragma omp parallel for collapse(2)
                    for(size_t i=0;i<PointsS;i++)
                        for(size_t j=0; j<PointsS;j++){
                                phase_sliced[i][j]= atan2(psi[2*whichPsi+1][i][j][int(PointsSS/2)+nghost], psi[2*whichPsi][i][j][int(PointsSS/2)+nghost]);
                            }
                #pragma omp barrier
                print2(phase_sliced,fileout);
    }

	
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
          double der_psi3re = derivative_3point(psi[find][i][j][cyc(k+1, PointsS)], psi[find][i][j][cyc(k-1, PointsS)], psi[find][i][j][cyc(k+2, PointsS)],
              psi[find][i][j][cyc(k-2, PointsS)], psi[find][i][j][cyc(k+3, PointsS)], psi[find][i][j][cyc(k-3, PointsS)])/deltaX;
          double der_psi3im = derivative_3point(psi[find+1][i][j][cyc(k+1, PointsS)], psi[find+1][i][j][cyc(k-1, PointsS)], psi[find+1][i][j][cyc(k+2, PointsS)],
              psi[find+1][i][j][cyc(k-2, PointsS)], psi[find+1][i][j][cyc(k+3, PointsS)], psi[find+1][i][j][cyc(k-3, PointsS)])/deltaX;
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
	    MPI_Allreduce(&totV, &locV, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
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
        virtual multi_array<double,2> profile_density(double density_max, int whichPsi){ // It computes the averaged density and energy as function of distance from soliton
        //we need to specify what is the maximum number of points we want to calculate the profile from the center {xmax,ymax,zmax}
        //You have to call find_maximum() first in order to set the maximum values to the class
       
        // note: results only make sense on node 0 

            // vector is easier for the MPI to handle
            //vector that cointains the binned density profile {d, rho(d), E_kin(d), E_pot(d), Phi(d), phase(d)}
            vector<vector<double>> binned(6,vector<double>(pointsmax, 0)); 

            //auxiliary vector to count the number of points in each bin, needed for average
            vector<int> count(pointsmax, 0);  

            int extrak= PointsSS*(world_rank-maxNode);

            #pragma omp parallel for collapse(3)
            for(int i=0;i<PointsS;i++)
              for(int j=0; j<PointsS;j++)
                for(int k=nghost; k<PointsSS+nghost;k++){
                    int Dx=maxx-(int)i; if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
                    int Dy=maxy-(int)j; if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // periodic boundary conditions!
                    int Dz=maxz-(int)k-extrak; if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // periodic boundary conditions!
                    int distance=pow(Dx*Dx+Dy*Dy+Dz*Dz, 0.5);
                    if(distance<pointsmax){
                        //adds up all the points in the 'shell' with radius = distance
                        binned[1][distance] = binned[1][distance] + pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2);
                        binned[2][distance] = binned[2][distance] + energy_kin(i,j,k,whichPsi)*pow(Length/PointsS,3);
                        binned[3][distance] = binned[3][distance] + energy_pot(i,j,k,whichPsi)*pow(Length/PointsS,3);
                        binned[4][distance] = binned[4][distance] + Phi[i][j][k-nghost];   // because Phi doesn't have ghosts
                        // binned[5][distance] = binned[5][distance] + atan2(psi[1][i][j][k], psi[0][i][j][k]);
                        count[distance]=count[distance]+1; //counts the points that fall in that shell
                }
            }
            // For the phase, I take only one ray
            // Only do this on the node that contains the maximum point 
            if(world_rank==maxNode){
              for(int i=0;i<PointsS;i++){
                int distance =maxx-(int)i; if(abs(distance)>PointsS/2){distance=abs(distance)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
                distance = abs(distance);
                if(distance<pointsmax){
                   //Takes only one ray passing from the center of the possible soliton
                   binned[5][distance] = atan2(psi[1][i][maxy][maxz], psi[0][i][maxy][maxz]);
                }
              }
            }


           // collect onto node 0
           if(world_rank!=0){
               MPI_Send(&count.front(), count.size(), MPI_INT, 0, 301, MPI_COMM_WORLD);   
               for(int lp=0;lp<6;lp++){
                  MPI_Send(&binned[lp].front(), binned[lp].size(), MPI_DOUBLE, 0, 300+lp, MPI_COMM_WORLD); 
              }    
           }

          if(world_rank==0){                  
              for(int lpb=1;lpb<world_size;lpb++){ // recieve from every other node
                 vector<vector<double>> recBinned(6, vector<double>(pointsmax,0 ));
                 vector<int> recCount(pointsmax,0); //vector to recieve data into
                 MPI_Recv(&recCount.front(),recCount.size(), MPI_INT, lpb, 301, MPI_COMM_WORLD,MPI_STATUS_IGNORE);            

                 for(int lpc=0;lpc<6;lpc++){
                     MPI_Recv(&recBinned[lpc].front(),recBinned[lpc].size(), MPI_DOUBLE, lpb, 300+lpc, MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
                 }
                 transform(count.begin(), count.end(), recCount.begin(), count.begin(), std::plus<int>());
                 for(int lpc=0;lpc<6;lpc++){
                    transform(binned[lpc].begin(), binned[lpc].end(), recBinned[lpc].begin(), binned[lpc].begin(), std::plus<double>());
                 }
            }
         }

            #pragma omp parallel for
            for(int lp=0;lp<pointsmax;lp++){
                if(count[lp]>0){
                  binned[0][lp]=(lp+0.5)*Length/PointsS;
                  binned[1][lp]=binned[1][lp]/count[lp];// the second component is the average density at that distance
                  binned[2][lp]=binned[2][lp]/count[lp];// Kinetic energy
                  binned[3][lp]=binned[3][lp]/count[lp];// Potential energy
                  binned[4][lp]=binned[4][lp]/count[lp];// Phi (radial)
                }
            }


            // convert back to a multiarray to return

            multi_array<double,2> binnedR(extents[6][pointsmax]);
            #pragma omp parallel for collapse (2)
            for(int ii=0;ii<6;ii++){
               for(int jj=0;jj<pointsmax;jj++){
                   binnedR[ii][jj]=binned[ii][jj];
               }
            }

            return binnedR;
        }




        // psi -> exp(-i tstep d_alpha Phi) psi; does a step forward or with the opposite sign by changing the sign of tstep
        virtual void expiPhi(double tstep, double da, int whichPsi){
            #pragma omp parallel for collapse(3)
                    for(size_t i=0;i<PointsS;i++)
                        for(size_t j=0;j<PointsS;j++)
                            for(size_t k=nghost;k<PointsSS+nghost;k++){
                                double Repsi=psi[2*whichPsi][i][j][k];
                                double Impsi=psi[2*whichPsi+1][i][j][k]; // caution: Phi doesn't have ghost cells so there's a relative shift in k
                                psi[2*whichPsi][i][j][k]=cos(-tstep*da*Phi[i][j][k-nghost])*Repsi - sin(-tstep*da*Phi[i][j][k-nghost])*Impsi;   //real part
                                psi[2*whichPsi+1][i][j][k]=sin(-tstep*da*Phi[i][j][k-nghost])*Repsi + cos(-tstep*da*Phi[i][j][k-nghost])*Impsi;   //im part
                        }
                    #pragma omp barrier
        }



        void snapshot(double stepCurrent,int whichPsi){//Outputs the full density profile; if 3dgrid is false, it outputs just sliced density
            cout.setf(ios_base::fixed);
            if(world_rank==0){
              timesfile_grid<<"{"<<tcurrent<<","<<maxdensity<<","<<maxx<<"," <<maxy<<","<<maxz<<","
                <<e_kin_full1(whichPsi)<<","<<full_energy_pot(whichPsi) << ","<<total_mass(whichPsi)<<"}"<<endl;
                // if(stepCurrent<numsteps){
                timesfile_grid<<","<<flush;
            }
            // } //if it's not the last timeshot put a comma { , , , , }
/*            if(Grid3D == true){
              ofstream grid;
              grid.open(outputname+"densitygrid"+to_string(snapshotcount)+".txt"); grid.setf(ios_base::fixed);
              outputfulldensity(grid);
              grid.close();
              snapshotcount++;
            }
*/
//            else {
            if(world_rank==maxNode){
                cout.setf(ios_base::fixed);
                outputSlicedDensity(profile_sliced,whichPsi);
                if(phaseGrid == true){
                  cout.setf(ios_base::fixed);
                  outputPhaseSlice(phase_slice,whichPsi);
                 }
              // if(stepCurrent< numsteps){
                profile_sliced<<","<<flush;
                if(phaseGrid == true){
                    phase_slice<<","<<flush;
                }
              // } //if it's not the last timeshot put a comma { , , , , }
//           }
            }
//              cout<<"output full grid"<<endl;
        }

        void set_grid(bool grid_bool){//If false, domain3 outputs only the 2D sliced density profile
          Grid3D = grid_bool;
        }

        void set_grid_phase(bool bool_phase){//If false, domain3 does not output the phase slice
          phaseGrid = bool_phase;
        }

/*        void set_backup_flag(bool bool_backup){//If false, domain3 does not output the phase slice
          start_from_backup = bool_backup;
        }
*/
        void snapshot_profile(double stepCurrent,int whichPsi){// Outputs only few relevant information like mass, energy, radial profiles
            cout.setf(ios_base::fixed);
            maxdensity = find_maximum(whichPsi);
            // cout<<"before profile = "<< time(NULL)- beginning <<endl;
            multi_array<double,2> profile = profile_density(maxdensity,whichPsi);
            // cout<<"after profile = "<< time(NULL)- beginning <<endl;
            if(world_rank==0){ print2(profile,profilefile);}
            long double e_pot_full = full_energy_pot(whichPsi);
            long double e_kin_full_1 = full_energy_kin(whichPsi);
            long double e_kin_full_FT = e_kin_full1(whichPsi);
            if(world_rank==0){ 
              timesfile_profile<<"{"<<tcurrent<<","<<maxdensity<<","<<maxx<<"," <<maxy<<","<<maxz<<","
                <<e_kin_full_FT<< ","<< e_kin_full_1 << "," << e_pot_full <<","<<e_kin_full_FT + e_pot_full
                << ","<< e_kin_full_1 + e_pot_full <<","<<total_mass(whichPsi) <<"}"<<endl;
              // if(stepCurrent<numsteps){
              timesfile_profile<<","<<flush;
              profilefile<<","<<flush;
            }
            // } //if it's not the last timeshot put a comma { , , , , }
        }

///////////////////////////////////////////////////////////


        virtual void makestep(double stepCurrent, double tstep){ // makes a step in a dt
            // loop over the 8 values of alpha
            for(int alpha=0;alpha<8;alpha++){
                //1: For any fixed value of alpha, perform the operation exp(-i c_\alpha dt k^2/2)\psi(t) in Fourier space
                for(int whichF=0;whichF<2;whichF++){
                    fgrid.inputpsi(psi,nghost,whichF);
                    fgrid.calculateFT();                                                  //calculates the FT
                    fgrid.kfactorpsi(tstep,Length,ca[alpha]);                             //multiples the psi by exp(-i c_\alpha dt k^2/2)
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
                    expiPhi(dt, da[alpha],0);
                    expiPhi(dt, da[alpha],1);
                }
            }
            if(world_rank==0){ cout<<"mean value of field 0 "<<psisqmean(0)<<"  and field 1 "<<psisqmean(1)<<endl;}
        }

        void solveConvDif(){
//            if (start_from_backup == true)
//              openfiles_backup();
//            else
            openfiles();
            setoutputs();
            int stepCurrent=0;
//            if (start_from_backup == false){
              tcurrent = 0;

// for now i just take snapshots of field 0
              snapshot(stepCurrent,0); // I want the snapshot of the initial conditions
              snapshot_profile(stepCurrent,0); // I want the snapshot of the initial conditions
              // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
              // (Phi is not computed yet), so store the initial energy after one step
              cout<<"current time = "<< tcurrent <<endl;
              cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
              makestep(stepCurrent,dt);
              tcurrent=tcurrent+dt;
              stepCurrent=stepCurrent+1;
              E_tot_initial = e_kin_full1(0) + full_energy_pot(0) + e_kin_full1(1) + full_energy_pot(1);
//            }
/*            else if (start_from_backup == true){
              ifstream infile(outputname+"runinfo.txt");
              string temp;
              size_t i = 0;
              while (std::getline(infile, temp, ' ') && i<1){ // convoluted way to read just the first character
                tcurrent = stod(temp);
                i++;
              }
              E_tot_initial = e_kin_full1() + full_energy_pot();
            }
*/
            while(stepCurrent<=numsteps){
                cout<<"current time = "<< tcurrent <<endl;
                cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
                makestep(stepCurrent,dt);
                tcurrent=tcurrent+dt;
                stepCurrent=stepCurrent+1;
                for(int index=0;index<=numoutputs;index++)
                    if(stepCurrent==jumps[index]) { snapshot(stepCurrent,0); }
                for(int index=0;index<=numoutputs_profile;index++)
                    if(stepCurrent==jumps_profile[index]) {
                      snapshot_profile(stepCurrent,0);
                      double etot_current = e_kin_full1(0) + full_energy_pot(0)+ e_kin_full1(1) + full_energy_pot(1);
                      // Criterium for dropping by half the time step if energy is not conserved well enough
                      if(abs(etot_current-E_tot_initial)/abs(etot_current + E_tot_initial) > 0.001 ){
                        dt = dt/2;
                        E_tot_initial = etot_current;
                      }
/*                  exportValues(); // for backup purposes
                  ofstream phi_final;
                  outputfullPhi(phi_final);
                  ofstream psiRe_final;
                  outputfullPsi_Re(psiRe_final);
                  ofstream psiIm_final;
                  outputfullPsi_Im(psiIm_final);
*/
                  }
            }
            closefiles();
            cout<<"end"<<endl;
       }

/*        void initial_cond_from_backup(){
          multi_array<double,1> Arr1D(extents[PointsS*PointsS*PointsS]);
          ifstream infile(outputname+"phi_final.txt");
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
          infile = ifstream(outputname+"psiRe_final.txt");
          l = 0;
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
                psi[0][i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS];
              }
          infile.close();
          infile = ifstream(outputname+"psiIm_final.txt");
          l = 0;
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
                psi[1][i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS];
              }
        }
*/

// Initial conditions functions
        // Initial condition with waves, test purposes
        void initial_waves(){
            #pragma omp parallel for collapse(3)
            for(size_t i=0;i<PointsS;i++)
                for(size_t j=0;j<PointsS;j++)
                    for(size_t k=nghost;k<PointsSS+nghost;k++){
                        size_t kreal=k+world_rank*PointsSS;
                        psi[0][i][j][k]=sin(i/20.)*cos(kreal/20.);
                        psi[1][i][j][k]=sin(kreal/20.)*cos((kreal+i)/20.);
                    }
        }

/*
        void setInitialSoliton_1_randomVel(double r_c){ // sets 1 soliton in the center of the grid as initial condition, for testing purposes, with random velocity
              int center = (int) PointsS / 2; // The center of the grid, more or less
                double v_x = fRand(0,1);
                double v_y = fRand(0,1);
                double v_z = fRand(0,1);
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          // Distance from the center of the soliton
                          double radius = deltaX * sqrt( pow( abs(i - center),2) + pow( abs(j - center),2) + pow( abs(k - center),2));
                          double phase = deltaX * (v_x * i + v_y * j + v_z * k);
                          psi[0][i][j][k] = psi_soliton(r_c, radius) * cos( phase );
                          psi[1][i][j][k] = psi_soliton(r_c, radius) * sin( phase );
                          }
                    }
                }
                #pragma omp barrier
            }

        void setInitialSoliton_1(double r_c){ // sets 1 soliton in the center of the grid as initial condition, for testing purposes
              int center = (int) PointsS / 2; // The center of the grid, more or less
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          // Distance from the center of the soliton
                          double radius = deltaX * sqrt( pow( abs(i - center),2) + pow( abs(j - center),2) + pow( abs(k - center),2));
                          psi[0][i][j][k] = psi_soliton(r_c, radius);
                          psi[1][i][j][k] = 0; // Set the imaginary part to zero
                          }
                    }
                }
                #pragma omp barrier
            }

        void setInitialSoliton_2(double r_c1, double r_c2, double distance){ // sets 2 solitons as initial condition, for testing purposes
              int center = (int) PointsS / 2; // The center of the grid, more or less
              int center1 = center + (int) round(distance/(2*deltaX)); // coordinates of first soliton in x axis
              int center2 = center - (int) round(distance/(2*deltaX)); // coordinates of second soliton in x axis
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          // Distance from the center of the soliton, located at the center of the grid but displaced in x axis by distance/2
                          double radius1 = deltaX * sqrt( pow( i-center1,2) + pow( j-center,2) + pow( k -center,2));
                          double radius2 = deltaX * sqrt( pow( i-center2,2) + pow( j-center,2) + pow( k-center,2));
                          psi[0][i][j][k] = psi_soliton(r_c1, radius1) + psi_soliton(r_c2, radius2);
                          psi[1][i][j][k] = 0; // Set the imaginary part to zero
                          }
                    }
                }
                #pragma omp barrier
            }

        // sets many solitons as initial condition, with random core radius whose centers are confined in a box of length length_lim
        void setManySolitons_random_radius(int num_Sol, double min_radius, double max_radius, double length_lim){
              int random_x[num_Sol];
              int random_y[num_Sol];
              int random_z[num_Sol];
              double r_c[num_Sol]; // the core radius array

              info_initial_cond.open(outputname+"initial_cond_info.txt");
              info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, min_rad, max_rad and length_lim, the other rows are centers of solitons and radii
              info_initial_cond<<"{{"<<num_Sol<<","<<min_radius<<","<<max_radius << ","<<length_lim << "}"<< "," <<endl;
              for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
                r_c[i] = fRand(min_radius, max_radius);
                random_x[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
                random_y[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
                random_z[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
                info_initial_cond<<"{"<< r_c[i] << "," <<random_x[i]<<","<<random_y[i]<< ","<<random_z[i]<< "}";
                if (i<num_Sol-1) {info_initial_cond<<","<<endl;} // If it is not the last data point, put a comma
              }
            info_initial_cond<<"}";
            info_initial_cond.close();
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          psi[0][i][j][k] = 0; //set it to zero just to be sure
                          for(int l=0; l<num_Sol; l++){
                            double radius = deltaX * sqrt( pow( i-random_x[l],2) + pow( j-random_y[l],2) + pow( k-random_z[l],2));
                            psi[0][i][j][k] += psi_soliton(r_c[l], radius);
                          }
                          psi[1][i][j][k] = 0; // Set the imaginary part to zero
                        }
                    }
                }
                #pragma omp barrier
            }

        // sets many solitons as initial condition, with same core radius whose centers are confined in a box of length length_lim
        void setManySolitons_same_radius(int num_Sol, double r_c, double length_lim){
                #pragma omp barrier
              int random_x[num_Sol];
              int random_y[num_Sol];
              int random_z[num_Sol];
              info_initial_cond.open(outputname+"initial_cond_info.txt");
              info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, r_c and length_lim, the other rows are centers of solitons
            info_initial_cond<<"{{"<<num_Sol<<","<<r_c<<","<<length_lim << "}"<< "," <<endl;
              for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
                random_x[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
                random_y[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
                random_z[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
                info_initial_cond<<"{"<<random_x[i]<<","<<random_y[i]<< ","<<random_z[i]<< "}";
                if (i<num_Sol-1) {info_initial_cond<<","<<endl;} // If it is not the last data point, put a comma
              }
            info_initial_cond<<"}";
            info_initial_cond.close();
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          psi[0][i][j][k] = 0; //set it to zero just to be sure
                          for(int l=0; l<num_Sol; l++){
                            double radius = deltaX * sqrt( pow( i-random_x[l],2) + pow( j-random_y[l],2) + pow( k-random_z[l],2));
                            psi[0][i][j][k] += psi_soliton(r_c, radius);
                          }
                          psi[1][i][j][k] = 0; // Set the imaginary part to zero
                        }
                    }
                }
        }

        // sets many solitons as initial condition, with same core radius, and random initial velocities (NOT REFINED)
        void setManySolitons_same_radius_random_vel(int num_Sol, double r_c){
                #pragma omp barrier
              int random_x[num_Sol];
              int random_y[num_Sol];
              int random_z[num_Sol];
                double v_x[num_Sol];
                double v_y[num_Sol];
                double v_z[num_Sol];
              for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
                random_x[i]= 2*round(r_c/deltaX) + rand()%(int)(PointsS - 4*round(r_c/deltaX) );
                random_y[i]= 2*round(r_c/deltaX) + rand()%(int)(PointsS - 4*round(r_c/deltaX) );
                random_z[i]= 2*round(r_c/deltaX) + rand()%(int)(PointsS - 4*round(r_c/deltaX) );
                v_x[i] = fRand(0,1);
                v_y[i] = fRand(0,1);
                v_z[i] = fRand(0,1);
              }
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          psi[0][i][j][k] = 0; //set it to zero just to be sure
                          psi[1][i][j][k] = 0; //set it to zero just to be sure
                          for(int l=0; l<num_Sol; l++){
                            double radius = deltaX * sqrt( pow( i-random_x[l],2) + pow( j-random_y[l],2) + pow( k-random_z[l],2));
                            double phase=deltaX*(v_x[l] *i + v_y[l] *j + v_z[l] *k );
                            psi[0][i][j][k] += psi_soliton(r_c, radius)*cos(phase);
                          psi[1][i][j][k] += psi_soliton(r_c, radius)*sin(phase); // Set the imaginary part to zero
                          }
                        }
                    }
                }
        }

        void setManySolitons_deterministic(double r_c, int num_sol){
        // sets num_sol solitons (maximum of 30 solitons allowed)
        // as initial condition, with same core radius, with fixed positions (chosen randomly), for testing purposes
                #pragma omp barrier
              float x_phys[30] ={-10.010355128996338, -5.632161391614467, -10.804162195884118, 27.74117880694743, -2.4011805803557067, -23.25311822651404, 25.705720076675775, 37.353679014311595, -6.946204576810793, -23.817538236920797, -14.655394739433873, -16.749637898661014, -19.580883326003217, 18.605740761167112, 27.25736094744731, -30.41462054172923, -21.33315514195605, 16.197348275236898, 3.799620205223448, 6.934702548652282, -16.941096181540544, 2.2781062254869866, 36.8554262382406, 36.35367270175507, 11.139625002335258, -30.58395762855323, 17.390547027384528, 35.351726508377, 34.30615396835661, 22.71362595499526} ;
              float y_phys[30] = {31.886070793802816, 39.38615345350138, 24.231389564760576, -15.835439299197375, -3.3411149363077755, -5.730194340705353, 0.7324849001187346, 33.325544978300755, -16.688024218649502, 9.90167057634315, 25.356905509911215, -2.122232317006876, -15.220208473815227, 7.791086225543076, 10.479565234160418, -1.9731717922202847, 15.462493807443863, -6.4553532820384305, -20.80826527413479, 22.966765839735572, -33.034190626746195, 28.861838426003686, 33.1885768361993, 14.555629618935612, -1.4330134748238805, -9.755948008891544, -2.6452408889946426, 16.665940880504834, 39.45576480300235, -24.93284829451899};
              float z_phys[30] = {-31.810875938706424, 36.60281686740946, 21.634248190233677, -1.0920954142129702, -35.06404006114212, 20.92585035786638, -9.280765022493433, 0.8541286579386949, 0.07578851246192642, -17.007163121712257, 18.26124323465008, 5.980555848100558, 6.464719622534602, -15.047019341759356, 16.629246652083538, 20.983444417130237, -10.701969326834153, -36.6850662895899, -9.833232106681116, -15.800997680268427, 35.40845053441541, 8.837229367223543, -16.95296121220478, -10.596280760678791, -22.789096875557515, -39.44176295453228, 31.794778400711067, -38.161234613721355, 17.82624369418062, -28.17723677444623};
              int center = (int) PointsS / 2; // The center of the grid, more or less
              int x[num_sol];
              int y[num_sol];
              int z[num_sol];
              // int center1 = center - (int) round((num_Sol_lin_dim -1)*distance/(2*deltaX)); // coordinates of edge of box
              for(int i=0;i<num_sol;i++){
                x[i]= center + (int) round(x_phys[i] / deltaX);
                y[i]= center + (int) round(y_phys[i] / deltaX);
                z[i]= center + (int) round(z_phys[i] / deltaX);
              }
              #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
                for(int i=0;i<PointsS;i++){
                    for(int j=0; j<PointsS;j++){
                        for(int k=0; k<PointsS;k++){
                          psi[0][i][j][k] = 0; //set it to zero just to be sure
                          for(int l=0; l<num_sol; l++){
                            double radius = deltaX * sqrt( pow( i-x[l],2) + pow( j-y[l],2) + pow( k-z[l],2));
                            psi[0][i][j][k] += psi_soliton(r_c, radius);
                          }
                          psi[1][i][j][k] = 0; // Set the imaginary part to zero
                        }
                    }
                }
            }
*/

        void set_waves_Levkov(double Npart, double vw){
          // Sets waves initial conditions a la Levkov, Npart is number of particles, vw is velocity (set it to zero)
            fgrid.inputSpectrum(Length, Npart,vw);
            fgrid.calculateFT();
            fgrid.transferpsi(psi, 1./pow(Length,3),nghost,0);

            fgrid.inputSpectrum(Length, Npart,vw);
            fgrid.calculateFT();
            fgrid.transferpsi(psi, 1./pow(Length,3),nghost,01);

            if(world_rank==0){
              info_initial_cond.open(outputname+"initial_cond_info.txt");
              info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, r_c and length_lim, the other rows are centers of solitons
              info_initial_cond<<"{"<<Npart<<","<<vw << "}" <<endl;
              info_initial_cond.close();
            }
        }

};




