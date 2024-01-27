#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include <cmath>

using namespace std;
using namespace boost;

Fourier::Fourier(size_t PS, size_t PSS, int WR, int WS, bool mpi){
  // In the z direction, less points; loops on z direction start from local_0_start
  Nx=PS; Nz=PSS;
  world_rank=WR; world_size=WS;
  size_t totalValues=Nx*Nx*Nz;
  int fftw_init_threads(void);
  if(mpi==true){
    alloc_local = fftw_mpi_local_size_3d(PS,PS,PS, MPI_COMM_WORLD,&local_n0, &local_0_start);
    rin = fftw_alloc_complex(alloc_local); // memory for input/ output
  }
  else {
    rin = (fftw_complex *) malloc(totalValues * sizeof(fftw_complex));
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
  if(mpi==true){
    plan = fftw_mpi_plan_dft_3d(Nx, Nx, Nx, rin , rin, MPI_COMM_WORLD,FFTW_FORWARD, FFTW_MEASURE);
    planback = fftw_mpi_plan_dft_3d(Nx, Nx, Nx, rin , rin, MPI_COMM_WORLD,FFTW_BACKWARD, FFTW_MEASURE);
  }
  else {
    plan = fftw_plan_dft_3d(Nx, Nx, Nx, rin , rin, FFTW_FORWARD, FFTW_MEASURE);
    planback= fftw_plan_dft_3d(Nx, Nx, Nx, rin , rin,FFTW_BACKWARD, FFTW_MEASURE);
  }
};

Fourier::Fourier(){};
Fourier::~Fourier(){};

void Fourier::calculateFT(){
  fftw_execute(plan);
}
// Calculate the unnormalized inverse FT, remember to divide the output by Nx^3
void Fourier::calculateIFT(){
  fftw_execute(planback);
}

// A sample grid for testing purposes
void Fourier::inputsamplegrid(){
  size_t i,j,k;
  #pragma omp parallel for collapse(3)
  for(i=0;i<Nx;i++){
    for(j=0; j<Nx;j++){
      for(k=0; k<Nz;k++){
        size_t realk= world_rank*Nz+k;
        rin[i+Nx*j+Nx*Nx*k][1]=cos(i);
        rin[i+Nx*j+Nx*Nx*k][0]=sin(realk);
      }
    }
  }
  #pragma omp barrier
}

// Insert on initial conditions the Levkov waves
void Fourier::inputSpectrum(double Length, double Npart, double r){
  //size_t i,j,k;
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
  for(size_t i=0;i<Nx;i++){
    for(size_t j=0; j<Nx;j++){
      for(size_t k=0; k<Nz;k++){
        size_t realk=world_rank*Nz+k;
        double phase=fRand(0,2*M_PI);
        double momentum2=
          pow(2*M_PI/Length,2)*(pow(shift(i,Nx),2) + pow(shift(j,Nx),2) + pow(shift(realk,Nx),2));
        double amplit=P_spec( momentum2/(r*r) , Npart)/r;

        rin[i+Nx*j+Nx*Nx*k][0]=amplit*cos(phase); //fill in the real part
        rin[i+Nx*j+Nx*Nx*k][1]=amplit*sin(phase); //fill in the imaginary part
      }
    }
  }
  #pragma omp barrier
}

// Insert on initial conditions a delta function in Fourier space
void Fourier::inputDelta(double Length, double Npart, double r){
  //size_t i,j,k;
  double amplit= sqrt(Npart *M_PI*Length/r);
  double r_comparison = r*Length/(2*M_PI);
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
  for(size_t i=0;i<Nx;i++){
    for(size_t j=0; j<Nx;j++){
      for(size_t k=0; k<Nz;k++){
        size_t realk=world_rank*Nz+k;
        double k_discrete=
          sqrt((pow(shift(i,Nx),2) + pow(shift(j,Nx),2) + pow(shift(realk,Nx),2)));
        if (k_discrete <= r_comparison+0.5 && k_discrete >= r_comparison-0.5 ) {
          double phase=fRand(0,2*M_PI);
          rin[i+Nx*j+Nx*Nx*k][0]=amplit*cos(phase); //fill in the real part
          rin[i+Nx*j+Nx*Nx*k][1]=amplit*sin(phase); //fill in the imaginary part
        }
      }
    }
  }
  #pragma omp barrier
}

// Insert on initial conditions a Heaviside function in Fourier space
void Fourier::inputTheta(double Length, double Npart, double r){
  //size_t i,j,k;
  double count = 0;
  double amplit= sqrt(6*Npart) * M_PI/r;
  double r_comparison = r*Length/(2*M_PI);
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
  for(size_t i=0;i<Nx;i++){
    for(size_t j=0; j<Nx;j++){
      for(size_t k=0; k<Nz;k++){
        size_t realk=world_rank*Nz+k;
        double k_discrete=
          sqrt((pow(shift(i,Nx),2) + pow(shift(j,Nx),2) + pow(shift(realk,Nx),2)));
        if (k_discrete <= r_comparison ) {
          count++;
          double phase=fRand(0,2*M_PI);
          rin[i+Nx*j+Nx*Nx*k][0]=amplit*cos(phase); //fill in the real part
          rin[i+Nx*j+Nx*Nx*k][1]=amplit*sin(phase); //fill in the imaginary part
        }
      }
    }
  }
  #pragma omp barrier
  cout<<count/(4*M_PI/3*pow(r_comparison,3))<<endl;
}

//functions needed for the evolution
//input psi on the array for fftw, note the shift due to the ghost cells
// whichPsi should be 0 or 1 depending on which field
void Fourier::inputpsi(multi_array<double,4> &psi, int nghost, int whichPsi){
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

void Fourier::input_arr(multi_array<double,1> &arr1d, int whichfield, int which_coordinate){
  size_t i;
  for(i=0;i<Nx*Nx*Nz;i++){
        rin[i][0]= arr1d[i+ which_coordinate*Nx*Nx*Nz + whichfield*Nx*Nx*Nz*3];
        rin[i][1]= 0;
  }
}

void Fourier::kfactor_vel(double Length,double r, int which_coord){
  // I have to distinguish between the two fields, field 1 has the mass ratio r attached
  size_t i,j,k;
  #pragma omp parallel for collapse(3)
  for(i=0;i<Nx;i++)
    for(j=0; j<Nx;j++)
      for(k=0; k<Nz;k++){
        size_t ktrue=world_rank*Nz+k;
        double ksq= (pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
        double kx;
        if(which_coord==0)
          kx = shift(i,Nx)*2*M_PI/Length;
        else if(which_coord==1)
          kx = shift(j,Nx)*2*M_PI/Length;
        if(which_coord==2)
          kx = shift(ktrue,Nx)*2*M_PI/Length;
        double repart=rin[i+Nx*j+Nx*Nx*k][0];
        double impart=rin[i+Nx*j+Nx*Nx*k][1];
        rin[i+Nx*j+Nx*Nx*k][0]= -r*kx*impart/ksq;
        rin[i+Nx*j+Nx*Nx*k][1]= r*kx*repart/ksq;
      }
  #pragma omp barrier
}

//function for putting -k^2 factor; more precisely, psi->exp(-i c_alpha dt k^2/2 ) psi
void Fourier::kfactorpsi(double tstep, double Length, double calpha, double r){
  // I have to distinguish between the two fields, field 1 has the mass ratio r attached
  size_t i,j,k;
  #pragma omp parallel for collapse(3)
  for(i=0;i<Nx;i++)
    for(j=0; j<Nx;j++)
      for(k=0; k<Nz;k++){
        size_t ktrue=world_rank*Nz+k;
        double ksq= -0.5*tstep*calpha/r*(pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
        double repart=rin[i+Nx*j+Nx*Nx*k][0];
        double impart=rin[i+Nx*j+Nx*Nx*k][1];
        rin[i+Nx*j+Nx*Nx*k][0]= (cos(ksq)*repart-sin(ksq)*impart);
        rin[i+Nx*j+Nx*Nx*k][1]= (sin(ksq)*repart+cos(ksq)*impart);
      }
  #pragma omp barrier
}

// Put on psi the result of Fourier transforms
void Fourier::transferpsi(multi_array<double,4> &psi, double factor, int nghost, int whichPsi){
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

// Add to psi the result of Fourier transforms, mostly used to stack different initial conditions
void Fourier::transferpsi_add(multi_array<double,4> &psi, double factor, int nghost, int whichPsi){
  size_t i,j,k;
  #pragma omp parallel for collapse(3)
  for(i=0;i<Nx;i++)
    for(j=0; j<Nx;j++)
      for(k=0; k<Nz;k++){
        psi[2*whichPsi][i][j][k+nghost]+=rin[i+Nx*j+Nx*Nx*k][0]*factor;
        psi[2*whichPsi+1][i][j][k+nghost]+=rin[i+Nx*j+Nx*Nx*k][1]*factor;
      }
  #pragma omp barrier
}

// input |psi|^2 to the memory that will be FTed
// virtual, should not go in definition, only in declaration
void Fourier::inputPhi(multi_array<double,4> &psi_in, int nghost, int nfields){ // Luca: I removed the psisqmean, since it is useless
#pragma omp parallel for collapse(3)
for(size_t i=0;i<Nx;i++)
  for(size_t j=0; j<Nx;j++)
    for(size_t k=0; k<Nz;k++){
      rin[i+Nx*j+Nx*Nx*k][0]= 0;
      for(size_t l = 0; l <2*nfields; l++){
        rin[i+Nx*j+Nx*Nx*k][0]+=pow(psi_in[l][i][j][k+nghost],2);
      }
      rin[i+Nx*j+Nx*Nx*k][1]=0;
    }
#pragma omp barrier
}

//function for putting -1/k^2 factor on the output of the FT
// needed for solving the Poisson equation to get Phi
void Fourier::kfactorPhi(double Length){
//size_t i,j,k;
#pragma omp parallel for collapse(3)
for(size_t i=0;i<Nx;i++)
  for(size_t j=0; j<Nx;j++)
    for(size_t k=0; k<Nz;k++){
      size_t ktrue=world_rank*Nz+k;
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
void Fourier::transferPhi(multi_array<double,3> &Phi, double factor){
    #pragma omp parallel for collapse(3)
    for(size_t i=0;i<Nx;i++)
        for(size_t j=0; j<Nx;j++)
            for(size_t k=0; k<Nz;k++)
                Phi[i][j][k]=rin[i+Nx*j+Nx*Nx*k][0]*factor;
    #pragma omp barrier
}

//function for the total kinetic energy using the FT
// note this does not sum up the values on different nodes
double Fourier::e_kin_FT(multi_array<double,4> &psi, double Length,int nghost, int whichPsi){
  inputpsi(psi,nghost,whichPsi);
  calculateFT();
  long double tot_en=0;
  #pragma omp parallel for collapse(3) reduction(+:tot_en)
  for(size_t i=0;i<Nx;i++)
    for(size_t j=0; j<Nx;j++)
      for(size_t k=0; k<Nz;k++){
        size_t ktrue=world_rank*Nz+k;
        double ksq=(pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
        tot_en=tot_en+ksq*(pow(rin[i+Nx*j+Nx*Nx*k][0],2)+ pow(rin[i+Nx*j+Nx*Nx*k][1],2));
      }
  #pragma omp barrier
  return 0.5*tot_en/pow(Nx,6);
}

//function for the center of mass velocity using the FT
// note this does not sum up the values on different nodes
double Fourier::v_center_mass_FT(multi_array<double,4> &psi, double Length,int nghost, int whichPsi, int coordinate){
  inputpsi(psi,nghost,whichPsi);
  calculateFT();
  long double tot_vcm=0;
  #pragma omp parallel for collapse(3) reduction(+:tot_vcm)
  for(size_t i=0;i<Nx;i++)
    for(size_t j=0; j<Nx;j++)
      for(size_t k=0; k<Nz;k++){
        size_t ktrue=world_rank*Nz+k;
        double kx;
        if (coordinate==0) kx = shift(i,Nx)*2*M_PI/Length;
        else if(coordinate==1) kx = shift(j,Nx)*2*M_PI/Length;
        else if(coordinate==2) kx = shift(k,Nx)*2*M_PI/Length;
        tot_vcm=tot_vcm+kx*(pow(rin[i+Nx*j+Nx*Nx*k][0],2)+ pow(rin[i+Nx*j+Nx*Nx*k][1],2));
      }
  #pragma omp barrier
  return tot_vcm/pow(Nx,6)*pow(Length,3);
}