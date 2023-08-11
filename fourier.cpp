#include "uldm_mpi_2field.h"

using namespace std;
using namespace boost;

Fourier::Fourier(size_t PS, size_t PSS, int WR, int WS){
  int fftw_init_threads(void);
  // In the z direction, less points; loops on z direction start from local_0_start
  Nx=PS; Nz=PSS;
  world_rank=WR; world_size=WS;
  size_t totalValues=Nx*Nx*Nz;
  alloc_local = fftw_mpi_local_size_3d(PS,PS,PS, MPI_COMM_WORLD,&local_n0, &local_0_start);
  rin = fftw_alloc_complex(alloc_local); // memory for input/ output
  fftw_plan_with_nthreads(omp_get_max_threads());
  plan = fftw_mpi_plan_dft_3d(Nx, Nx, Nx, rin , rin, MPI_COMM_WORLD,FFTW_FORWARD, FFTW_MEASURE);
  planback = fftw_mpi_plan_dft_3d(Nx, Nx, Nx, rin , rin, MPI_COMM_WORLD,FFTW_BACKWARD, FFTW_MEASURE);
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
        size_t realk= local_0_start+k;
        rin[i+Nx*j+Nx*Nx*k][1]=cos(i);
        rin[i+Nx*j+Nx*Nx*k][0]=sin(realk);
      }
    }
  }
  #pragma omp barrier
}

// Insert on initial conditions the Levkov waves
void Fourier::inputSpectrum(double Length, double Npart, double vw){
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

//function for putting -k^2 factor; more precisely, psi->exp(-i c_alpha dt k^2/2 ) psi
void Fourier::kfactorpsi(double tstep, double Length, double calpha, int whichPsi, double r){
  // I have to distinguish between the two fields, field 1 has the mass ratio r attached
  size_t i,j,k;
  if(whichPsi==0){r=1;} // Set to one in case you deal with the field 0
  #pragma omp parallel for collapse(3)
  for(i=0;i<Nx;i++)
    for(j=0; j<Nx;j++)
      for(k=0; k<Nz;k++){
        size_t ktrue=local_0_start+k;
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

// input |psi|^2 to the memory that will be FTed
// virtual, should not go in definition, only in declaration
void Fourier::inputPhi(multi_array<double,4> &psi_in, int nghost){ // Luca: I removed the psisqmean, since it is useless
#pragma omp parallel for collapse(3)
for(size_t i=0;i<Nx;i++)
  for(size_t j=0; j<Nx;j++)
    for(size_t k=0; k<Nz;k++){
      rin[i+Nx*j+Nx*Nx*k][0]=
        pow(psi_in[0][i][j][k+nghost],2)+pow(psi_in[1][i][j][k+nghost],2)+pow(psi_in[2][i][j][k+nghost],2)
        +pow(psi_in[3][i][j][k+nghost],2);
      rin[i+Nx*j+Nx*Nx*k][1]=0;
    }
#pragma omp barrier
};

//function for putting -1/k^2 factor on the output of the FT
// needed for solving the Poisson equation to get Phi
void Fourier::kfactorPhi(double Length){
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
      size_t ktrue=local_0_start+k;
      double ksq=(pow(shift(i,Nx),2)+pow(shift(j,Nx),2)+pow(shift(ktrue,Nx),2))*4*M_PI*M_PI/(Length*Length);
      tot_en=tot_en+ksq*(pow(rin[i+Nx*j+Nx*Nx*k][0],2)+ pow(rin[i+Nx*j+Nx*Nx*k][1],2));
    }
return 0.5*tot_en/pow(Nx,6);
}