#include "uldm_mpi_2field.h"

using namespace std;
using namespace boost;

double fRand(double fMin, double fMax) //random double betwenn fMin and fMax
{
    double f=static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    return fMin + f * (fMax - fMin);
}

// shift function for fourier transform conventions, this one has minus signs
double shift(float i, float N){
    return i<N/2.? i: i-N;
}

//For Levkov waves initial conditions, Np is the number of particles and k2 the squared momentum
double P_spec(double k2, double Np){
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

/////////////////////  stuff for the ghosts ///////////////////////////////
// New function to send
void sendg( multi_array<double,4>  &grid,int ii, int world_rank, int world_size, int dir, int nghost){
        // first package the data that needs to be sent into a vector since MPI knows how to deal with this already
        // grid is psi vector, ii is real/imaginary index, dir = 0 is the top of the grid,
        // dir=1 is the bottom of the grid; apparently, this sends the physical layer (i.e. no ghost layers)
        // just above (dir=1) or below (dir=0) the ghost layer to the ghost layer of neighbors.
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

      // Sends ro the node with world_rank+1 (account for periodicity), but if dir==1, it sends to world_rank-1
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


// MPI stuff to sort out the ghosts, needed for gradients in real space
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