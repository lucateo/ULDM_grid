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

// Cyclic boundary conditions with double
double cyc_double(double ar, double le){ // ar is the index where you compute something, le is the size of the grid.
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

// First order derivative with 5 points-midpoint, f2_plus = f(x + 2h), f_minus = f(x-h) etc.
double derivative_5midpoint(double f2_plus, double f_plus, double f_minus, double f2_minus, double deltaX){
  return (f2_minus-8*f_minus+8*f_plus-f2_plus)/12/deltaX;
}

// Soliton profile in grid units
double psi_soliton(double r_c, double r, double ratio){
    // lambda^2 of soliton solution with r_c in units of 1/m
    double a = 0.228*sqrt(ratio);
    double b = 4.071;
    // double a = 0.320748*sqrt(ratio);
    // double b = 4.29637293;
    double factor_half = pow(2,(double)1.0/(2*b)) -1; // factor which defines r_c, the radius where density drops by half
    double lambda2 = factor_half/(a*a*r_c*r_c);
    return lambda2 / pow(1 + factor_half*pow(r/r_c,2), b);
    // return r_c*r_c / pow(1 + pow(a*r*r_c,2), b);
}

vector<double> num_second_derivative(vector<double> & xarr, vector<double> & yarr){
  int Ndim = xarr.size();
  cout<< Ndim << endl;
  vector<double> result(Ndim-2);
  for(int i=1; i< Ndim-1; i++){ // avoid boundaries
    double der = yarr[i+1]*(xarr[i] - xarr[i-1]) + yarr[i-1]*(xarr[i+1] - xarr[i]) - yarr[i]*(xarr[i+1]- xarr[i-1]);
    der = der/(0.5*(xarr[i+1]- xarr[i-1])* (xarr[i+1] - xarr[i])*(xarr[i]- xarr[i-1]));
    result[i-1] = der;
  }
  return result;
}

multi_array<double,1> Integral_trapezoid(multi_array<double,1> & xarr, multi_array<double,1> & yarr){
  int Ndim = xarr.shape()[0];
  multi_array<double,1> result(extents[Ndim]);
  result[0] = yarr[0]*xarr[0]; // Assume "x[-1]" to be 0 and y to be constant before x[0]
  for(int i=1; i< Ndim; i++){ // avoid boundaries
    result[i] = result[i-1] + 0.5*(yarr[i-1] + yarr[i])*(xarr[i] - xarr[i-1]);
  }
  return result;
}

double interpolant(double x, vector<double> & xarr, vector<double> & yarr){
  int Nx = xarr.size();
  double result=-1;
  int itrue;
  if(x<= xarr[0]){
    result = yarr[0];
  }
  else if (x<= xarr[Nx-1]){
    for (int i = 1; i < Nx;i++){
      if(x <= xarr[i]){
        itrue = i;
        result = (x- xarr[i-1]) * (yarr[i] - yarr[i-1])/(xarr[i] - xarr[i-1]) + yarr[i-1];
        break;
      }
    }
  }
  else{
   result = yarr[Nx-1]; 
  }
  return result;
}

void export_for_plot(string name, vector<double> & xarr, vector<double> & yarr){
  int Nx = xarr.size();
  ofstream file(name);
  for(int i = 0; i < Nx-1;i++) {
    file << xarr[i] << "\t" << yarr[i] << "\n";
  }
  file << xarr[Nx-1] << "\t" << yarr[Nx-1];
  file.close();
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

// xii is grid point int, need to multiply by deltax when computing interpolation
double linear_interp_3D(multi_array<double, 1> x_compute, multi_array<int, 2> xii, 
    multi_array<double,3> fii, double deltax){
  int n_points=xii.shape()[0]; // Number of interpolating points, which should be 2
  int dims=xii.shape()[1]; // Number of dimensions
  double denominator = 1;
  for(int i=0;i<dims; i++){
    denominator = denominator * (xii[1][i]-xii[0][i]);
  }
  // I follow wikipedia article for trilinear interpolation
  // Remember that xii are integers (grid points), multiply by deltax to recover the physical one
  double xd = (x_compute[0] - deltax*xii[0][0] )/(xii[1][0]-xii[0][0])/deltax;
  double yd = (x_compute[1] - deltax*xii[0][1] )/(xii[1][1]-xii[0][1])/deltax;
  double zd = (x_compute[2] - deltax*xii[0][2] )/(xii[1][2]-xii[0][2])/deltax;
  double c00 = fii[0][0][0]*(1-xd) +fii[1][0][0]*xd;
  double c01 = fii[0][0][1]*(1-xd) +fii[1][0][1]*xd;
  double c10 = fii[0][1][0]*(1-xd) +fii[1][1][0]*xd;
  double c11 = fii[0][1][1]*(1-xd) +fii[1][1][1]*xd;
  double c0 = c00*(1-yd) +c10*yd;
  double c1 = c01*(1-yd) + c11*yd;
  return c0*(1-zd) +c1*zd;
  // double num1= (deltax*xii[1][0] - x_compute[0])*(deltax*xii[1][1] - x_compute[1])*(deltax*xii[1][2] - x_compute[2])*fii[0][0][0];
  // double num2= (-deltax*xii[0][0] + x_compute[0])*(deltax*xii[1][1] - x_compute[1])*(deltax*xii[1][2] - x_compute[2])*fii[1][0][0];
  // double num3= (deltax*xii[1][0] - x_compute[0])*(-deltax*xii[0][1] + x_compute[1])*(deltax*xii[1][2] - x_compute[2])*fii[0][1][0];
  // double num4= (-deltax*xii[0][0] + x_compute[0])*(-deltax*xii[0][1] + x_compute[1])*(deltax*xii[1][2] - x_compute[2])*fii[1][1][0];
  // double num5= (deltax*xii[1][0] - x_compute[0])*(deltax*xii[1][1] - x_compute[1])*(-deltax*xii[0][2] + x_compute[2])*fii[0][0][1];
  // double num6= (-deltax*xii[0][0] + x_compute[0])*(deltax*xii[1][1] - x_compute[1])*(-deltax*xii[0][2] + x_compute[2])*fii[1][0][1];
  // double num7= (deltax*xii[1][0] - x_compute[0])*(-deltax*xii[0][1] + x_compute[1])*(-deltax*xii[0][2] + x_compute[2])*fii[0][1][1];
  // double num8= (-deltax*xii[0][0] + x_compute[0])*(-deltax*xii[0][1] + x_compute[1])*(-deltax*xii[0][2] + x_compute[2])*fii[1][1][1];
  // return (num1+num2+num3+num4+num5+num6+num6+num7+num8)/denominator;
}