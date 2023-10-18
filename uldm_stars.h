#ifndef ULDM_STARS_H
#define ULDM_STARS_H
#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include <cmath>
class domain_stars: public domain3
{
  public:
    multi_array<double,2> stars;
    double soften_param = 1E-5; // Softening parameter for gravitational law
    domain_stars(size_t PS,size_t PSS, double L, int nfields, int Numsteps, double DT, int Nout, int Nout_profile, 
            string Outputname, int pointsm, int WR, int WS, int Nghost, bool mpi_flag, int num_stars):
      domain3{PS, PSS, L, nfields, Numsteps, DT, Nout, Nout_profile, 
          Outputname, pointsm, WR, WS, Nghost, mpi_flag}
      {
          stars(extents[num_stars][7]);
      };
    domain_stars() { }; // Default constructor
    ~domain_stars() { };
void put_initial_stars(multi_array<double, 2> input){
  // Input should have, for each star, {mass, x,y,z,vx,vy,vz}
  stars = input;
}

vector<double> acceleration_from_point(int i, int j, int k, double x, double y, double z, double softening){
  double mass = 0;
  vector<double> acceleration(3,0);
  for(int f = 0; f <2*nfields; f++){
    mass += pow(deltaX,3)* pow(psi[f][i][j][k],2);
  }
  int keff = k + world_rank*PointsSS - nghost;
  double distance = sqrt(pow(i*deltaX - x,2) + pow(j*deltaX-y,2) + pow(keff*deltaX-z,2) + softening);
  acceleration[0] = 1/(4*M_PI) * mass * (x - i*deltaX)/ pow(distance,3);
  acceleration[1] = 1/(4*M_PI) * mass * (y - j*deltaX)/ pow(distance,3);
  acceleration[2] = 1/(4*M_PI) * mass * (z - keff*deltaX)/ pow(distance,3);
  return acceleration;
}

void step_stars(){
  for (int s = 0; s <stars.size(); s++){
    double x = stars[s][1];
    double y = stars[s][2];
    double z = stars[s][3];
    double ax = 0;
    double ay = 0;
    double az = 0;
    // Check how to meaningfully do the reduction!
    #pragma omp parallel for collapse (3) reduction (+:ax,ay,az)
    for (int i = 0; i <PointsS; i++)
      for (int j = 0; j <PointsS; j++)
        for (int k = nghost; k <PointsSS+nghost; k++){ // CHANGE FOR MPI
          vector<double> acc_from_point = acceleration_from_point(i, j, k, x, y, z, soften_param);
          ax += acc_from_point[0];
          ay += acc_from_point[1];
          az += acc_from_point[2];
        }
    double ax_shared, ay_shared, az_shared; // total summed up across all nodes
    if(mpi_bool==true){
      MPI_Allreduce(&ax, &ax_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&ay, &ay_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&az, &az_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    }
    else {
      ax_shared=ax; ay_shared=ay; az_shared=az;
    }
    #pragma omp barrier
    double vx = stars[s][4]+ax_shared*dt ;
    double vy = stars[s][5] + ay_shared*dt ;
    double vz = stars[s][6] + az_shared*dt;
    stars[s][1] = stars[s][1] + dt*(vx+stars[s][4])/2;
    stars[s][2] = stars[s][2] + dt*(vy+stars[s][5])/2;
    stars[s][3] = stars[s][3] + dt*(vz+stars[s][6])/2;
    stars[s][4] = vx;
    stars[s][5] = vy;
    stars[s][6] = vz;
  }
}
};
#endif

