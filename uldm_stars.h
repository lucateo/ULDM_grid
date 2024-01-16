#ifndef ULDM_STARS_H
#define ULDM_STARS_H
#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include <cmath>
#include <fstream>
// In this version, stars do not have feedback on the background potential and they do not feel themselves
// (so specify their mass is actually not required)
class domain_stars: public domain3
{
  protected:
    double soften_param = deltaX/10.; // Softening parameter for gravitational law
    // double soften_param = deltaX/2; // Maybe better putting something related to deltaX?
    multi_array<double,2> stars;
    ofstream stars_filename;
  public:
    domain_stars(size_t PS,size_t PSS, double L, int nfields, int Numsteps, double DT, int Nout, int Nout_profile, 
            string Outputname, int pointsm, int WR, int WS, int Nghost, bool mpi_flag, int num_stars):
      domain3{PS, PSS, L, nfields, Numsteps, DT, Nout, Nout_profile, 
          Outputname, pointsm, WR, WS, Nghost, mpi_flag} , stars(extents[num_stars][7])
      { };
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
    // Leapfrog second order, ADAPTIVE STEP NOT IMPLEMENTED
    double x = stars[s][1]+ 0.5*dt*stars[s][4];
    double y = stars[s][2]+ 0.5*dt*stars[s][5];
    double z = stars[s][3]+ 0.5*dt*stars[s][6];
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
    #pragma omp barrier
    double ax_shared, ay_shared, az_shared; // total summed up across all nodes
    if(mpi_bool==true){
      MPI_Allreduce(&ax, &ax_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&ay, &ay_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&az, &az_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    }
    else {
      ax_shared=ax; ay_shared=ay; az_shared=az;
    }
    stars[s][4] = stars[s][4] -ax_shared*dt ;
    stars[s][5] = stars[s][5] - ay_shared*dt ;
    stars[s][6] = stars[s][6] - az_shared*dt ;
    stars[s][1] = stars[s][1] + dt*stars[s][4]/2;
    stars[s][2] = stars[s][2] + dt*stars[s][5]/2;
    stars[s][3] = stars[s][3] + dt*stars[s][6]/2;
  }
}

void output_stars(){
  print2(stars, stars_filename);
  if(world_rank==0) stars_filename<<"\n"<<","<<flush;
}

void out_star_backup(){
    ofstream file_star_backup;
    file_star_backup.open(outputname+"star_baxkup.txt"); 
    file_star_backup.setf(ios_base::fixed);
    int Nx=stars.shape()[0];
    int Ny=stars.shape()[1];
    for(int s = 0;s < Nx; s++){
      for(int l = 0;l < Ny; l++){
        file_star_backup<< scientific << stars[s][l];
        if(s!=(Nx-1) || l!=(Nx-1)) //If it is not the last one
          file_star_backup<< " ";
      }
    }
    file_star_backup.close();
}

void open_filestars(){
  if(world_rank==0 && start_from_backup == false){
    stars_filename.open(outputname+"stars.txt");
    stars_filename<<"{";   
    stars_filename.setf(ios_base::fixed);
  }
  else if (world_rank==0 && start_from_backup == false){
    stars_filename.open(outputname+"stars.txt", ios_base::app); 
    stars_filename.setf(ios_base::fixed);
  }
}

// Insert output functions!
virtual void solveConvDif(){
  int beginning=time(NULL);
  int count_energy=0; // count how many times you decrease the time step before changing to E_tot_running
  int switch_en_count = 0; // Keeps track on how many times you switch energy
  if (start_from_backup == true)
    openfiles_backup();
  else
    openfiles();
  open_filestars();

  int stepCurrent=0;
  if (start_from_backup == false){
    tcurrent = 0;
    snapshot(stepCurrent); // I want the snapshot of the initial conditions
    snapshot_profile(stepCurrent);
    output_stars(); 
    // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
    // (Phi is not computed yet), so store the initial energy after one step
    if(world_rank==0){
      cout<<"current time = "<< tcurrent << " step " << stepCurrent << " / " << numsteps<<" dt "<<dt<<endl;
      cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
    }
    makestep(stepCurrent,dt);
    tcurrent=tcurrent+dt;
    step_stars();
    stepCurrent=stepCurrent+1;
  }
  else if (start_from_backup == true){
    ifstream infile(outputname+"runinfo.txt");
    string temp;
    size_t i = 0;
    while (std::getline(infile, temp, ' ') && i<6){ // convoluted way to read just the character I need
      if(i==0)
        tcurrent = stod(temp);
      // else if(i==5)
      //   dt=stod(temp);
      i++;
    }
  }
  E_tot_initial = 0; // The very initial energy
  double E_tot_running = 0; // The total energy when the time step decreases
  for(int i=0;i<nfields;i++){
    E_tot_initial += e_kin_full1(i) + full_energy_pot(i);
  }

  while(stepCurrent<numsteps){
    if(world_rank==0){
      cout<<"current time = "<< tcurrent  << " step " << stepCurrent << " / " << numsteps<<" dt "<<dt<<endl;
      cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
    }
    makestep(stepCurrent,dt);
    tcurrent=tcurrent+dt;
    step_stars();
    stepCurrent=stepCurrent+1;
    double etot_current = 0;
    for(int i=0;i<nfields;i++){
      etot_current += e_kin_full1(i) + full_energy_pot(i);
    }
    double compare_energy = abs(etot_current-E_tot_initial)/abs(etot_current + E_tot_initial);
    double compare_energy_running = abs(etot_current-E_tot_running)/abs(etot_current + E_tot_running);
    if (world_rank==0) cout<<"E tot current "<<etot_current << " E tot initial " << E_tot_initial << " compare E ratio "<<compare_energy <<endl;
    // Criterium for dropping by half the time step if energy is not conserved well enough
    if(compare_energy > 0.001 ){
      dt = dt/2;
      count_energy++;
      if (compare_energy_running < 1E-5 && count_energy > 2 + switch_en_count){
        E_tot_initial = E_tot_running;
        count_energy = 0;
        if (world_rank==0) cout<<"Switch energy "<<switch_en_count <<" --------------------------------------------------------------------------------------------" <<endl;
        switch_en_count++;
      }
    }
    else if(compare_energy<1E-5 && compare_energy_running>1E-8){
      dt = dt*1.2; // Less aggressive when increasing the time step, rather than when decreasing it
    }
    else if (compare_energy_running < 1E-8) {
      dt = dt*2; // If it remains stuck to an incredibly low dt, try to unstuck it
      cout<<"Unstucking --------------------------------------------------------------------------------------------------------------------"<<endl;
    }
    E_tot_running = etot_current;
    
    if(stepCurrent%numoutputs==0 || stepCurrent==numsteps) {
      if (mpi_bool==true){ 
        sortGhosts(); // Should be called, to do derivatives in real space
      }
      snapshot(stepCurrent);
      if (world_rank==0){
        output_stars();
        out_star_backup();
      } 
      exportValues(); // for backup purposes
      ofstream phi_final;
      outputfullPhi(phi_final);
      ofstream psi_final;
      outputfullPsi(psi_final);
    }
    if(stepCurrent%numoutputs_profile==0 || stepCurrent==numsteps) {
      if (mpi_bool==true){ 
        sortGhosts(); // Should be called, to do derivatives in real space
      }
      snapshot_profile(stepCurrent);
    }
  }
  closefiles();
  stars_filename.close();
  cout<<"end"<<endl;
}
};
#endif

