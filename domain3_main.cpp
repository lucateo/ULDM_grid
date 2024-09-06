#include "uldm_mpi_2field.h"
#include <boost/multi_array/multi_array_ref.hpp>
#include <iostream>
#include <ostream>
#include <vector>

domain3::domain3(size_t PS,size_t PSS, double L, int n_fields, int Numsteps, double DT, int Nout, int Nout_profile, 
    int pointsm, int WR, int WS, int Nghost, bool mpi_flag):
  nghost(Nghost),
  mpi_bool(mpi_flag),
  nfields(n_fields),
  psi(extents[n_fields*2][PS][PS][PSS+2*Nghost]), //real and imaginary parts are stored consequently, i.e. psi[0]=real part psi1 and psi[1]=imaginary part psi1, then the same for psi2
  Phi(extents[PS][PS][PSS]),
  ratio_mass(extents[n_fields]),
  fgrid(PS,PSS,WR,WS, mpi_flag), // class for Fourier trasform, defined above
  ca(extents[8]),
  da(extents[8]),
  PointsS(PS),
  PointsSS(PSS),
  Length(L),
  dt(DT),
  numsteps(Numsteps),
  pointsmax(pointsm),
  numoutputs(Nout), //outputs animation every Nout steps
  numoutputs_profile(Nout_profile), // outputs profile radially averaged every Nout_profile steps
  outputname("out_"), // A default output name, to be changed with set_output_name function
  maxx(extents[n_fields][3]), // maxx, y,z of the n_fields fields, initialized to zero to avoid overflow
  maxdensity(extents[n_fields]), // maxdensity of the fields
  world_rank(WR),
  world_size(WS)
  {
    deltaX=Length/PointsS;
    snapshotcount=0; // snapshot count needed if Grid3D is true
    first_initial_cond = true; // To deal with append modes in initial condition info output file
    for (int i=0; i<n_fields; i++)
      ratio_mass[i]=1; // the first mass ratio is always 1, but initialize everything to one
    // stepping numbers, as defined in axionyx documentation
    ca[0]=0.39225680523878;   ca[1]=0.51004341191846;   ca[2] =-0.47105338540976; ca[3]=0.06875316825251;    ca[4]=0.06875316825251;    ca[5]=-0.47105338540976;  ca[6]=0.51004341191846;   ca[7]=0.39225680523878;
    da[0]=0.784513610477560;  da[1]=0.235573213359359;  da[2]=-1.17767998417887;  da[3]=1.3151863206839023;  da[4]=-1.17767998417887;   da[5]=0.235573213359359;  da[6]=0.784513610477560;  da[7]=0;
  }; // constructor
domain3::domain3() {};
domain3::~domain3() {};

long double domain3::psisqmean(int whichPsi){// Computes the mean |psi|^2 of field i=0,1
  long double totV=0;
  #pragma omp parallel for collapse(3) reduction(+:totV)
  for(size_t i=0;i<PointsS;i++)
    for(size_t j=0;j<PointsS;j++)
      for(size_t k=nghost;k<PointsSS+nghost;k++)
        totV=totV+pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi+1][i][j][k],2);
  #pragma omp barrier

  long double totVshared; // total summed up across all nodes
  if(mpi_bool==true){
    MPI_Allreduce(&totV, &totVshared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    totVshared=totVshared/(PointsS*PointsS*PointsS);
  }
  else {
    totVshared=totV/(PointsS*PointsS*PointsS);
  }
  return totVshared;
}
 
double domain3::total_mass(int whichPsi){ // Computes the total mass
  return psisqmean(whichPsi) * pow(Length, 3);
}

// cycs not needed in the z direction, but do no harm
// note k is counted including ghosts; you should call sort_ghost() before calling this function
double domain3::energy_kin(const int & i, const int & j, const int & k, int whichPsi){// it computes the kinetic energy density at grid point (i,j,k)
  int find=2*whichPsi; // to save typing
  double r = ratio_mass[whichPsi]; // Correct energy for field with mass ratio r wrt field 0
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
  return 0.5/(r*r)* (pow(der_psi1im,2) + pow(der_psi1re,2) + pow(der_psi2im,2) + pow(der_psi2re,2)+ pow(der_psi3im,2) + pow(der_psi3re,2));
}

// it computes the potential energy density at grid point (i,j,k)
// virtual for inheritance (external potential)
// note that Psi fields have ghost points Phi doesn't, k includes ghosts
double domain3::energy_pot(const int & i, const int & j, const int & k, int whichPsi){
  return 0.5*(pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2))*Phi[i][j][k-nghost];
}

double domain3::e_kin_full1(int whichPsi){//Full kinetic energy with Fourier
  double r = ratio_mass[whichPsi];// Correct energy for field with mass ratio r wrt field 0
  double locV= pow(Length,3)/(r*r)*fgrid.e_kin_FT(psi,Length,nghost,whichPsi);
  double totV;
  // gather all the locV for all processes, sums them (MPI_SUM) and returns totV 
  if(mpi_bool==true){
    MPI_Allreduce(&locV, &totV, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
  else {
    totV=locV;
  }
  return totV;
}

//Finds velocity of the center of mass of field whichPsi. The total velocity of center of mass can be found by 
// finding the center of mass velocity of center of mass velocities in post-processing.
double domain3::v_center_mass(int coordinate, int whichF){
  double mass=total_mass(whichF); 
  double locV= fgrid.v_center_mass_FT(psi,Length,nghost,whichF, coordinate) / (mass * ratio_mass[whichF]);
  double totV;
  // gather all the locV for all processes, sums them (MPI_SUM) and returns totV 
  if(mpi_bool==true){
    MPI_Allreduce(&locV, &totV, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
  else {
    totV=locV;
  }
  return totV;
}
//Finds position of the center of mass of 1 field. The total center of mass can be found by 
// finding the center of mass of center of masses.
double domain3::x_center_mass(int coordinate, int whichPsi){
  double mass=total_mass(whichPsi); 
  long double totV=0;
  #pragma omp parallel for collapse(3) reduction(+:totV)
  for(size_t i=0;i<PointsS;i++)
    for(size_t j=0;j<PointsS;j++)
      for(size_t k=nghost;k<PointsSS+nghost;k++){
        size_t real_k = k-nghost + PointsSS*world_rank;
        // It computes it around the maximum of density, to hopefully mitigate the periodic boundary problem
        if(coordinate==0) {
          int Dd=maxx[whichPsi][0]-(int)i; if(abs(Dd)>PointsS/2){Dd=abs(Dd)-(int)PointsS;} 
          totV=totV+(pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi+1][i][j][k],2)) * Dd *Length/PointsS;
        }
        else if(coordinate==1){ 
          int Dd=maxx[whichPsi][1]-(int)j; if(abs(Dd)>PointsS/2){Dd=abs(Dd)-(int)PointsS;} 
          totV=totV+(pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi+1][i][j][k],2)) * Dd *Length/PointsS;
        }
        else if(coordinate==2) {
          int Dd=maxx[whichPsi][2]-(int)real_k; if(abs(Dd)>PointsS/2){Dd=abs(Dd)-(int)PointsS;} 
          totV=totV+(pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi+1][i][j][k],2)) * Dd *Length/PointsS;
        }
      }
  #pragma omp barrier
  long double totVshared; // total summed up across all nodes
  if(mpi_bool==true){
    MPI_Allreduce(&totV, &totVshared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    totVshared=totVshared*pow(Length/PointsS,3) / mass;
  }
  else {
    totVshared=totV*pow(Length/PointsS,3) / mass;
  }
  // Add the maximum point to recover the actual point coordinates of the grid
  return totVshared + maxx[whichPsi][coordinate];
}

long double domain3::full_energy_kin(int whichPsi){// it computes the kinetic energy of the whole box with derivatives
  long double total_energy = 0;
  #pragma omp parallel for collapse (3) reduction(+:total_energy)
  for(int i=0;i<PointsS;i++)
    for(int j=0; j<PointsS;j++)
      for(int k=nghost; k<PointsSS+nghost;k++){
        total_energy = total_energy + energy_kin(i,j,k,whichPsi);
      }
  #pragma omp barrier
  long double total_energy_shared; // total summed up across all nodes
  if(mpi_bool==true){
    MPI_Allreduce(&total_energy, &total_energy_shared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
  else {
    total_energy_shared=total_energy;
  }
  return total_energy_shared*pow(Length/PointsS,3);
}

long double domain3::full_energy_pot(int whichPsi){// it computes the potential energy of the whole box
  long double total_energy = 0;
  #pragma omp parallel for collapse (3) reduction(+:total_energy)
  for(int i=0;i<PointsS;i++)
    for(int j=0; j<PointsS;j++)
      for(int k=nghost; k<PointsSS+nghost;k++){
        total_energy = total_energy + energy_pot(i,j,k,whichPsi);
      }
  #pragma omp barrier
  long double total_energy_shared; // total summed up across all nodes
  if(mpi_bool==true){
    MPI_Allreduce(&total_energy, &total_energy_shared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  }
  else {
    total_energy_shared=total_energy;
  }
  return total_energy_shared*pow(Length/PointsS,3);
}


double domain3::find_maximum(int whichPsi){ // Sets maxx, maxy, maxz equal to the maximum, it just checks for one global maximum
  maxdensity[whichPsi] = 0;
  #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++)
    for(int j=0; j<PointsS;j++)
      for(int k=nghost; k<PointsSS+nghost;k++){
        double density_current = pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2);
        if (density_current > maxdensity[whichPsi]){// convention is that maxz does not count ghost; this is the true maxz of the full grid
          maxx[whichPsi][0]=i; maxx[whichPsi][1]=j; maxx[whichPsi][2]=k+world_rank*PointsSS-nghost; 
          maxdensity[whichPsi] =density_current;
        }  
      }
    #pragma omp barrier
  // now compare across nodes (there's probably a better way to do this, but it's ok for now)
  maxNode=0;
  int maxx_local = maxx[whichPsi][0];
  int maxy_local = maxx[whichPsi][1];
  int maxz_local = maxx[whichPsi][2];
  double maxdensity_local = maxdensity[whichPsi];
  if(world_rank!=0 && mpi_bool==true){  // collect onto node 0
    vector<float> sendup {(float) 1.0000001*maxx_local,(float)1.00000001* maxy_local,
      (float)1.000001*maxz_local,(float) maxdensity_local };
    MPI_Send(&sendup.front(), sendup.size(), MPI_FLOAT, 0, 9090, MPI_COMM_WORLD);
  }
  if(world_rank==0 && mpi_bool==true){  // send to every other node
      for(int lpb=1;lpb<world_size;lpb++){ // recieve from every other node
          vector<float> reci(4,0); //vector to recieve data into
          MPI_Recv(&reci.front(),reci.size(), MPI_FLOAT, lpb, 9090, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          if(reci[3]> maxdensity_local){
            maxdensity_local=reci[3]; maxx_local=reci[0];  
            maxy_local=reci[1];  maxz_local=reci[2];maxNode=lpb; 
          }
      }
  }

  // now correct on node 0, but now we need to tell all the nodes
  if(mpi_bool==true){
    MPI_Bcast(&maxNode, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxx_local, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxy_local, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxz_local, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxdensity_local, 1, MPI_DOUBLE,0, MPI_COMM_WORLD);
  }
  maxx[whichPsi][0]=maxx_local; maxx[whichPsi][1]=maxy_local; maxx[whichPsi][2]=maxz_local;
  maxdensity[whichPsi]=maxdensity_local; 
  return maxdensity[whichPsi];
}

void domain3::sortGhosts(){
    for(int ii=0;ii<nfields*2;ii++){
      transferghosts(psi,ii, world_rank, world_size, nghost);
    }
  }
void domain3::expiPhi(double tstep, double da, int whichPsi){
  // Insert the ratio of mass for the second field
  double r = ratio_mass[whichPsi];
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

void domain3::set_grid(bool grid_bool){//If false, domain3 outputs only the 2D sliced density profile
  Grid3D = grid_bool;
}
void domain3::set_grid_phase(bool bool_phase){//If false, domain3 does not output the phase slice
  phaseGrid = bool_phase;
}
void domain3::set_backup_flag(bool bool_backup){//If false, no backup
  start_from_backup = bool_backup;
}
void domain3::set_ratio_masses(multi_array<double,1> ratio_masses){//
  ratio_mass = ratio_masses;
}
void domain3::set_adaptive_dt_flag(bool flag_adaptive){//
  adaptive_timestep = flag_adaptive;
}
void domain3::set_reduce_grid(int reduce_grid){//
  reduce_grid_param = reduce_grid;
}
void domain3::set_spectrum_flag(bool spect){//
  spectrum_bool = spect;
  // To avoid memory consumptions if spectrum output is not required, initialize psi_backup only now
  if (spect == true) {
    psi_backup.resize(extents[nfields*2][PointsS][PointsS][PointsSS+2*nghost]);  
  }
}
void domain3::set_output_name(string name){//
  outputname = name;
}

// Sets the number of steps
void domain3::set_numsteps(int Numsteps){//
  numsteps = Numsteps;
}

void domain3::makestep(double stepCurrent, double tstep){ // makes a step in a dt
  // loop over the 8 values of alpha
  for(int alpha=0;alpha<8;alpha++){
    //1: For any fixed value of alpha, perform the operation exp(-i c_\alpha dt k^2/2)\psi(t) in Fourier space
    for(int whichF=0;whichF<nfields;whichF++){
      fgrid.inputpsi(psi,nghost,whichF);
      fgrid.calculateFT();                                                  //calculates the FT
      fgrid.kfactorpsi(tstep,Length,ca[alpha],ratio_mass[whichF]);                             //multiples the psi by exp(-i c_\alpha dt k^2/2)
      fgrid.calculateIFT();                                                 //calculates the inverse FT
      fgrid.transferpsi(psi,1./pow(PointsS,3),nghost,whichF);                             //divides it by 1/PS^3 (to get the correct normalizaiton of the FT)
    }
    //2: Then perform the operation exp(-i d_\alpha dt \Phi(x)) applied to the previous output
    if(alpha!=7){  //da[7]=0 so no evolution here anyway
      //first, calculate Phi(x) by solving nabla^2Phi = |psi^2|-<|psi^2|> in Fourier space
//                    double psisqavg= psisqmean();                   //calculates the number |psi^2| (it should be conserved)
      fgrid.inputPhi(psi,nghost,nfields);                                     //inputs |psi^2|
      fgrid.calculateFT();                                              //calculates its FT, FT(|psi^2|)
      fgrid.kfactorPhi(Length);                                         //calculates -1/k^2 FT(|psi^2|)
      fgrid.calculateIFT();                                             //calculates the inverse FT
      fgrid.transferPhi(Phi,1./pow(PointsS,3));                     //transfers the result into the xytzgrid Phi and multiplies it by 1/PS^3
      //second, multiply the previous output of psi by exp(-i d_\alpha dt \Phi(x)+Phi_ext), in coordinate space
      for(int i=0;i<nfields;i++){
        expiPhi(dt, da[alpha],i);
      }
    }
  }
  // multi_array<long double, 1> psismean(extents[nfields]);
  // for(int i=0;i<nfields;i++)
  //   psismean[i]=psisqmean(i);
  // if(world_rank==0){ 
  //   cout<<"mean value of fields ";
  //   for(int i=0;i<nfields;i++){
  //     cout<<i<<" "<<psismean[i]<<", ";
  //   }
  // cout<<endl;
  // }
}

void domain3::solveConvDif(){
  int beginning=time(NULL);
  int count_energy=0; // count how many times you decrease the time step before changing to E_tot_running
  int switch_en_count = 0; // Keeps track on how many times you switch energy
  if (start_from_backup == true)
    openfiles_backup();
  else
    openfiles();
  int stepCurrent=0;
  double t_spectrum;
  vector<vector<double>> spectrum_vector;
  if (world_rank==0) cout<< "File name of the run " << outputname << endl;
  if (start_from_backup == false){
    tcurrent = 0;
    exportValues(); // for backup purposes
    ofstream psi_final;
    outputfullPsi(psi_final,true,1);
    snapshot(stepCurrent); // I want the snapshot of the initial conditions
    snapshot_profile(stepCurrent); 
    t_spectrum = tcurrent; // The time at which computes the energy spectrum
    if(spectrum_bool == true){
      #pragma omp parallel for collapse(4)
      for(int l = 0; l < 2*nfields; l++)
        for(size_t i=0;i<PointsS;i++)
          for(size_t j=0; j<PointsS;j++)
            for(size_t k=nghost; k<PointsSS+nghost;k++){
              // psi_backup has no ghosts
              psi_backup[l][i][j][k-nghost]= psi[l][i][j][k];
            }
      #pragma omp barrier
      spectrum_output(spectrum_vector,stepCurrent, t_spectrum, tcurrent);
    }
    // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
    // (Phi is not computed yet), so store the initial energy after one step
    if(world_rank==0){
      cout<<"current time = "<< tcurrent << " step " << stepCurrent << " / " << numsteps<<" dt "<<dt<<endl;
      cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
    }
    makestep(stepCurrent,dt);
    tcurrent=tcurrent+dt;
    stepCurrent=stepCurrent+1;
    
    if(spectrum_bool == true)
      spectrum_output(spectrum_vector,stepCurrent, t_spectrum, tcurrent);
    
    if (mpi_bool==true){ 
      sortGhosts(); // Should be called, to do derivatives in real space
    }
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
    t_spectrum = tcurrent; // The time at which computes the energy spectrum
    if(spectrum_bool == true){
      #pragma omp parallel for collapse(4)
      for(int l = 0; l < 2*nfields; l++)
        for(size_t i=0;i<PointsS;i++)
          for(size_t j=0; j<PointsS;j++)
            for(size_t k=nghost; k<PointsSS+nghost;k++){
              // psi_backup has no ghosts
              psi_backup[l][i][j][k-nghost]= psi[l][i][j][k];
            }
      #pragma omp barrier
      spectrum_output(spectrum_vector,stepCurrent, t_spectrum, tcurrent);
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
    stepCurrent=stepCurrent+1;
    if (spectrum_bool==true){
      spectrum_output(spectrum_vector,stepCurrent, t_spectrum, tcurrent);
    }
    double etot_current = 0;
    for(int i=0;i<nfields;i++){
      etot_current += e_kin_full1(i) + full_energy_pot(i);
    }
    if(adaptive_timestep==true){
      double compare_energy = abs(etot_current-E_tot_initial)/abs(etot_current + E_tot_initial);
      double compare_energy_running = abs(etot_current-E_tot_running)/abs(etot_current + E_tot_running);
      if (world_rank==0) cout<<"E tot current "<<etot_current << " E tot initial " << E_tot_initial << " compare E ratio "<<compare_energy <<endl;
      // Criterium for dropping by half the time step if energy is not conserved well enough
      if(compare_energy > 0.001 ){
        dt = dt/2;
        count_energy++;
        if (compare_energy_running < 1E-6 && count_energy > 2 + 2*switch_en_count){
          E_tot_initial = E_tot_running;
          count_energy = 0;
          if (world_rank==0) cout<<"Switch energy "<<switch_en_count <<" --------------------------------------------------------------------------------------------" <<endl;
          switch_en_count++;
        }
      }
      else if(compare_energy<1E-6 && compare_energy_running>1E-9){
        dt = dt*1.2; // Less aggressive when increasing the time step, rather than when decreasing it
      }
      else if (compare_energy_running < 1E-9) {
        dt = dt*1.5; // If it remains stuck to an incredibly low dt, try to unstuck it
      if (world_rank==0)
        cout<<"Unstucking, count energy " << count_energy << " switch energy count " << switch_en_count 
          << " --------------------------------------------------------------------------------------------------------------------"<<endl;
      }
      E_tot_running = etot_current;
    }
    if(stepCurrent%numoutputs==0 || stepCurrent==numsteps) {
      if (mpi_bool==true){ 
        sortGhosts(); // Should be called, to do derivatives in real space
      }
      snapshot(stepCurrent);
      if(spectrum_bool == true ) { 
        t_spectrum = tcurrent; // Changes the time at which spectrum is computed 
        #pragma omp parallel for collapse(4)
        for(int l = 0; l < 2*nfields; l++)
          for(size_t i=0;i<PointsS;i++)
            for(size_t j=0; j<PointsS;j++)
              for(size_t k=nghost; k<PointsSS+nghost;k++){
                // psi_backup has no ghosts
                psi_backup[l][i][j][k-nghost]= psi[l][i][j][k];
              }
        #pragma omp barrier
        // I want the integral also at time difference zero
        spectrum_write(spectrum_vector);
        spectrum_vector.clear();
        spectrum_output(spectrum_vector, stepCurrent, t_spectrum, tcurrent);
      }
    }
    if(stepCurrent%numoutputs_profile==0 || stepCurrent==numsteps) {
      if (mpi_bool==true){ 
        sortGhosts(); // Should be called, to do derivatives in real space
      }
      snapshot_profile(stepCurrent);
      exportValues(); // for backup purposes
      ofstream psi_final;
      outputfullPsi(psi_final,true,1);
    }
  }
  closefiles();
  cout<<"end"<<endl;
}


void domain3::initial_cond_from_backup(){
  // multi_array<double,1> Arr1D(extents[PointsS*PointsS*PointsSS]);
  // string outputname_file;
  // if(mpi_bool==true){
  //   outputname_file = outputname + "phi_final_" + to_string(world_rank) + ".txt";
  // }
  // else {
  //   outputname_file = outputname + "phi_final.txt";
  // }
  // ifstream infile(outputname_file);
  // string temp;
  // double num;
  // // size_t l = 0;
  // size_t i = 0; size_t j = 0; size_t k = 0;
  // // Get the input from the input file
  // while (std::getline(infile, temp, ' ')) {
  // // Add to the list of output strings
  //   if(i<PointsS){
  //     Phi[i][j][k] = stod(temp);
  //     i++;
  //   }
  //   if(i==PointsS){
  //     if(j<PointsS-1) {
  //       i=0;
  //       j++;
  //     }
  //     else if(j==PointsS-1 && k<PointsSS-1){
  //       i=0; j=0; k++;
  //     }
  //   }
  // }
  // for(size_t i =0; i<PointsS; i++)
  //   for(size_t j =0; j<PointsS; j++)
  //     for(size_t k =0; k<PointsSS; k++){
  //       // cout<<i<<endl;
  //       Phi[i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS];
  //     }
  // infile.close();
  string outputname_file;
  if(mpi_bool==true){
    outputname_file = outputname + "psi_final_" + to_string(world_rank) + ".txt";
  }
  else {
    outputname_file = outputname + "psi_final.txt";
  }
  ifstream infile(outputname_file);
  size_t i = 0; size_t j = 0; size_t k = 0;
  size_t m =0;
  string temp;
  while (std::getline(infile, temp, ' ')) {
  // Add to the list of output strings
    if(i<PointsS){
      psi[m][i][j][k+nghost] = stod(temp);
      i++;
    }
    if(i==PointsS){
      if(j<PointsS-1) {
        i=0;
        j++;
      }
      else if(j==PointsS-1 && k<PointsSS-1){
        i=0; j=0; k++;
      }
      else if(k==PointsSS-1 && j==PointsS-1 ){
        i=0; j=0; k=0; m++;
      }
    }
  }
  // l = 0;
  // multi_array<double,1> Arr1Dpsi(extents[PointsS*PointsS*PointsSS*nfields*2]);
  // while (std::getline(infile, temp, ' ')) {
  //   // Add to the list of output strings
  //   num = stod(temp);
  //   Arr1Dpsi[l] = num;
  //   l++;
  // }
  // for(size_t m =0; m<nfields*2; m++)
  //   for(size_t i =0; i<PointsS; i++)
  //     for(size_t j =0; j<PointsS; j++)
  //       for(size_t k =0; k<PointsSS; k++){
  //         psi[m][i][j][k+nghost] = Arr1Dpsi[i+j*PointsS+k*PointsS*PointsS+m*PointsS*PointsS*PointsSS];
  //       }
  infile.close();
}