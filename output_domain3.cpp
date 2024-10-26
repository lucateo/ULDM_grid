#include "uldm_mpi_2field.h"
#include <ios>
#include <string>
#include <vector>

using namespace std;
using namespace boost;

void domain3::openfiles(){ // It opens all the output files, and it inserts an initial {
  if(world_rank==0){
    timesfile_grid.open(outputname+"times_grid.txt");       timesfile_grid<<"{";   timesfile_grid.setf(ios_base::fixed);
    timesfile_profile.open(outputname+"times_profile.txt");       timesfile_profile<<"{";   timesfile_profile.setf(ios_base::fixed);
    profilefile.open(outputname+"profiles.txt");  profilefile<<"{"; profilefile.setf(ios_base::fixed);
    phase_slice.open(outputname+"phase_slice.txt");  phase_slice<<"{"; phase_slice.setf(ios_base::fixed);
    profile_sliced.open(outputname+"profile_sliced.txt");profile_sliced<<"{"; profile_sliced.setf(ios_base::fixed);
    if (spectrum_bool == true){
      spectrum_energy.open(outputname+"times_spectrum.txt");       
      spectrum_energy<<"{";   
      spectrum_energy.setf(ios_base::fixed);
    }   
  }
}

void domain3::openfiles_backup(){ //It opens all the output files in append mode, with no initial insertion of { (for backup mode)
  if(world_rank==0){  
    timesfile_grid.open(outputname+"times_grid.txt", ios_base::app); timesfile_grid.setf(ios_base::fixed);
    timesfile_profile.open(outputname+"times_profile.txt", ios_base::app);  timesfile_profile.setf(ios_base::fixed);
    profilefile.open(outputname+"profiles.txt", ios_base::app); profilefile.setf(ios_base::fixed);
    phase_slice.open(outputname+"phase_slice.txt", ios_base::app);   phase_slice.setf(ios_base::fixed);
    profile_sliced.open(outputname+"profile_sliced.txt", ios_base::app);profile_sliced.setf(ios_base::fixed);
    if (spectrum_bool == true){
      spectrum_energy.open(outputname+"times_spectrum.txt",ios_base::app);       
      spectrum_energy.setf(ios_base::fixed);
    }   
  }
}

// Considering I am implementing the possbility to use backups, this function closes files
// WITHOUT putting a final }; if one wants to read these arrays with a program (Mathematica etc.),
// one should load those arrays and put a final }
void domain3::closefiles(){
  if(world_rank==0){
    timesfile_grid.close();
    timesfile_profile.close();
    profilefile.close();
    profile_sliced.close();
    phase_slice.close();
    if (spectrum_bool==true){
      spectrum_energy.close();
    }
  }
}

// It stores run output info
void domain3::exportValues(){
  if(world_rank==0){
    runinfo.open(outputname+"runinfo.txt");
    runinfo.setf(ios_base::fixed);
    runinfo<< scientific<<tcurrent<<" "<<E_tot_initial<<" "<< Length<<" ";
    runinfo<< fixed << numsteps<<" "<<PointsS<<" ";
    runinfo<<fixed<<dt<<" " <<numoutputs<<" "<<numoutputs_profile;
    for(int i=0; i<nfields; i++)
    {
      runinfo<<" "<< ratio_mass[i];
    }
    runinfo<<endl;
    runinfo<<"\n";
    runinfo.close();
  }
}


void domain3::outputfullPhi(ofstream& fileout){// Outputs the full 3D Phi potential, for backup purposes
  if(mpi_bool==true){//if mpi_bool==true, append the world_rank to the file name
    fileout.open(outputname+"phi_final_"+to_string(world_rank)+".txt"); fileout.setf(ios_base::fixed);
  }
  else {
    fileout.open(outputname+"phi_final.txt"); fileout.setf(ios_base::fixed);
  }
  print3_cpp(Phi,fileout);
  fileout.close();
}


// if Grid3D is true, set bool backup to false; reduce_grid reduces the output resolution
void domain3::outputfullPsi(ofstream& fileout, bool backup, int reduce_grid){// Outputs the full 3D psi, every field, for backup purposes
  if (backup==true){
    if(mpi_bool==true){//if mpi_bool==true, append the world_rank to the file name
      fileout.open(outputname+"psi_final_"+to_string(world_rank)+".txt"); fileout.setf(ios_base::fixed);
    }
    else {
      fileout.open(outputname+"psi_final.txt"); fileout.setf(ios_base::fixed);
    }
  }
  else if (backup==false){
    if(mpi_bool==true){//if mpi_bool==true, append the world_rank to the file name
      fileout.open(outputname+"psi_snapshot_"+to_string(snapshotcount)+"_wr_"+to_string(world_rank)+".txt"); 
      fileout.setf(ios_base::fixed);
      snapshotcount++;
    }
    else {
      fileout.open(outputname+"psi_snapshot_"+to_string(snapshotcount)+".txt"); 
      fileout.setf(ios_base::fixed);
      snapshotcount++;
    }
  }
  print4_cpp(psi,fileout,nghost, reduce_grid);
  fileout.close();
}

// Returns the maximum coordinate coord value of the field whichPsi
double domain3::get_maxx(int whichPsi, int coord){
  return maxx[whichPsi][coord];
}


void domain3::outputSlicedDensity(ofstream& fileout){ // Outputs the projected 2D density profile
  // multi_array<double,3> density_sliced(extents[nfields][PointsS][PointsS]);
  vector<double> density_sliced(nfields*PointsS*PointsS,0);
  #pragma omp parallel for collapse(3)
  for(int l = 0; l < nfields; l++)
    for(size_t i=0;i<PointsS;i++)
      for(size_t j=0; j<PointsS;j++)
        for(size_t k=nghost; k<PointsSS+nghost;k++){
        density_sliced[l+i*nfields+j*nfields*PointsS]= density_sliced[l+i*nfields+j*nfields*PointsS] 
                                                     + pow(psi[2*l][i][j][k],2)+pow(psi[2*l+1][i][j][k],2);
        }
  #pragma omp barrier

  // send to node 0, 4th entry should be the node you send to, 5th entry is a tag (to be shared between receiver and transmitter)
  if(world_rank!=0 && mpi_bool==true){
    MPI_Send(&density_sliced.front(), density_sliced.size(), MPI_DOUBLE, 0, 500, MPI_COMM_WORLD);
  }
  // receive from other nodes, 4th entry should be the node you receive from, 5th entry is a tag (to be shared between receiver and transmitter)
  if(world_rank==0 && mpi_bool==true){
    vector<double> density_sliced_rec(nfields*PointsS*PointsS,0); // The receiving vector
    for(int i=1;i<world_size;i++){
      MPI_Recv(&density_sliced_rec.front(),density_sliced_rec.size(), MPI_DOUBLE, i, 500, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      transform(density_sliced.begin(), density_sliced.end(), density_sliced_rec.begin(), density_sliced.begin(), std::plus<double>());
    }
  }
  if(world_rank==0){
    vector<int> dims = {nfields,int(PointsS),int(PointsS)};
    print3_1dvector(density_sliced, dims, fileout);
  }
}

void domain3::outputPhaseSlice(ofstream& fileout){ // Outputs a 2D slice of the phase
  multi_array<double,3> phase_sliced(extents[nfields][PointsS][PointsS]);
  #pragma omp parallel for collapse(3)
  for(int l = 0; l < nfields; l++)
    for(size_t i=0;i<PointsS;i++)
      for(size_t j=0; j<PointsS;j++){
        phase_sliced[l][i][j]= atan2(psi[2*l+1][i][j][int(PointsSS/2)+nghost], psi[2*l][i][j][int(PointsSS/2)+nghost]);
      }
  #pragma omp barrier
  print3(phase_sliced,fileout);
}


// Virtual because for the NFW case (for example) you want to compute radial functions starting from the center of the box and not the maximum
multi_array<double,2> domain3::profile_density(int whichPsi){ // It computes the averaged density and energy as function of distance from soliton
  //we need to specify what is the maximum number of points we want to calculate the profile from the center {xmax,ymax,zmax}
  //You have to call find_maximum() first in order to set the maximum values to the class
  // note: results only make sense on node 0
  // vector is easier for the MPI to handle
  //vector that cointains the binned density profile {d, rho(d), E_kin(d), E_pot(d), Phi(d), phase(d)}
  vector<vector<double>> binned(6,vector<double>(pointsmax, 0));// Initialize vector of vectors, all with zero entries

  //auxiliary vector to count the number of points in each bin, needed for average
  vector<int> count(pointsmax, 0); // Initialize vector of dimension pointsmax, with 0s
  // maxz does not have ghost cells
  int extrak = PointsSS*world_rank -nghost;
  // #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++)
    for(int j=0; j<PointsS;j++)
      for(int k=nghost; k<PointsSS+nghost;k++){
        int Dx=maxx[whichPsi][0]-(int)i; if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        int Dy=maxx[whichPsi][1]-(int)j; if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // periodic boundary conditions!
        int Dz=maxx[whichPsi][2]-(int)k-extrak; if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // periodic boundary conditions!
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
  // #pragma omp barrier
  // For the phase, I take only one ray
  // Only do this on the node that contains the maximum point
  if(world_rank==maxNode || mpi_bool==false){
    int maxz_node = maxx[whichPsi][2] - PointsSS*world_rank +nghost;//maxz in max node with ghost cell
    for(int i=0;i<PointsS;i++){
      int distance =maxx[whichPsi][0]-(int)i; if(abs(distance)>PointsS/2){distance=abs(distance)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
      distance = abs(distance);
      if(distance<pointsmax){
          //Takes only one ray passing from the center of the possible soliton
          binned[5][distance] = atan2(psi[2*whichPsi+1][i][maxx[whichPsi][1]][maxz_node], 
                                      psi[2*whichPsi][i][maxx[whichPsi][2]][maxz_node]);
      }
    }
  }

  // collect onto node 0
  if(world_rank!=0 && mpi_bool==true){
    MPI_Send(&count.front(), count.size(), MPI_INT, 0, 301, MPI_COMM_WORLD);
    for(int lp=0;lp<6;lp++){
      MPI_Send(&binned[lp].front(), binned[lp].size(), MPI_DOUBLE, 0, 300+lp, MPI_COMM_WORLD);
    }
  }

  if(world_rank==0 && mpi_bool==true){
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
  #pragma omp barrier

  // convert back to a multiarray to return
  multi_array<double,2> binnedR(extents[6][pointsmax]);
  #pragma omp parallel for collapse (2)
  for(int ii=0;ii<6;ii++){
    for(int jj=0;jj<pointsmax;jj++){
      binnedR[ii][jj]=binned[ii][jj];
    }
  }
  #pragma omp barrier
  return binnedR;
}

void domain3::snapshot(double stepCurrent){//Outputs the full density profile; if 3dgrid is false, it outputs just sliced density
  cout.setf(ios_base::fixed);
  if(world_rank==0)
    timesfile_grid<<"{"<<scientific<<tcurrent;
      // if(stepCurrent<numsteps){
  double Etot =0;
  for(int l=0;l<nfields;l++){
    double Ek_full = e_kin_full1(l);
    double Epot_full = full_energy_pot(l);
    double M_tot = total_mass(l);
    Etot = Etot + Ek_full + Epot_full;
    if(world_rank==0){
      timesfile_grid<<","<<scientific<<maxdensity[l]<<","; 
      timesfile_grid<<fixed<<maxx[l][0]<<"," <<maxx[l][1]<<","<<maxx[l][2]<<",";
      timesfile_grid<<scientific<<Ek_full<<","<<Epot_full<<","<<M_tot;
      // if(stepCurrent<numsteps){
    }
  }
  if(world_rank==0){
    timesfile_grid<<","<<scientific<<Etot<<"}\n"<<","<<flush;
    cout<<"Output animation results"<<endl;
    cout.setf(ios_base::fixed);
  }
  if(Grid3D==false){
    outputSlicedDensity(profile_sliced); // Output of both fields, this has to run on all nodes!!!!
    if(world_rank==0) profile_sliced<<"\n"<<","<<flush;
  }
  // } //if it's not the last timeshot put a comma { , , , , }
  else if(Grid3D == true){
    ofstream grid;
    outputfullPsi(grid, false,reduce_grid_param);
  }
  if(world_rank==maxNode || mpi_bool==false){
    if(phaseGrid == true){
      cout.setf(ios_base::fixed);
      outputPhaseSlice(phase_slice);
      phase_slice<<"\n"<<","<<flush;
    }
  }
}

void domain3::spectrum_output(vector<vector<double>> &spectrum_vect, double stepCurrent, double t_in, double tcurr){
  double repsi=0;
  double impsi=0;
  // if(world_rank==0)
  //   spectrum_energy<<"{{"<<flush;
  for (size_t whichPsi=0; whichPsi<nfields; whichPsi++){
    // if (whichPsi>0) spectrum_energy<<",{";
    #pragma omp parallel for collapse(3) reduction(+:repsi,impsi)
    for(size_t i=0;i<PointsS;i++)
      for(size_t j=0;j<PointsS;j++)
        for(size_t k=nghost;k<PointsSS+nghost;k++){
          repsi=repsi+psi_backup[2*whichPsi][i][j][k-nghost]*psi[2*whichPsi][i][j][k] 
              +psi_backup[2*whichPsi+1][i][j][k-nghost]*psi[2*whichPsi+1][i][j][k];
          impsi=impsi+psi_backup[2*whichPsi][i][j][k-nghost]*psi[2*whichPsi+1][i][j][k] 
              -psi_backup[2*whichPsi+1][i][j][k-nghost]*psi[2*whichPsi][i][j][k];
        }
    #pragma omp barrier

    double totRepsi; // summed up across all nodes
    double totImpsi; // summed up across all nodes
    if(mpi_bool==true){
      MPI_Allreduce(&repsi, &totRepsi, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&impsi, &totImpsi, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      totRepsi = totRepsi*pow(Length/PointsS,3);
      totImpsi = totImpsi*pow(Length/PointsS,3);
    }
    else {
      totRepsi = repsi*pow(Length/PointsS,3);
      totImpsi = impsi*pow(Length/PointsS,3);
    }
    if(world_rank==0){
      vector<double> spect{totRepsi, totImpsi, (double)whichPsi, t_in, tcurr};
      spectrum_vect.push_back(spect);
      // spectrum_energy<<scientific<<totRepsi<<","<<totImpsi<<","<<whichPsi<<","<<t_in<<","<<tcurr <<"}"<<flush;
    }
  }
  // if(world_rank==0)
  //   spectrum_energy<<"}\n"<< ","<<flush;
}

void domain3::spectrum_write(vector<vector<double>> &spectrum_vect){
  if (world_rank == 0) {
    for(int i=0; i<spectrum_vect.size();i++){
      spectrum_energy<<"{"<<scientific<<spectrum_vect[i][0]<<","<<spectrum_vect[i][1]<<","<<spectrum_vect[i][2]<<","
        <<spectrum_vect[i][3]<<","<<spectrum_vect[i][4] <<"}\n"<<","<<flush;
    }
  }
}

void domain3::snapshot_profile(double stepCurrent){// Outputs only few relevant information like mass, energy, radial profiles
  cout.setf(ios_base::fixed);
  if(world_rank==0)
    profilefile<<"{";
  for(int l=0;l<nfields;l++){
    maxdensity[l] = find_maximum(l);
    multi_array<double,2> profile = profile_density(l);
    if(world_rank==0){
      print2(profile,profilefile);
      if(l<nfields-1)
        profilefile<<","<<flush;
    }
  }
  if(world_rank==0){
    profilefile<<"}\n" <<","<<flush;
    timesfile_profile<<"{"<<scientific<<tcurrent;
  }
  // The total quantities computation needs to be run on all nodes
  double Etot =0;
  vector<double> x_CM(3,0); // position of CM
  vector<double> v_CM(3,0); //velocity of CM
  for(int l=0;l<nfields;l++){
    double Ek_full = e_kin_full1(l);
    double Epot_full = full_energy_pot(l);
    double M_tot = total_mass(l);
    Etot = Etot + Ek_full + Epot_full;
    if(world_rank==0){
      timesfile_profile<<","<<scientific<<maxdensity[l]<<","; 
      timesfile_profile<<fixed<<maxx[l][0]<<"," <<maxx[l][1]<<","<<maxx[l][2]<<",";
      timesfile_profile<<scientific<<Ek_full<<","<<Epot_full<<","<<M_tot;
      // if(stepCurrent<numsteps){
    }
  }
  if(world_rank==0) timesfile_profile<<","<<scientific<<Etot;

  // Last points are the center of mass position and velocity
  for(int l=0;l<nfields;l++){
    for(int coord=0;coord<3;coord++){
      x_CM[coord] = x_center_mass(coord, l);
      v_CM[coord] = v_center_mass(coord, l);
    }
    if(world_rank==0){
      timesfile_profile<<","<<scientific<<x_CM[0]<<"," <<x_CM[1]<<"," <<x_CM[2]<<","
        <<v_CM[0]<<"," <<v_CM[1]<<"," <<v_CM[2];
    }
  }
  if(world_rank==0){
    timesfile_profile<<"}\n"<<","<<flush;
    cout<<"Output profile results"<<endl;
  }
}