#include "uldm_mpi_2field.h"

using namespace std;
using namespace boost;

void domain3::openfiles(){ // It opens all the output files, and it inserts an initial {
  if(world_rank==0){
  timesfile_grid.open(outputname+"times_grid.txt");       timesfile_grid<<"{";   timesfile_grid.setf(ios_base::fixed);
  timesfile_profile.open(outputname+"times_profile.txt");       timesfile_profile<<"{";   timesfile_profile.setf(ios_base::fixed);
  profilefile.open(outputname+"profiles.txt");  profilefile<<"{"; profilefile.setf(ios_base::fixed);
  phase_slice.open(outputname+"phase_slice.txt");  phase_slice<<"{"; phase_slice.setf(ios_base::fixed);
  }
  // This you have to open on all nodes, if mpi_bool==true, append the world_rank to the file name
  if(mpi_bool==true){
    profile_sliced.open(outputname+"profile_sliced_node"+to_string(world_rank)+".txt");profile_sliced<<"{"; profile_sliced.setf(ios_base::fixed);
  }
  else {
    profile_sliced.open(outputname+"profile_sliced.txt");profile_sliced<<"{"; profile_sliced.setf(ios_base::fixed);
  }
}

void domain3::openfiles_backup(){ //It opens all the output files in append mode, with no initial insertion of { (for backup mode)
  if(world_rank==0){  
  timesfile_grid.open(outputname+"times_grid.txt", ios_base::app); timesfile_grid.setf(ios_base::fixed);
  timesfile_profile.open(outputname+"times_profile.txt", ios_base::app);  timesfile_profile.setf(ios_base::fixed);
  profilefile.open(outputname+"profiles.txt", ios_base::app); profilefile.setf(ios_base::fixed);
  phase_slice.open(outputname+"phase_slice.txt", ios_base::app);   phase_slice.setf(ios_base::fixed);
  }
  // This you have to open on all nodes, if mpi_bool==true, append the world_rank to the file name
  if(mpi_bool==true){
    profile_sliced.open(outputname+"profile_sliced_node"+to_string(world_rank)+".txt", ios_base::app);profile_sliced.setf(ios_base::fixed);
  }
  else {
    profile_sliced.open(outputname+"profile_sliced.txt", ios_base::app);profile_sliced.setf(ios_base::fixed);
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
  }
}

// It stores run output info
void domain3::exportValues(){
  if(world_rank==0){
    runinfo.open(outputname+"runinfo.txt");
    runinfo.setf(ios_base::fixed);
    runinfo<<tcurrent<<" "<<E_tot_initial<<" "<< Length<<" "<<numsteps<<" "<<PointsS<<" "
        <<numoutputs<<" "<<numoutputs_profile;
    for(int i=0; i<nfields; i++)
    {
      runinfo<<" "<< ratio_mass[i];
    }
    runinfo<<endl;
    runinfo<<"\n";
    runinfo.close();
  }
}

// fileout should have a different name on each node
void domain3::outputfulldensity(ofstream& fileout,int whichPsi){// Outputs the full 3D density profile
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
void domain3::outputfullPsi(ofstream& fileout){// Outputs the full 3D psi, every field, for backup purposes
  if(mpi_bool==true){//if mpi_bool==true, append the world_rank to the file name
    fileout.open(outputname+"psi_final_"+to_string(world_rank)+".txt"); fileout.setf(ios_base::fixed);
  }
  else {
    fileout.open(outputname+"psi_final.txt"); fileout.setf(ios_base::fixed);
  }
  print4_cpp(psi,fileout,nghost);
  fileout.close();
}
void domain3::outputSlicedDensity(ofstream& fileout){ // Outputs the projected 2D density profile
  multi_array<double,3> density_sliced(extents[nfields][PointsS][PointsS]);
  #pragma omp parallel for collapse(4)
  for(int l = 0; l < nfields; l++)
    for(size_t i=0;i<PointsS;i++)
      for(size_t j=0; j<PointsS;j++)
        for(size_t k=nghost; k<PointsSS+nghost;k++){
        density_sliced[l][i][j]= density_sliced[l][i][j] + pow(psi[2*l][i][j][k],2)+pow(psi[2*l+1][i][j][k],2);
        }
  #pragma omp barrier
  print3(density_sliced,fileout);
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
multi_array<double,2> domain3::profile_density(double density_max, int whichPsi){ // It computes the averaged density and energy as function of distance from soliton
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
  #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++)
    for(int j=0; j<PointsS;j++)
      for(int k=nghost; k<PointsSS+nghost;k++){
        int Dx=maxx-(int)i; if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        int Dy=maxy-(int)j; if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // periodic boundary conditions!
        int Dz=maxz-(int)k+extrak; if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // periodic boundary conditions!
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
  #pragma omp barrier
  // For the phase, I take only one ray
  // Only do this on the node that contains the maximum point
  if(world_rank==maxNode || mpi_bool==false){
    int maxz_node = maxz - PointsSS*world_rank +nghost;//maxz in max node with ghost cell
    for(int i=0;i<PointsS;i++){
      int distance =maxx-(int)i; if(abs(distance)>PointsS/2){distance=abs(distance)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
      distance = abs(distance);
      if(distance<pointsmax){
          //Takes only one ray passing from the center of the possible soliton
          binned[5][distance] = atan2(psi[2*whichPsi+1][i][maxy][maxz_node], psi[2*whichPsi][i][maxy][maxz_node]);
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
    timesfile_grid<<"{"<<tcurrent<<","<<maxdensity<<","<<maxx<<"," <<maxy<<","<<maxz;
      // if(stepCurrent<numsteps){
  for(int l=0;l<nfields;l++){
    double Ek_full = e_kin_full1(l);
    double Epot_full = full_energy_pot(l);
    double M_tot = total_mass(l);
    if(world_rank==0){
      timesfile_grid<<","<<Ek_full<<","<<Epot_full<<","<<M_tot;
      // if(stepCurrent<numsteps){
    }
  }
  if(world_rank==0){
    timesfile_grid<<"},\n"<<flush;
    cout<<"Output results"<<endl;
  }
  cout.setf(ios_base::fixed);
  outputSlicedDensity(profile_sliced); // Output of both fields
  profile_sliced<<"\n"<<","<<flush;
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
  if(world_rank==maxNode || mpi_bool==false){
    if(phaseGrid == true){
      cout.setf(ios_base::fixed);
      outputPhaseSlice(phase_slice);
      phase_slice<<"\n"<<","<<flush;
    }
  }
}

void domain3::snapshot_profile(double stepCurrent){// Outputs only few relevant information like mass, energy, radial profiles
  cout.setf(ios_base::fixed);
  if(world_rank==0)
    profilefile<<"{";
  for(int l=0;l<nfields;l++){
    double maxdensityn = find_maximum(l);
    multi_array<double,2> profile = profile_density(maxdensityn,l);
    if(world_rank==0){
      print2(profile,profilefile);
      if(l<nfields-1)
        profilefile<<","<<flush;
    }
  }
  if(world_rank==0)
    profilefile<<"}\n" <<","<<flush;
  
  // The total quantities computation needs to be run on all nodes
  if(world_rank==0)
    timesfile_profile<<"{"<<tcurrent<<","<<maxdensity<<","<<maxx<<"," <<maxy<<","<<maxz;
  for(int l=0;l<nfields;l++){
    double Ek_full = e_kin_full1(l);
    double Epot_full = full_energy_pot(l);
    double M_tot = total_mass(l);
    if(world_rank==0){
      timesfile_profile<<","<<Ek_full<<","<<Epot_full<<","<<M_tot;
      // if(stepCurrent<numsteps){
    }
  }
  if(world_rank==0){
    timesfile_profile<< "}\n" <<","<<flush;
  }
}