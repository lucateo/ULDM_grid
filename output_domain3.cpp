#include "uldm_mpi_2field.h"

using namespace std;
using namespace boost;

void domain3::openfiles(){ // It opens all the output files, and it inserts an initial {
  if(world_rank==0){
  timesfile_grid.open(outputname+"times_grid.txt");       timesfile_grid<<"{";   timesfile_grid.setf(ios_base::fixed);
  timesfile_profile.open(outputname+"times_profile.txt");       timesfile_profile<<"{";   timesfile_profile.setf(ios_base::fixed);
  profilefile.open(outputname+"profiles.txt");  profilefile<<"{"; profilefile.setf(ios_base::fixed);
  phase_slice.open(outputname+"phase_slice.txt");  phase_slice<<"{"; phase_slice.setf(ios_base::fixed);
  profilefile1.open(outputname+"profiles_field1.txt");  profilefile1<<"{"; profilefile1.setf(ios_base::fixed);
  phase_slice1.open(outputname+"phase_slice_field1.txt");  phase_slice1<<"{"; phase_slice1.setf(ios_base::fixed);
  }
  // This you have to open on all nodes
  if(mpi_bool==true){
    profile_sliced.open(outputname+"profile_sliced_node"+to_string(world_rank)+".txt");profile_sliced<<"{"; profile_sliced.setf(ios_base::fixed);
    profile_sliced1.open(outputname+"profile_sliced_node"+to_string(world_rank)+"_field1.txt");profile_sliced1<<"{"; profile_sliced1.setf(ios_base::fixed);
  }
  else {
    profile_sliced.open(outputname+"profile_sliced.txt");profile_sliced<<"{"; profile_sliced.setf(ios_base::fixed);
    profile_sliced1.open(outputname+"profile_sliced_field1.txt");profile_sliced1<<"{"; profile_sliced1.setf(ios_base::fixed);
  }
}

void domain3::openfiles_backup(){ //It opens all the output files in append mode, with no initial insertion of { (for backup mode)
  if(world_rank==0){  
  timesfile_grid.open(outputname+"times_grid.txt", ios_base::app); timesfile_grid.setf(ios_base::fixed);
  timesfile_profile.open(outputname+"times_profile.txt", ios_base::app);  timesfile_profile.setf(ios_base::fixed);
  profilefile.open(outputname+"profiles.txt", ios_base::app); profilefile.setf(ios_base::fixed);
  phase_slice.open(outputname+"phase_slice.txt", ios_base::app);   phase_slice.setf(ios_base::fixed);
  profilefile1.open(outputname+"profiles_field1.txt", ios_base::app); profilefile1.setf(ios_base::fixed);
  phase_slice1.open(outputname+"phase_slice_field1.txt", ios_base::app);   phase_slice1.setf(ios_base::fixed);
  }
  // This you have to open on all nodes
  if(mpi_bool==true){
    profile_sliced.open(outputname+"profile_sliced_node"+to_string(world_rank)+".txt", ios_base::app);profile_sliced.setf(ios_base::fixed);
    profile_sliced1.open(outputname+"profile_sliced_node"+to_string(world_rank)+"_field1.txt", ios_base::app);profile_sliced1.setf(ios_base::fixed);
  }
  else {
    profile_sliced.open(outputname+"profile_sliced.txt", ios_base::app);profile_sliced.setf(ios_base::fixed);
    profile_sliced1.open(outputname+"profile_sliced_field1.txt", ios_base::app);profile_sliced1.setf(ios_base::fixed);
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
    profilefile1.close();
    profile_sliced1.close();
    phase_slice1.close();
  }
}

// It stores run output info
void domain3::exportValues(){
  if(world_rank==0){
    runinfo.open(outputname+"runinfo.txt");
    runinfo.setf(ios_base::fixed);
    runinfo<<tcurrent<<" "<<E_tot_initial<<" "<< Length<<" "<<numsteps<<" "<<PointsS<<" " <<ratio_mass <<" "
        <<numoutputs<<" "<<numoutputs_profile<<endl;
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
  if(mpi_bool==true){
    fileout.open(outputname+"phi_final_"+to_string(world_rank)+".txt"); fileout.setf(ios_base::fixed);
  }
  else {
    fileout.open(outputname+"phi_final.txt"); fileout.setf(ios_base::fixed);
  }
  print3_cpp(Phi,fileout);
  fileout.close();
}
void domain3::outputfullPsi(ofstream& fileout){// Outputs the full 3D psi, both fields, for backup purposes
  if(mpi_bool==true){
    fileout.open(outputname+"psi_final_"+to_string(world_rank)+".txt"); fileout.setf(ios_base::fixed);
  }
  else {
    fileout.open(outputname+"psi_final.txt"); fileout.setf(ios_base::fixed);
  }
  print4_cpp(psi,fileout,nghost);
  fileout.close();
}
void domain3::outputSlicedDensity(ofstream& fileout, int whichPsi){ // Outputs the projected 2D density profile
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

void domain3::outputPhaseSlice(ofstream& fileout, int whichPsi){ // Outputs a 2D slice of the phase
  multi_array<double,2> phase_sliced(extents[PointsS][PointsS]);
  #pragma omp parallel for collapse(2)
    for(size_t i=0;i<PointsS;i++)
      for(size_t j=0; j<PointsS;j++){
        phase_sliced[i][j]= atan2(psi[2*whichPsi+1][i][j][int(PointsSS/2)+nghost], psi[2*whichPsi][i][j][int(PointsSS/2)+nghost]);
      }
  #pragma omp barrier
  print2(phase_sliced,fileout);
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
  // maxz is within the subgrid itself; the actual maxz of the full grid would be maxz+ maxNode*PointsSS
  int extrak;
  if(mpi_bool==true){
    extrak= PointsSS*(world_rank-maxNode); // Correct
  }
  else {
    extrak=0;
  }

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
  if(world_rank==maxNode || mpi_bool==false){
    for(int i=0;i<PointsS;i++){
      int distance =maxx-(int)i; if(abs(distance)>PointsS/2){distance=abs(distance)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
      distance = abs(distance);
      if(distance<pointsmax){
          //Takes only one ray passing from the center of the possible soliton
          binned[5][distance] = atan2(psi[2*whichPsi+1][i][maxy][maxz], psi[2*whichPsi][i][maxy][maxz]);
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

void domain3::snapshot(double stepCurrent){//Outputs the full density profile; if 3dgrid is false, it outputs just sliced density
  cout.setf(ios_base::fixed);
  double Ek_full = e_kin_full1(0);
  double Ek_full1 = e_kin_full1(1);
  double Epot_full = full_energy_pot(0);
  double Epot_full1 = full_energy_pot(1);
  double M_tot = total_mass(0);
  double M_tot1 = total_mass(1);
  if(world_rank==0){
  timesfile_grid<<"{"<<tcurrent<<","<<maxdensity<<","<<maxx<<"," <<maxy<<","<<maxz<<","
      <<Ek_full<<","<<Ek_full1<<","<<Epot_full << ","<<Epot_full1<<","<<M_tot<< ","<<M_tot1<<"}"<<endl;
      // if(stepCurrent<numsteps){
      timesfile_grid<<","<<flush;
      cout<<"Output results"<<endl;
  }
  cout.setf(ios_base::fixed);
  outputSlicedDensity(profile_sliced,0); // Output of both fields
  profile_sliced<<","<<flush;
  cout.setf(ios_base::fixed);
  outputSlicedDensity(profile_sliced1,1);
  profile_sliced1<<","<<flush;
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
          outputPhaseSlice(phase_slice,0);
          phase_slice<<","<<flush;
          cout.setf(ios_base::fixed);
          outputPhaseSlice(phase_slice1,1);
          phase_slice1<<","<<flush;
          }
      // if(stepCurrent< numsteps){
      // } //if it's not the last timeshot put a comma { , , , , }
//           }
  }
//              cout<<"output full grid"<<endl;
}

  void domain3::snapshot_profile(double stepCurrent){// Outputs only few relevant information like mass, energy, radial profiles
    cout.setf(ios_base::fixed);
    double maxdensity0 = find_maximum(0);
    int maxx0 = maxx;
    int maxy0 = maxy;
    int maxz0 = maxz;
    multi_array<double,2> profile = profile_density(maxdensity0,0);
    if(world_rank==0){ print2(profile,profilefile);}
    
    double maxdensity1 = find_maximum(1);
    int maxx1 = maxx;
    int maxy1 = maxy;
    int maxz1 = maxz;
    // cout<<"before profile = "<< time(NULL)- beginning <<endl;
    // profile_density contains MPI_SEND and MPI_RECEIVE sort of functions, so it should run in all nodes nevertheless
    profile = profile_density(maxdensity1,1);
    if(world_rank==0){ print2(profile,profilefile1);}
    
    // The total quantities computation needs to be run on all nodes
    long double e_pot_full = full_energy_pot(0);
    long double e_kin_full_1 = full_energy_kin(0);
    long double e_kin_full_FT = e_kin_full1(0);
    long double M_tot = total_mass(0);
    long double e_pot_full1 = full_energy_pot(1);
    long double e_kin_full_11 = full_energy_kin(1);
    long double e_kin_full_FT1 = e_kin_full1(1);
    long double M_tot1 = total_mass(1);
    if(world_rank==0){
    timesfile_profile<<"{"<<tcurrent<<","<<maxdensity0<<","<<maxx0<<"," <<maxy0<<","<<maxz0<<","
        <<e_kin_full_FT<< ","<< e_kin_full_1 << "," << e_pot_full <<","<<e_kin_full_FT + e_pot_full
        << ","<< e_kin_full_1 + e_pot_full <<","<< M_tot<<","<<maxdensity1<<","<<maxx1<<"," <<maxy1<<","<<maxz1<<","
        <<e_kin_full_FT1<< ","<< e_kin_full_11 << "," << e_pot_full1 <<","<<e_kin_full_FT1 + e_pot_full1
        << ","<< e_kin_full_11 + e_pot_full1 <<","<< M_tot1<<","<< e_kin_full_FT1 + e_pot_full1 + e_kin_full_FT + e_pot_full<<"}"<<endl;
    // if(stepCurrent<numsteps){
    timesfile_profile<<","<<flush;
    profilefile<<","<<flush;
    profilefile1<<","<<flush;
    }
    // } //if it's not the last timeshot put a comma { , , , , }
}
        // void outputfullPsi_Im(ofstream& fileout){// Outputs the full 3D imaginary psi, for backup purposes
        //   multi_array<double,3> psi_store(extents[PointsS][PointsS][PointsSS]);
        //   #pragma omp parallel for collapse(3)
        //   for(size_t i=0;i<PointsS;i++)
        //     for(size_t j=0; j<PointsS;j++)
        //       for(size_t k=nghost; k<PointsSS+nghost;k++){
        //         psi_store[i][j][k]= psi[1][i][j][k];
        //       }
        //   #pragma omp barrier
        //   fileout.open(outputname+"psiIm_final_"+to_string(world_rank)+".txt"); fileout.setf(ios_base::fixed);
        //   print3_cpp(psi_store,fileout);
        //   fileout.close();
        // }