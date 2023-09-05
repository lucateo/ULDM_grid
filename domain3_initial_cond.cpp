#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include "eddington.h"
// Initial conditions functions
// Initial condition with waves, test purposes
void domain3::initial_waves(){
  #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++)
    for(int j=0;j<PointsS;j++)
      for(int k=nghost;k<PointsSS+nghost;k++){
        int kreal=k+world_rank*PointsSS;
        psi[0][i][j][k]+=sin(i/20.)*cos(kreal/20.);
        psi[1][i][j][k]+=sin(kreal/20.)*cos((kreal+i)/20.);
        // psi[2][i][j][k]=sin(i/20.)*cos(kreal/20.);
        // psi[3][i][j][k]=sin(kreal/20.)*cos((kreal+i)/20.);
      }
}

// void setInitialSoliton_1_randomVel(double r_c){ // sets 1 soliton in the center of the grid as initial condition, for testing purposes, with random velocity
//       int center = (int) PointsS / 2; // The center of the grid, more or less
//         double v_x = fRand(0,1);
//         double v_y = fRand(0,1);
//         double v_z = fRand(0,1);
//       #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
//         for(int i=0;i<PointsS;i++){
//             for(int j=0; j<PointsS;j++){
//                 for(int k=0; k<PointsS;k++){
//                   // Distance from the center of the soliton
//                   double radius = deltaX * sqrt( pow( abs(i - center),2) + pow( abs(j - center),2) + pow( abs(k - center),2));
//                   double phase = deltaX * (v_x * i + v_y * j + v_z * k);
//                   psi[0][i][j][k] = psi_soliton(r_c, radius) * cos( phase );
//                   psi[1][i][j][k] = psi_soliton(r_c, radius) * sin( phase );
//                   }
//             }
//         }
//         #pragma omp barrier
//     }

void domain3::setInitialSoliton_1(double r_c, int whichpsi){ // sets 1 soliton in the center of the grid as initial condition, for field 0
  int center = (int) PointsS / 2; // The center of the grid, more or less
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are. If mpi==false, world rank is initialized to 0
  double r = ratio_mass[whichpsi]; //
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
    for(int i=0;i<PointsS;i++){
      for(int j=0; j<PointsS;j++){
        for(int k=nghost; k<PointsSS+nghost;k++){
          // Distance from the center of the soliton
          double radius = deltaX * sqrt( pow( abs(i - center),2) + pow( abs(j - center),2) + pow( abs(k + extrak - center),2));
          psi[2*whichpsi][i][j][k] += psi_soliton(r_c, radius, r);
          // psi[2*whichpsi+1][i][j][k] = 0; // Set the imaginary part to zero
        }
      }
    }
  #pragma omp barrier
}

// void domain3::setInitialSoliton_2(double r_c1, double r_c2, double distance){ // sets 2 solitons as initial condition, for testing purposes
//       int center = (int) PointsS / 2; // The center of the grid, more or less
//       int center1 = center + (int) round(distance/(2*deltaX)); // coordinates of first soliton in x axis
//       int center2 = center - (int) round(distance/(2*deltaX)); // coordinates of second soliton in x axis
//       #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
//         for(int i=0;i<PointsS;i++){
//             for(int j=0; j<PointsS;j++){
//                 for(int k=0; k<PointsS;k++){
//                   // Distance from the center of the soliton, located at the center of the grid but displaced in x axis by distance/2
//                   double radius1 = deltaX * sqrt( pow( i-center1,2) + pow( j-center,2) + pow( k -center,2));
//                   double radius2 = deltaX * sqrt( pow( i-center2,2) + pow( j-center,2) + pow( k-center,2));
//                   psi[0][i][j][k] = psi_soliton(r_c1, radius1) + psi_soliton(r_c2, radius2);
//                   psi[1][i][j][k] = 0; // Set the imaginary part to zero
//                   }
//             }
//         }
//         #pragma omp barrier
//     }

// sets many solitons as initial condition, with random core radius whose centers are confined in a box of length length_lim
// For now, implemented only for field one
void domain3::setManySolitons_random_radius(int num_Sol, double min_radius, double max_radius, double length_lim){
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are. If mpi==false, world rank is initialized to 0
  vector<int> random_x(num_Sol,0);
  vector<int> random_y(num_Sol,0);
  vector<int> random_z(num_Sol,0);
  vector<double> r_c(num_Sol,0); // the core radius array

  if(world_rank==0){
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, min_rad, max_rad and length_lim, the other rows are centers of solitons and radii
    info_initial_cond<<"{{"<<num_Sol<<","<<min_radius<<","<<max_radius << ","<<length_lim << "}"<< "," <<endl;
    for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
      r_c[i] = fRand(min_radius, max_radius);
      random_x[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_y[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_z[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      info_initial_cond<<"{"<< r_c[i] << "," <<random_x[i]<<","<<random_y[i]<< ","<<random_z[i]<< "}";
      if (i<num_Sol-1) {info_initial_cond<<","<<endl;} // If it is not the last data point, put a comma
    }
    info_initial_cond<<"}";
    info_initial_cond.close();
  }
  // send to other nodes
  if(world_rank==0 && mpi_bool==true){
    for(int lpb=1;lpb<world_size;lpb++){ // send to every other node
      MPI_Send(&random_x.front(), random_x.size(), MPI_INT, lpb, 100+lpb, MPI_COMM_WORLD);
      MPI_Send(&random_y.front(), random_y.size(), MPI_INT, lpb, 200+lpb, MPI_COMM_WORLD);
      MPI_Send(&random_z.front(), random_x.size(), MPI_INT, lpb, 300+lpb, MPI_COMM_WORLD);
      MPI_Send(&r_c.front(), r_c.size(), MPI_DOUBLE, lpb, 400+lpb, MPI_COMM_WORLD);
    }
  }
  // receive from node 0
  if(world_rank!=0 && mpi_bool==true){
    MPI_Recv(&random_x.front(),random_x.size(), MPI_INT, 0, 100+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&random_y.front(),random_y.size(), MPI_INT, 0, 200+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&random_z.front(),random_z.size(), MPI_INT, 0, 300+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&r_c.front(),r_c.size(), MPI_DOUBLE, 0, 400+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
  for(int i=0;i<PointsS;i++){
    for(int j=0; j<PointsS;j++){
      for(int k=nghost; k<PointsSS+nghost;k++){
        for(int l=0; l<num_Sol; l++){
          double radius = deltaX * sqrt( pow( i-random_x[l],2) + pow( j-random_y[l],2) + pow( k-random_z[l]+extrak,2));
          psi[0][i][j][k] += psi_soliton(r_c[l], radius, 1);
        }
        // psi[1][i][j][k] = 0; // Set the imaginary part to zero
      }
    }
  }
  #pragma omp barrier
}

// sets many solitons as initial condition, with same core radius whose centers are confined in a box of length length_lim
// For now, implemented only for field one
void domain3::setManySolitons_same_radius(int num_Sol, double r_c, double length_lim){
  #pragma omp barrier
  vector<int> random_x(num_Sol,0);
  vector<int> random_y(num_Sol,0);
  vector<int> random_z(num_Sol,0);
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are
  if(world_rank==0){
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, r_c and length_lim, the other rows are centers of solitons
    info_initial_cond<<"{{"<<num_Sol<<","<<r_c<<","<<length_lim << "}"<< "," <<endl;
    for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
      random_x[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_y[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_z[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      info_initial_cond<<"{"<<random_x[i]<<","<<random_y[i]<< ","<<random_z[i]<< "}";
      if (i<num_Sol-1) {info_initial_cond<<","<<endl;} // If it is not the last data point, put a comma
    }
    info_initial_cond<<"}";
    info_initial_cond.close();
  }
    // send to other nodes, 4th entry should be the node you send to, 5th entry is a tag (to be shared between receiver and transmitter)
  if(world_rank==0 && mpi_bool==true){
    for(int lpb=1;lpb<world_size;lpb++){ // send to every other node
      MPI_Send(&random_x.front(), random_x.size(), MPI_INT, lpb, 100+lpb, MPI_COMM_WORLD);
      MPI_Send(&random_y.front(), random_y.size(), MPI_INT, lpb, 200+lpb, MPI_COMM_WORLD);
      MPI_Send(&random_z.front(), random_x.size(), MPI_INT, lpb, 300+lpb, MPI_COMM_WORLD);
    }
  }
  // receive from node 0, 4th entry should be the node you receive from, 5th entry is a tag (to be shared between receiver and transmitter)
  if(world_rank!=0 && mpi_bool==true){
    MPI_Recv(&random_x.front(),random_x.size(), MPI_INT, 0, 100+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&random_y.front(),random_y.size(), MPI_INT, 0, 200+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&random_z.front(),random_z.size(), MPI_INT, 0, 300+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
  for(int i=0;i<PointsS;i++){
    for(int j=0; j<PointsS;j++){
      for(int k=nghost; k<PointsSS+nghost;k++){
        for(int l=0; l<num_Sol; l++){
          double radius = deltaX * sqrt( pow( i-random_x[l],2) + pow( j-random_y[l],2) + pow( k+extrak - random_z[l],2));
          psi[0][i][j][k] += psi_soliton(r_c, radius, 1); //implemented only for field 1
        }
        // psi[1][i][j][k] = 0; // Set the imaginary part to zero
      }
    }
  }
}

// // sets many solitons as initial condition, with same core radius, and random initial velocities (NOT REFINED)
// void domain3::setManySolitons_same_radius_random_vel(int num_Sol, double r_c){
//         #pragma omp barrier
//       int random_x[num_Sol];
//       int random_y[num_Sol];
//       int random_z[num_Sol];
//         double v_x[num_Sol];
//         double v_y[num_Sol];
//         double v_z[num_Sol];
//       for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
//         random_x[i]= 2*round(r_c/deltaX) + rand()%(int)(PointsS - 4*round(r_c/deltaX) );
//         random_y[i]= 2*round(r_c/deltaX) + rand()%(int)(PointsS - 4*round(r_c/deltaX) );
//         random_z[i]= 2*round(r_c/deltaX) + rand()%(int)(PointsS - 4*round(r_c/deltaX) );
//         v_x[i] = fRand(0,1);
//         v_y[i] = fRand(0,1);
//         v_z[i] = fRand(0,1);
//       }
//       #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
//       for(int i=0;i<PointsS;i++){
//         for(int j=0; j<PointsS;j++){
//           for(int k=0; k<PointsS;k++){
//             psi[0][i][j][k] = 0; //set it to zero just to be sure
//             psi[1][i][j][k] = 0; //set it to zero just to be sure
//             for(int l=0; l<num_Sol; l++){
//               double radius = deltaX * sqrt( pow( i-random_x[l],2) + pow( j-random_y[l],2) + pow( k-random_z[l],2));
//               double phase=deltaX*(v_x[l] *i + v_y[l] *j + v_z[l] *k );
//               psi[0][i][j][k] += psi_soliton(r_c, radius)*cos(phase);
//             psi[1][i][j][k] += psi_soliton(r_c, radius)*sin(phase); // Set the imaginary part to zero
//             }
//           }
//         }
//       }
//   }

void domain3::setManySolitons_deterministic(double r_c, int num_sol){
// sets num_sol solitons (maximum of 30 solitons allowed)
// as initial condition, with same core radius, with fixed positions (chosen randomly), for testing purposes
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are
  float x_phys[30] ={-10.010355128996338, -5.632161391614467, -10.804162195884118, 27.74117880694743, -2.4011805803557067, -23.25311822651404, 25.705720076675775, 37.353679014311595, -6.946204576810793, -23.817538236920797, -14.655394739433873, -16.749637898661014, -19.580883326003217, 18.605740761167112, 27.25736094744731, -30.41462054172923, -21.33315514195605, 16.197348275236898, 3.799620205223448, 6.934702548652282, -16.941096181540544, 2.2781062254869866, 36.8554262382406, 36.35367270175507, 11.139625002335258, -30.58395762855323, 17.390547027384528, 35.351726508377, 34.30615396835661, 22.71362595499526} ;
  float y_phys[30] = {31.886070793802816, 39.38615345350138, 24.231389564760576, -15.835439299197375, -3.3411149363077755, -5.730194340705353, 0.7324849001187346, 33.325544978300755, -16.688024218649502, 9.90167057634315, 25.356905509911215, -2.122232317006876, -15.220208473815227, 7.791086225543076, 10.479565234160418, -1.9731717922202847, 15.462493807443863, -6.4553532820384305, -20.80826527413479, 22.966765839735572, -33.034190626746195, 28.861838426003686, 33.1885768361993, 14.555629618935612, -1.4330134748238805, -9.755948008891544, -2.6452408889946426, 16.665940880504834, 39.45576480300235, -24.93284829451899};
  float z_phys[30] = {-31.810875938706424, 36.60281686740946, 21.634248190233677, -1.0920954142129702, -35.06404006114212, 20.92585035786638, -9.280765022493433, 0.8541286579386949, 0.07578851246192642, -17.007163121712257, 18.26124323465008, 5.980555848100558, 6.464719622534602, -15.047019341759356, 16.629246652083538, 20.983444417130237, -10.701969326834153, -36.6850662895899, -9.833232106681116, -15.800997680268427, 35.40845053441541, 8.837229367223543, -16.95296121220478, -10.596280760678791, -22.789096875557515, -39.44176295453228, 31.794778400711067, -38.161234613721355, 17.82624369418062, -28.17723677444623};
  int center = (int) PointsS / 2; // The center of the grid, more or less
  int x[num_sol];
  int y[num_sol];
  int z[num_sol];
  // int center1 = center - (int) round((num_Sol_lin_dim -1)*distance/(2*deltaX)); // coordinates of edge of box
  for(int i=0;i<num_sol;i++){
    x[i]= center + (int) round(x_phys[i] / deltaX);
    y[i]= center + (int) round(y_phys[i] / deltaX);
    z[i]= center + (int) round(z_phys[i] / deltaX);
  }
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
    for(int i=0;i<PointsS;i++){
        for(int j=0; j<PointsS;j++){
            for(int k=nghost; k<PointsSS+nghost;k++){
              for(int l=0; l<num_sol; l++){
                double radius = deltaX * sqrt( pow( i-x[l],2) + pow( j-y[l],2) + pow( k +extrak-z[l],2));
                psi[0][i][j][k] += psi_soliton(r_c, radius,1);
              }
              // psi[1][i][j][k] = 0; // Set the imaginary part to zero
            }
        }
    }
}

void domain3::set_waves_Levkov(multi_array<double, 1> Nparts){
  // Sets waves initial conditions a la Levkov, Nparts is number of particles, for the various fields
  for(int i=0; i<nfields; i++){
    fgrid.inputSpectrum(Length, Nparts[i], ratio_mass[i]);
    fgrid.calculateFT();
    fgrid.transferpsi_add(psi, 1./pow(Length,3),nghost,i);
  }
  if(world_rank==0){
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    info_initial_cond.setf(ios_base::fixed); 
    info_initial_cond<<"{";
    for(int i=0; i<nfields; i++){
      info_initial_cond<<Nparts[i];
      if(i<nfields-1) info_initial_cond<<",";
    }
    if(nfields>1){
      for(int i=1; i<nfields; i++){
        info_initial_cond<<ratio_mass[i];
        if(i<nfields-1) info_initial_cond<<",";
      }
    }
    info_initial_cond<<"}" <<endl;
    info_initial_cond.close();
  }
}

void domain3::set_delta(multi_array<double, 1> Nparts){
  // Sets delta in Fourier space initial conditions, Nparts is number of particles, for the various fields
  for(int i=0; i<nfields; i++){
    fgrid.inputDelta(Length, Nparts[i], ratio_mass[i]);
    fgrid.calculateFT();
    fgrid.transferpsi_add(psi, 1./pow(Length,3),nghost,i);
  }
  if(world_rank==0){
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    info_initial_cond.setf(ios_base::fixed); 
    info_initial_cond<<"{";
    for(int i=0; i<nfields; i++){
      info_initial_cond<<Nparts[i];
      if(i<nfields-1) info_initial_cond<<",";
    }
    if(nfields>1){
      for(int i=1; i<nfields; i++){
        info_initial_cond<<ratio_mass[i];
        if(i<nfields-1) info_initial_cond<<",";
      }
    }
    info_initial_cond<<"}" <<endl;
    info_initial_cond.close();
  }
}

void domain3::set_theta(multi_array<double, 1> Nparts){
  // Sets delta in Fourier space initial conditions, Nparts is number of particles, for the various fields
  for(int i=0; i<nfields; i++){
    fgrid.inputTheta(Length, Nparts[i], ratio_mass[i]);
    fgrid.calculateFT();
    fgrid.transferpsi_add(psi, 1./pow(Length,3),nghost,i);
  }
  if(world_rank==0){
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    info_initial_cond.setf(ios_base::fixed); 
    info_initial_cond<<"{";
    for(int i=0; i<nfields; i++){
      info_initial_cond<<Nparts[i];
      if(i<nfields-1) info_initial_cond<<",";
    }
    if(nfields>1){
      for(int i=1; i<nfields; i++){
        info_initial_cond<<ratio_mass[i];
        if(i<nfields-1) info_initial_cond<<",";
      }
    }
    info_initial_cond<<"}" <<endl;
    info_initial_cond.close();
  }
}

void domain3::setEddington(Eddington *eddington, int numpoints, double radmin, double radmax){
  if (eddington->analytic_Eddington == false){
    eddington->compute_d2rho_dpsi2_arr(numpoints, radmin, radmax);
    eddington->compute_fE_arr();
  }
  Profile *profile = eddington->get_profile();
  int center = int(PointsS/2);
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are
  // double density_convert = 4*M_PI*G_ASTRO/pow(m,2) *pow(HBAR_EV*CLIGHT /(1E3*PC2METER) ,2)/pow(beta,2);
  // double x_convert = 1/sqrt(beta)/m*HBAR_EV*CLIGHT/1E3/PC2METER;
  vector <double> phases;
  for(int i = 0; i <PointsS/2; i++)
    phases.push_back(fRand(0, 2*M_PI)); // Different phases should depend only on velocity, to ensure continuity on physical space
  #pragma omp parallel for collapse(3)
  for(int i=0; i<PointsS; i++)
    for(int j=0; j<PointsS; j++)
      for(int k=nghost; k<PointsSS+nghost; k++){
        // f(E) is defined up to a minimum energy given by radmin, ensure you do not go below it
        double distance = max(Length/PointsS *sqrt( pow(i-center,2) + pow(j-center,2) + pow(k+extrak-center,2)), radmin);
        // ceil rounds up to nearest integer
        int kmax = min(int(PointsS/2),int (ceil(sqrt(2*profile->Psi(distance)) * Length/(2*M_PI) ) ));
        double psi_point_real = 0;
        double psi_point_im = 0;
        for(int vv=0; vv<kmax; vv++){
          double vtilde = 2*M_PI/Length *vv;
          double E = profile->Psi(distance) - pow(vtilde,2)/2; // You have to ensure E > 0, for bound system
          double fe = eddington->fE_func(E);
          double psi_point = 4*sqrt(M_PI/distance)*M_PI/Length*sqrt(fe *vv * sin(vv*distance*2*M_PI/Length)) ;
          psi_point_real += psi_point*cos(phases[vv]);
          psi_point_im += psi_point*sin(phases[vv]);
        }
        psi[0][i][j][k] +=   psi_point_real;
        psi[1][i][j][k] += psi_point_im;
      }
}