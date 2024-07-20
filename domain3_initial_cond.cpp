#include "uldm_mpi_2field.h"
#include <boost/multi_array.hpp>
#include <cmath>
#include <cstddef>
#include <string>
#include <random>
#include "eddington.h"
// Initial conditions functions
// Initial condition with waves, test purposes
void domain3::initial_waves(int whichF){
  #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++)
    for(int j=0;j<PointsS;j++)
      for(int k=nghost;k<PointsSS+nghost;k++){
        int kreal=k+world_rank*PointsSS;
        psi[2*whichF][i][j][k]+=sin(i/20.)*cos(kreal/20.);
        psi[2*whichF +1][i][j][k]+=sin(kreal/20.)*cos((kreal+i)/20.);
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

void domain3::setInitialSoliton_1(double r_c, int whichpsi){ // sets 1 soliton in the center of the grid as initial condition, for field whichpsi
  int center = (int) PointsS / 2; // The center of the grid, more or less
  // int center = 20; // The center of the grid, more or less
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are. If mpi==false, world rank and nghost are initialized to 0
  double ratio = ratio_mass[whichpsi]; //
  if (first_initial_cond == true){ 
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    first_initial_cond = false;
  }
  else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
  info_initial_cond<<"1Soliton " << r_c << " " << whichpsi << endl;
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
    for(int i=0;i<PointsS;i++){
      for(int j=0; j<PointsS;j++){
        for(int k=nghost; k<PointsSS+nghost;k++){
          // Distance from the center of the soliton
          double radius = deltaX * sqrt( pow( abs(i - center),2) + pow( abs(j - center),2) + pow( abs(k + extrak - center),2));
          // double phase = deltaX * (0.1 * i + 0.3 * j - 0.05*(k+extrak));
          psi[2*whichpsi][i][j][k] += psi_soliton(r_c, radius, ratio) ;
          psi[2*whichpsi+1][i][j][k] = 0; 
        }
      }
    }
  #pragma omp barrier
}

void domain3::setTest(){
  info_initial_cond.open(outputname+"initial_cond_info.txt");
  info_initial_cond<<"Test" << endl;
}

// Uniform density sphere for test purposes, od density rho0 and radius rad
void domain3::uniform_sphere(double rho0, double rad){
  int center = (int) PointsS / 2; // The center of the grid, more or less
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are. If mpi==false, world rank and nghost are initialized to 0
  if (first_initial_cond == true){ 
    info_initial_cond.open(outputname+"initial_cond_info.txt");
    first_initial_cond = false;
  }
  else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
  info_initial_cond<<"Uniform_sphere " << rho0 << " " << rad << endl;
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
    for(int i=0;i<PointsS;i++){
      for(int j=0; j<PointsS;j++){
        for(int k=nghost; k<PointsSS+nghost;k++){
          // Distance from the center of the soliton
          double radius = deltaX * sqrt( pow( abs(i - center),2) + pow( abs(j - center),2) + pow( abs(k + extrak - center),2));
          if (radius < rad)
            psi[0][i][j][k] += sqrt(rho0);
        }
      }
    }
  #pragma omp barrier
}

// sets many solitons as initial condition, with random core radius whose centers are confined in a box of length length_lim
void domain3::setManySolitons_random_radius(int num_Sol, double min_radius, double max_radius, double length_lim, int whichF){
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are. If mpi==false, world rank is initialized to 0
  vector<int> random_x(num_Sol,0);
  vector<int> random_y(num_Sol,0);
  vector<int> random_z(num_Sol,0);
  vector<double> r_c(num_Sol,0); // the core radius array
  double ratio = ratio_mass[whichF];

  if(world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, min_rad, max_rad and length_lim, the other rows are centers of solitons and radii
    info_initial_cond<<"Mocz "<<num_Sol<<" "<<min_radius<<" "<<max_radius << " "<<length_lim << " " << ratio  <<endl;
    for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
      r_c[i] = fRand(min_radius, max_radius);
      random_x[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_y[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_z[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      info_initial_cond<<"Mocz_soliton_"+to_string(i)<<" " << r_c[i] << " " <<random_x[i]<<" "<<random_y[i]<< " "<<random_z[i]<<endl;
    }
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
          psi[2*whichF][i][j][k] += psi_soliton(r_c[l], radius, ratio);
        }
        // psi[1][i][j][k] = 0; // Set the imaginary part to zero
      }
    }
  }
  #pragma omp barrier
}

// sets many solitons as initial condition, with same core radius whose centers are confined in a box of length length_lim
void domain3::setManySolitons_same_radius(int num_Sol, double r_c, double length_lim, int whichF){
  #pragma omp barrier
  vector<int> random_x(num_Sol,0);
  vector<int> random_y(num_Sol,0);
  vector<int> random_z(num_Sol,0);
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are
  double ratio = ratio_mass[whichF];
  if(world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed); // First row of the file is num_sol, r_c and length_lim, the other rows are centers of solitons
    info_initial_cond<<"Schive "<<num_Sol<<" "<<r_c<<" "<<length_lim << " " << ratio <<endl;
    for(int i=0;i<num_Sol;i++){//Leave some space with respect to the edge of the grid
      random_x[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_y[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      random_z[i]= round((Length - length_lim)/(2*deltaX)) + rand()%(int)(PointsS - round((Length - length_lim)/deltaX) );
      info_initial_cond<<"Schive_soliton_"+to_string(i)<<" "<<random_x[i]<<" "<<random_y[i]<< " "<<random_z[i]<< endl;
    }
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
          psi[2*whichF][i][j][k] += psi_soliton(r_c, radius, ratio); 
        }
        // psi[1][i][j][k] = 0; // Set the imaginary part to zero
      }
    }
  }
}


void domain3::setManySolitons_deterministic(double r_c, int num_sol, int whichF){
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
  if(world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed);
    info_initial_cond<<"deterministic"; 
    info_initial_cond<<" "<< r_c << " " <<num_sol << " " << ratio_mass[whichF] << endl;
    info_initial_cond.close();
  }
  double ratio = ratio_mass[whichF];
  #pragma omp parallel for collapse(3) //not sure whether this is parallelizable
    for(int i=0;i<PointsS;i++){
        for(int j=0; j<PointsS;j++){
            for(int k=nghost; k<PointsSS+nghost;k++){
              for(int l=0; l<num_sol; l++){
                double radius = deltaX * sqrt( pow( i-x[l],2) + pow( j-y[l],2) + pow( k +extrak-z[l],2));
                psi[2*whichF][i][j][k] += psi_soliton(r_c, radius,ratio);
              }
              // psi[1][i][j][k] = 0; // Set the imaginary part to zero
            }
        }
    }
}


void domain3::set_waves_Levkov(double Nparts, int whichF){
  // Sets waves initial conditions a la Levkov, Nparts is number of particles, for the various fields
  fgrid.inputSpectrum(Length, Nparts, ratio_mass[whichF]);
  fgrid.calculateFT();
  fgrid.transferpsi_add(psi, 1./pow(Length,3),nghost,whichF);
  if(world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed);
    info_initial_cond<<"Levkov_field_"+to_string(nfields); 
    info_initial_cond<<" "<< Nparts << " " << ratio_mass[whichF] << endl;
    info_initial_cond.close();
  }
}


void domain3::set_delta(double Nparts, int whichF){
  // Sets delta in Fourier space initial conditions, Nparts is number of particles, for the various fields
  fgrid.inputDelta(Length, Nparts, ratio_mass[whichF]);
  fgrid.calculateFT();
  fgrid.transferpsi_add(psi, 1./pow(Length,3),nghost,whichF);
  if(world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed); 
    info_initial_cond<<"Delta_field_"+to_string(nfields); 
    info_initial_cond<<" "<<Nparts<< " "<< ratio_mass[whichF] << endl;
    info_initial_cond.close();
  }
}

void domain3::set_theta(double Nparts, int whichF){
  // Sets delta in Fourier space initial conditions, Nparts is number of particles, for the various fields
  fgrid.inputTheta(Length, Nparts, ratio_mass[whichF]);
  fgrid.calculateFT();
  fgrid.transferpsi_add(psi, 1./pow(Length,3),nghost,whichF);
  if(world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed); 
    info_initial_cond<<"Theta_field_"+to_string(nfields); 
    info_initial_cond<<" "<<Nparts<<" "<< ratio_mass[whichF] << endl;
    info_initial_cond.close();
  }
}

// sets |psi|^2 = norm*Exp(-(x/a_e)^2 - (y/b_e)^2 - (z/c_e)^2), for field whichPsi; 
// if random is 1, insert gaussian correlation C(x) = A\exp(-x^2/l_corr^2); if random=1, addition this field configuration AFTER
// anothe initial condition IS NOT SUPPORTED!!! (Yet)
// The major axis of the ellipsoid are dispalced by two random angles with respect to x,y,z grid directions
void domain3::setEllitpicCollapse(double norm, double a_e, double b_e, double c_e, int whichPsi, bool rand_phases, double A_rand, double l_corr){ 
  int center = (int) PointsS / 2; // The center of the grid, more or less
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are. If mpi==false, world rank and nghost are initialized to 0
  
  srand(time(NULL)); // Initialize the random seed for random phases
  // Two random angles, theta for rotation around z axis followed by rotation around x axis of angle phi
  double theta_rand = fRand(0, 2*M_PI);
  double phi_rand = fRand(0, 2*M_PI);
  // double ratio = ratio_mass[whichpsi]; //
  
  if (world_rank == 0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond<<"Elliptic_Collapse " << norm << " " << a_e << " " << b_e << " " << c_e << " " << ratio_mass[whichPsi] 
      << " " << random << " " << A_rand << " " << l_corr << " " << theta_rand << " " << phi_rand << endl;
  }
  if (A_rand != 0 && rand_phases==false){
    fgrid.input_Gauss_corr(A_rand, l_corr, Length);
    fgrid.calculateFT();
    fgrid.transferpsi_add(psi, 1/pow(Length,3), nghost, whichPsi); 
  }
  // Starts with putting phases,
  else if (A_rand != 0 && rand_phases==true){
    // I will take just the imaginary part, so multiply A_rand by two to obtain the correct correlation
    fgrid.input_Gauss_corr(2*A_rand, l_corr, Length);
    fgrid.calculateFT();
    fgrid.transferpsi_add(psi, 1/pow(Length,3), nghost, whichPsi); 
  }

  #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++){
    for(int j=0; j<PointsS;j++){
      for(int k=nghost; k<PointsSS+nghost;k++){
        // Rotations are around the center of the grid, 
        int reali = i - center;
        int realj = j- center;
        int realk = k + extrak - center;
        double xprime = deltaX *(reali* cos(theta_rand) + realj* sin(theta_rand));
        double yprime = deltaX*(-cos(phi_rand)*sin(theta_rand)*reali + cos(phi_rand)*cos(theta_rand)*realj + sin(phi_rand)*realk);  
        double zprime = deltaX*(sin(phi_rand)*sin(theta_rand)*reali - sin(phi_rand)*cos(theta_rand)*realj + cos(phi_rand)*realk);  
        double exponent = (pow( abs(xprime)/a_e,2)/2 + pow( abs(yprime)/b_e,2)/2 
            + pow( abs(zprime)/c_e,2)/2);
        if(rand_phases == false) psi[2*whichPsi][i][j][k] += sqrt(norm)*exp(-exponent);
        else if (rand_phases == true) {
          // I will use the phases stored in the imaginary axis
          psi[2*whichPsi][i][j][k] = sqrt(norm)*exp(-exponent)*cos(psi[2*whichPsi+1][i][j][k]);
          psi[2*whichPsi+1][i][j][k] = sqrt(norm)*exp(-exponent)*sin(psi[2*whichPsi+1][i][j][k]);
        }
      }
    }
  }
  #pragma omp barrier
}



// NEEDS SERIOUS DEBUGGING
void domain3::set_initial_from_file(string filename_in, string filename_vel){
  multi_array<double,1> Arr1D(extents[nfields*PointsS*PointsS*PointsSS]);
  ifstream infile(filename_in);
  string temp;
  double num;
  size_t l = 0;
  // Get the input from the input file
  while (std::getline(infile, temp, ' ')) {
    // Add to the list of output strings
    num = stod(temp);
    Arr1D[l] = num;
    l++;
  }
  infile.close();
  // Now the velocities
  infile = ifstream(filename_vel);
  l=0;
  multi_array<double,1> Arr1D_vel(extents[3*nfields*PointsS*PointsS*PointsSS]);
  while (std::getline(infile, temp, ' ')) {
    // Add to the list of output strings
    num = stod(temp);
    Arr1D_vel[l] = num;
    l++;
  }
  infile.close();

  for(int whichF=0;whichF<nfields;whichF++){
    for(int which_coord=0;which_coord<3;which_coord++){
      fgrid.input_arr(Arr1D_vel,whichF, which_coord);
      fgrid.calculateFT();                                         //calculates the FT
      fgrid.kfactor_vel(Length, ratio_mass[whichF],which_coord);   //multiplies by \vec{k}*\vec{x}/k^2
      fgrid.calculateIFT();                                        //calculates the inverse FT
      // Put it in psi arr to spare memory. FT and IFT should be linear, so I
      // just add k_x*v_x(k) + k_y*v_y(k) + k_z*v_z(k)
      // The imaginary part of psi array should all be close to zero, since we are doing Fourier transform of a real quantity
      fgrid.transferpsi_add(psi,1./pow(PointsS,3),nghost,whichF);   //divides it by 1/PS^3 (to get the correct normalizaiton of the FT)
    }
  }
  // This is me trying to do it with line integral, probably not the smartest of choices
  // for(size_t nf = 0; nf < nfields; nf++)
  //   for(size_t i =0; i<PointsS; i++){
  //         if(i!=0) psi[2*nf+1][i][0][0] = Arr1D[i-1+0*PointsS+0*PointsS*PointsS + nf*PointsS*PointsS*PointsSS] *deltaX + psi[nf][i][0][0];
  //     for(size_t j =0; j<PointsS-1; j++){
  //         if(j!=0) psi[2*nf+1][i][j][0] = Arr1D[i+(j-1)*PointsS+0*PointsS*PointsS + (nf+1)*PointsS*PointsS*PointsSS] *deltaX + psi[nf][i][j][0];
  //       for(size_t k =0; k<PointsSS-1; k++){
  //         // Put the phases in imaginary psi array to not use further memory
  //         psi[2*nf+1][i][j][k+1] = Arr1D[i+j*PointsS+k*PointsS*PointsS + (nf+2)*PointsS*PointsS*PointsSS] *deltaX + psi[nf][i][j][k];
  //       }
  //     }     
  //   }
  // recall that phases are stored on psi array
  for(size_t nf = 0; nf < nfields; nf++)
    for(size_t i =0; i<PointsS; i++)
      for(size_t j =0; j<PointsS; j++)
        for(size_t k =0; k<PointsSS; k++){
          // Phases are stored on real part of psi array, so initialize imaginary psi first!
          psi[2*nf+1][i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS + PointsS*PointsS*PointsSS*nf]*sin(psi[2*nf][i][j][k]);
          psi[2*nf][i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS + PointsS*PointsS*PointsSS*nf]*cos(psi[2*nf][i][j][k]);
        }
}

void domain3::setEddington(Eddington *eddington, int numpoints, double radmin, double radmax, int fieldid, 
  double ratiomass, int num_k, bool simplify_k, int center_x, int center_y,int center_z){
  // If simplify_k is true, then it inserts the random phases as gaussians on |k|; if false, then it inserts the random phases
  // in every \vec{k} point. num_k is the total number of k points you run the nested loop over (to speed up computation), put 16 or 32
  // If the profile does not have analytic formulas for f(E), or potential profile \neq density profile, or there are multiple profiles,
  // compute it numerically
  if ((eddington->analytic_Edd[0] == false || eddington->same_profile_den_pot ==false) || eddington->analytic_Edd.size()>1){ 
    eddington->compute_d2rho_dpsi2_arr(numpoints, radmin, radmax);
    eddington->compute_fE_arr();
    cout<<"Finished computing numeric f(E) for Eddington initial conditions"<<endl;
    if (world_rank == 0){
      vector<double> psiarr = eddington->get_psiarr();
      vector<double> fe_arr = eddington->get_fE_arr();
      cout<< "E values" << "\t"<<"f(E)" << endl;
      for(int i=0; i< psiarr.size(); i++){
        cout<< psiarr[i] << "\t"<<fe_arr[i] << endl;
      }
    }
  }
  srand(time(NULL)); // Initialize the random seed for random phases
  
  int extrak= PointsSS*world_rank -nghost; // Take into account the node where you are
  
  // Maximum kmax for phases initialization
  int kmax_phases = min(int(PointsS/2),int (ceil(sqrt(2*eddington->psi_potential(radmin)) * Length*ratiomass/(2*M_PI) ) ) );
  // Maximum k allowed given the potential
  int kmax_global = int (ceil(sqrt(2*eddington->psi_potential(radmin)) * Length*ratiomass/(2*M_PI) ) );
  // If kmax_global is too big, skips some k to speed up the initialization
  int num_k_real = min(int(PointsS/2), num_k); // The total number of k should not surpass PointsS/2
  // int num_k_real = num_k; 
  int skip_k = int(ceil(float(kmax_global)/num_k_real)); // Remember to ensure float division!
  if(world_rank==0) cout<< "skip k " << skip_k << endl;
  bool klow =false; // if true, it means that the kmax is too low, hence needing to do the sum in a different way to recover the correct target density
  if (kmax_global <10) {
    klow =true;
  }
  
  if (world_rank==0){
    if (first_initial_cond == true){ 
      info_initial_cond.open(outputname+"initial_cond_info.txt");
      first_initial_cond = false;
    }
    else info_initial_cond.open(outputname+"initial_cond_info.txt", ios_base::app);
    info_initial_cond.setf(ios_base::fixed); 
    for (int id_prof=0; id_prof< eddington->profile_pot_size(); id_prof++){
      info_initial_cond<<eddington->get_profile_pot(id_prof)->name_profile;
      for(int i = 0; i < eddington->get_profile_pot(id_prof)->params.size(); i++){
        info_initial_cond<<" "<<eddington->get_profile_pot(id_prof)->params_name[i]<<" "
          <<eddington->get_profile_pot(id_prof)->params[i] << " ";
      }
    } 
    info_initial_cond<<"ratio_mass "<< ratiomass << " " << num_k << " " <<simplify_k << " " << center_x
        << " " << center_y << " " << center_z;
    
    //If the density profile does not fully source the potential, print information about it as well
    if(eddington->same_profile_den_pot==false){
      for (int id_prof=0; id_prof< eddington->profile_den_size(); id_prof++){
        info_initial_cond<<" " << eddington->get_profile_den(id_prof)->name_profile <<"__density";
      for(int i = 0; i < eddington->get_profile_den(id_prof)->params.size(); i++){
        info_initial_cond<<" "<<eddington->get_profile_den(id_prof)->params_name[i]
          <<" "<<eddington->get_profile_den(id_prof)->params[i];
      } 
    }
    info_initial_cond<<endl;
    info_initial_cond.close();
    cout<< "kmax global for Eddington initial conditions: "<< kmax_global<<" "
      << sqrt(2*eddington->psi_potential(radmin)) *ratiomass* Length/(2*M_PI)  <<endl;
    }
  }
  vector<double> phases_send(int(pow(2*num_k_real,3)), 0);
  random_device rd;
  default_random_engine generator(rd()); // To generate gaussian distributed numbers, rd should initialize a random seed

  if (simplify_k==false){ 
    // multi_array<double, 3> phases(extents[2*kmax_global][2*kmax_global][2*kmax_global]);
    // Phases should depend only on velocity, so initialize them once here, on world rank 0
    if (world_rank ==0){
      for(size_t k=0; k<2*num_k_real; k++){
        for(size_t j=0; j<2*num_k_real; j++)
          for(size_t i=0; i<2*num_k_real; i++){
            phases_send[i + j*size_t(2*num_k_real) + k*size_t(4*num_k_real*num_k_real)] = fRand(0, 2*M_PI); // Phases should depend only on velocities
          }
      }
    }
  }
  else if (simplify_k==true){ 
    // Phases should depend only on velocity, so initialize them once here, on world rank 0
    if (world_rank ==0){
      for(int i=0; i<2; i++) // real and imaginary part
        for(int j=0; j<kmax_phases; j++){
          normal_distribution<double> distribution(0, sqrt(2*M_PI*j*j + 1E-10)); // Avoid zero standard deviation
          double draw = distribution(generator);
          phases_send[j + i*kmax_phases] =draw; // Phases should depend only on velocities
        }
    }
  }
  // send to other nodes, 4th entry should be the node you send to, 5th entry is a tag (to be shared between receiver and transmitter)
  if(world_rank==0 && mpi_bool==true){
    for(int lpb=1;lpb<world_size;lpb++){ // send to every other node
      MPI_Send(&phases_send.front(), phases_send.size(), MPI_DOUBLE, lpb, 100+lpb, MPI_COMM_WORLD);
    }
  }
  // receive from node 0, 4th entry should be the node you receive from, 5th entry is a tag (to be shared between receiver and transmitter)
  if(world_rank!=0 && mpi_bool==true){
    MPI_Recv(&phases_send.front(),phases_send.size(), MPI_DOUBLE, 0, 100+world_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  cout<<"Phases check "<< phases_send[1] << " "<< phases_send[4] << " world rank " << world_rank<<endl;

  int check_index = 0;
  int time_comp_initial = time(NULL);
  #pragma omp parallel for collapse(3)
  for(int i=0; i<PointsS; i++){
    for(int j=0; j<PointsS; j++)
      for(int k=nghost; k<PointsSS+nghost; k++){
      // cout<< k + PointsSS*j + PointsS*PointsS*i<<endl;
        // I want this to understand at which point of the initialization I am in
        if(j==0 and k==nghost) {
          cout<< "In initial condition, arrived at point in x dimension "<< i << " for world rank " <<world_rank<<endl;
          // check_index++;
        }
        // f(E) is defined up to a minimum energy given by radmin, ensure you do not go below it
        double Dx = i-center_x;if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        double Dy = j-center_y;if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        double Dz = k+extrak-center_z;if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        double distance = max(Length/PointsS *sqrt( pow(Dx,2) + pow(Dy,2) + pow(Dz,2)), radmin);
        
        // ceil rounds up to nearest integer; this is the "local" maximum k, given Psi(r)
        // int kmax = min(int(PointsS/2), int(ceil(sqrt(2*profile->Psi(distance)) * Length/(2*M_PI) ) ));
        int kmax = int (ceil(sqrt(2*eddington->psi_potential(distance)) * ratiomass * Length/(2*M_PI) ) );
        // Get to the closest, rounded up, k vector allowed by the num_k splitting; it has to be divisible by skip_k
        if(kmax%skip_k != 0) {
          kmax = kmax + skip_k- (kmax % skip_k); 
        }
        double psi_point_real = 0;
        double psi_point_im = 0;
        if(simplify_k == false){
          for(int v1=-kmax; v1<kmax; v1=v1+skip_k){
            for(int v2=-kmax; v2<kmax; v2=v2+skip_k)
              for(int v3=-kmax; v3<kmax; v3=v3+skip_k){
                double vv = sqrt(v1*v1 + v2*v2 + v3*v3); // |k_f|
                double vx = Length/PointsS*(v1*i + v2*j + v3*(k+extrak)); 
                double vtilde = 2*M_PI/ratiomass/Length *vv;// taking into account the ratio_mass r, to ensure periodic boundary conditions
                double E = eddington->psi_potential(distance) - pow(vtilde,2)/2; // You have to ensure E > 0, for bound system
                // if (klow == true && (kmax==kmax_global && vv ==0)){//do the low k sum on 20 points, exploiting spherical symmetry on kf_here
                //   double fe = eddington->fE_func(E);
                //   for (int kl=0; kl < 20; kl++){
                //     double kf_here = 0.5*(kl+1)/20;
                //     double psi_point = pow(2*M_PI/Length/ratiomass,3./2)*(0.5/20)*(4*M_PI*kf_here*kf_here)*sqrt(fe);
                //     size_t index = size_t( size_t(v1/skip_k)+num_k_real) 
                //           + size_t(2*num_k_real*size_t( size_t(v2/skip_k)+num_k_real))
                //           + size_t(pow(2*size_t(num_k_real),2))*size_t( size_t(v3/skip_k)+num_k_real);
                //     psi_point_real+=psi_point*cos(phases_send[index]); 
                //     psi_point_im+=psi_point*sin(phases_send[index]); 
                //   }                
                // }
                // else{
                  if (E >0){
                    double fe = eddington->fE_func(E);
                    double psi_point = pow(2*M_PI/Length * skip_k/ratiomass, 3./2)*sqrt(fe);
                    size_t index = size_t( size_t(v1/skip_k)+num_k_real) 
                          + size_t(2*num_k_real*size_t( size_t(v2/skip_k)+num_k_real))
                          + size_t(pow(2*size_t(num_k_real),2))*size_t( size_t(v3/skip_k)+num_k_real);
                    psi_point_real +=  psi_point*cos(phases_send[index] + 2*M_PI/Length *vx);
                    psi_point_im +=  psi_point*sin(phases_send[index] + 2*M_PI/Length *vx);
                  }
                // }
              }
          }
        }
        else if (simplify_k ==true ){
          for(int vv=0; vv<kmax; vv++){
            double vtilde = 2*M_PI/Length *vv;
            double E = eddington->psi_potential(distance) - pow(vtilde,2)/2; // You have to ensure E > 0, for bound system
            if (E >0){
              double fe = eddington->fE_func(E);
              double psi_point = pow(2*M_PI/Length, 3./2)*sqrt(fe);
              psi_point_real += psi_point*phases_send[vv];
              psi_point_im += psi_point*phases_send[vv + kmax_global];
            }
          }
        }
        psi[2*fieldid][i][j][k] +=   psi_point_real;
        psi[2*fieldid+1][i][j][k] += psi_point_im;
      }
  }
  #pragma omp barrier
  cout<<"Computation time to initialize the field "<< time(NULL) - time_comp_initial << endl;
}