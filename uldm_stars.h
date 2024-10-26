#ifndef ULDM_STARS_H
#define ULDM_STARS_H
#include "uldm_mpi_2field.h"
#include "eddington.h"
#include <boost/multi_array.hpp>
#include <cmath>
#include <fstream>
// In this version, stars do not have feedback on the background potential and they do not feel themselves
// (so specify their mass is actually not required)


/**
 * @class domain_stars
 * @brief A class for handling star dynamics within a domain.
 * 
 * This class extends the domain3 class and includes additional functionality
 * for managing stars within the domain, including their initial conditions,
 * gravitational interactions, and time-stepping. Stars are treated as point masses
 * that interact with the background gravitational potential but not with each other.
 * and no backreaction on the background potential.
 */
class domain_stars: public domain3
{
  protected:
    double soften_param = deltaX/10.; ///< Softening parameter for gravitational law
    multi_array<double,2> stars; ///< Array to store star data
    ofstream stars_filename; ///< File stream for star data output
    ofstream timesfile_profile_stars; ///< File stream for star profile output (center of mass of stars)
    ofstream profilefile_stars; ///< File stream for star profile output (density)
    int num_stars_eff; ///< Effective number of stars
    multi_array<double,1> center_mass_stars; ///< center of mass x, y,z of the stars

  public:
    /**
     * @brief Constructor for domain_stars.
     * 
     * @param PS Points in space
     * @param PSS Points in space squared
     * @param L Length of the domain
     * @param nfields Number of fields
     * @param Numsteps Number of steps
     * @param DT Time step
     * @param Nout Number of outputs
     * @param Nout_profile Number of profile outputs
     * @param pointsm Points in mass
     * @param WR World rank
     * @param WS Write size
     * @param Nghost Number of ghost points
     * @param mpi_flag MPI flag
     * @param num_stars Number of stars
     */
    domain_stars(size_t PS, size_t PSS, double L, int nfields, int Numsteps, double DT, int Nout, int Nout_profile, 
                 int pointsm, int WR, int WS, int Nghost, bool mpi_flag, int num_stars):
      domain3{PS, PSS, L, nfields, Numsteps, DT, Nout, Nout_profile, 
              pointsm, WR, WS, Nghost, mpi_flag}, 
              stars(extents[num_stars][8]),
              center_mass_stars(extents[3])
      { num_stars_eff = num_stars; };

    /**
     * @brief Default constructor for domain_stars.
     */
    domain_stars() { };

    /**
     * @brief Destructor for domain_stars.
     */
    ~domain_stars() { };

    /**
     * @brief Set initial star conditions.
     * 
     * @param input Multi-array containing initial star data.
     */
    void put_initial_stars(multi_array<double, 2> input){
      stars = input;
    }

    /**
     * @brief Set the effective number of stars.
     * 
     * @param num Number of stars.
     */
    void put_numstar_eff(int num){
      num_stars_eff = num;
    }

    /**
     * @brief Calculate acceleration from a point.
     * 
     * @param i Index in x-direction.
     * @param j Index in y-direction.
     * @param k Index in z-direction.
     * @param x X-coordinate of the point.
     * @param y Y-coordinate of the point.
     * @param z Z-coordinate of the point.
     * @param softening Softening parameter.
     * @return Vector of accelerations in x, y, and z directions.
     */
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

// Cyclic for boundary conditions, for computing the center of mass from the maximum density point
// of the grid
double cyc_stars(double ar, double le){
  if(ar>le/2) ar=ar-le;
  else if(ar<-le/2) ar=ar+le;
  return ar;
}

// Find the center of mass of the stars, no mpi
void find_center_mass(int whichPsi){
  double xcm = 0;
  double ycm = 0;
  double zcm = 0;
  double mass_tot = 0;
  for(int s = 0; s <num_stars_eff; s++){
    xcm += cyc_stars(stars[s][1]-maxx[whichPsi][0]*deltaX,Length)*stars[s][0];
    ycm += cyc_stars(stars[s][2]-maxx[whichPsi][1]*deltaX,Length)*stars[s][0];
    zcm += cyc_stars(stars[s][3]-maxx[whichPsi][2]*deltaX,Length)*stars[s][0];
    mass_tot += stars[s][0];
  }
  // From the center of mass with respect to the maximum point, to the correct part of the grid
  center_mass_stars[0] = xcm/mass_tot + maxx[0][0]*deltaX;
  center_mass_stars[1] = ycm/mass_tot + maxx[0][1]*deltaX;
  center_mass_stars[2] = zcm/mass_tot + maxx[0][2]*deltaX;
}

// It computes the averaged density and energy as function of distance from center of mass of the stars
multi_array<double,2> profile_density_star_center(int whichPsi){ 
  vector<vector<double>> binned(6,vector<double>(pointsmax, 0));// Initialize vector of vectors, all with zero entries
  //auxiliary vector to count the number of points in each bin, needed for average
  vector<int> count(pointsmax, 0); // Initialize vector of dimension pointsmax, with 0s
  // maxz does not have ghost cells
  int extrak = PointsSS*world_rank -nghost;
  // #pragma omp parallel for collapse(3)
  for(int i=0;i<PointsS;i++)
    for(int j=0; j<PointsS;j++)
      for(int k=nghost; k<PointsSS+nghost;k++){
        double Dx=center_mass_stars[0]-i*deltaX; if(abs(Dx)>Length/2){Dx=abs(Dx)-Length;} // workaround which takes into account the periodic boundary conditions
        double Dy=center_mass_stars[1]-j*deltaX; if(abs(Dy)>Length/2){Dy=abs(Dy)-Length;} // periodic boundary conditions!
        double Dz=center_mass_stars[2]-(k+extrak)*deltaX; if(abs(Dz)>Length/2){Dz=abs(Dz)-Length;} // periodic boundary conditions!
        int distance=int(pow(Dx*Dx+Dy*Dy+Dz*Dz, 0.5)/deltaX);
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


    /**
     * @brief Perform a time step for the stars.
     */
    void step_stars(){
      for (int s = 0; s <num_stars_eff; s++){
        double x = stars[s][1]+ 0.5*dt*stars[s][4];
        double y = stars[s][2]+ 0.5*dt*stars[s][5];
        double z = stars[s][3]+ 0.5*dt*stars[s][6];
        double ax = 0;
        double ay = 0;
        double az = 0;
        #pragma omp parallel for collapse (3) reduction (+:ax,ay,az)
        for (int i = 0; i <PointsS; i++)
          for (int j = 0; j <PointsS; j++)
            for (int k = nghost; k <PointsSS+nghost; k++){
              vector<double> acc_from_point = acceleration_from_point(i, j, k, x, y, z, soften_param);
              ax += acc_from_point[0];
              ay += acc_from_point[1];
              az += acc_from_point[2];
            }
        #pragma omp barrier
        double ax_shared, ay_shared, az_shared;
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
        stars[s][1] = x + dt*stars[s][4]/2;
        stars[s][2] = y + dt*stars[s][5]/2;
        stars[s][3] = z + dt*stars[s][6]/2;
      }
    }

    /**
     * @brief Calculate the gradient of the potential.
     * 
     * @param arr Multi-array to store the gradient.
     */
    void gradient_potential(multi_array<double,4> &arr){
      #pragma omp parallel for collapse (4)
      for(int coord=0;coord<3; coord++)
        for (int i = 0; i <PointsS; i++)
          for (int j = 0; j <PointsS; j++)
            for (int k = nghost; k <PointsSS+nghost; k++){
              double Phi_2plus; double Phi_plus; double Phi_minus; double Phi_2minus;
              if(coord==0){
                Phi_2plus = Phi[cyc(i+2,PointsS)][j][k];
                Phi_plus = Phi[cyc(i+1,PointsS)][j][k];
                Phi_minus = Phi[cyc(i-1,PointsS)][j][k];
                Phi_2minus = Phi[cyc(i-2,PointsS)][j][k];
              }
              else if (coord==1){
                Phi_2plus = Phi[i][cyc(j+2,PointsS)][k];
                Phi_plus = Phi[i][cyc(j+1,PointsS)][k];
                Phi_minus = Phi[i][cyc(j-1,PointsS)][k];
                Phi_2minus = Phi[i][cyc(j-2,PointsS)][k];
              }
              else if (coord==2){
                Phi_2plus = Phi[i][j][cyc(k+2,PointsSS+2*nghost)];
                Phi_plus = Phi[i][j][cyc(k+1,PointsSS+2*nghost)];
                Phi_minus = Phi[i][j][cyc(k-1,PointsSS+2*nghost)];
                Phi_2minus = Phi[i][j][cyc(k-2,PointsSS+2*nghost)];
              }
              arr[coord][i][j][k] = derivative_5midpoint(Phi_2plus,Phi_plus,Phi_minus,Phi_2minus, deltaX);
            }
      #pragma omp barrier
    }

    /**
     * @brief Perform a time step for the stars using interpolating potential.
     */
    void step_stars_fourier(){
      multi_array<double,4> arr(extents[3][PointsS][PointsS][PointsS]);
      gradient_potential(arr);
      // The total mass inside the box, used to remove mirror images contributions
      double mass_box = total_mass(0);
      int num_mirrors = 0;
      
      multi_array<double,1> x_compute(extents[3]);
      multi_array<int,2> xii(extents[2][3]);
      multi_array<double,3> fii(extents[2][2][2]);
      multi_array<double,1> gradPhi(extents[3]);
      int k0_cell = PointsSS*world_rank - nghost;
      // #pragma omp parallel for collapse(1)
      for (int s = 0; s <num_stars_eff; s++){
        for(int i=0; i<3;i++){
          double coord_comp = cyc_double(stars[s][i+1]+ 0.5*dt*stars[s][i+4],Length);
          xii[0][i] = floor(coord_comp/deltaX);
          xii[1][i] = ceil(coord_comp/deltaX+1E-10); 
          x_compute[i] = coord_comp;
        }
        double ax = 0;
        double ay = 0;
        double az = 0;
        double ax_shared, ay_shared, az_shared;
        if (x_compute[2]>=deltaX*k0_cell && x_compute[2]<deltaX*(k0_cell+PointsSS)){
          for(int coord=0; coord<3; coord++){
            for(int i1 =0;i1<2;i1++)
              for(int i2 =0;i2<2;i2++)
                for(int i3 =0;i3<2;i3++){
                  fii[i1][i2][i3] = arr[coord][cyc(xii[i1][0],PointsS)][cyc(xii[i2][1],PointsS)][cyc(xii[i3][2],PointsS)];
                }
            gradPhi[coord] = linear_interp_3D(x_compute, xii,fii, deltaX);
          }
          ax=gradPhi[0]; ay=gradPhi[1]; az=gradPhi[2];
        }
        if(mpi_bool==true){
          MPI_Allreduce(&ax, &ax_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
          MPI_Allreduce(&ay, &ay_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
          MPI_Allreduce(&az, &az_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        }
        else {
          ax_shared=ax; ay_shared=ay; az_shared=az;
        }
        // Now correct for the mirror images
        for(int nmirx = -num_mirrors; nmirx< num_mirrors+1;nmirx++){
          for(int nmiry = -num_mirrors; nmiry< num_mirrors+1;nmiry++)
            for(int nmirz = -num_mirrors; nmirz< num_mirrors+1;nmirz++){
              if(nmirx==0 && nmiry==0 && nmirz==0)
                continue;
              double x_mirror = x_compute[0] -maxx[0][0]*deltaX-nmirx*Length;
              double y_mirror = x_compute[1] -maxx[0][1]*deltaX-nmiry*Length;
              double z_mirror = x_compute[2] -maxx[0][2]*deltaX-nmirz*Length;
              // cout<< "Mirror coord " << x_mirror << " " << y_mirror << " " << z_mirror << endl;
              // cout<<"Maxx coord " << maxx[0][0] * deltaX - nmirx*Length << " " << maxx[0][1] * deltaX - nmiry*Length << " " << maxx[0][2] * deltaX - nmirz*Length << endl;
              double distance = sqrt(pow(x_mirror,2) + pow(y_mirror,2) + pow(z_mirror,2) + soften_param);
              double ax_mirror = -1/(4*M_PI) * mass_box * x_mirror / pow(distance,3);
              double ay_mirror = -1/(4*M_PI) * mass_box * y_mirror / pow(distance,3);
              double az_mirror = -1/(4*M_PI) * mass_box * z_mirror / pow(distance,3);
              // cout<< "Mirror acc " << ax_shared << " " << ax_mirror << endl;
              ax_shared -= ax_mirror;
              ay_shared -= ay_mirror;
              az_shared -= az_mirror;
            }
        }

        stars[s][4]= stars[s][4] - ax_shared*dt;
        stars[s][5]= stars[s][5] - ay_shared*dt;
        stars[s][6]= stars[s][6] - az_shared*dt;
        stars[s][1] = cyc_double(x_compute[0] + dt*stars[s][4]/2, Length);
        stars[s][2] = cyc_double(x_compute[1] + dt*stars[s][5]/2, Length);
        stars[s][3] = cyc_double(x_compute[2] + dt*stars[s][6]/2, Length);
      }
      // #pragma omp barrier
    }

  /**
  * @brief Outputs the star data to a file.
  * 
  * This function computes the potential energy for each star and outputs the star data to a file.
  */
  void output_stars(){
    multi_array<double,1> x_compute(extents[3]);
    multi_array<int,2> xii(extents[2][3]);
    multi_array<double,3> fii(extents[2][2][2]);
    int k0_cell = PointsSS*world_rank - nghost;
    for (int s = 0; s <num_stars_eff; s++){
      double pot_en = 0;
      for(int i=0; i<3;i++){
        xii[0][i] = floor(stars[s][1+i]/deltaX);
        xii[1][i] = ceil((stars[s][1+i]/deltaX+1E-10));
        x_compute[i] = stars[s][i+1];
      }
      if (x_compute[2]>=deltaX*k0_cell && x_compute[2]<deltaX*(k0_cell+PointsSS)){
        for(int i1 =0;i1<2;i1++)
          for(int i2 =0;i2<2;i2++)
            for(int i3 =0;i3<2;i3++){
              fii[i1][i2][i3] = Phi[cyc(xii[i1][0],PointsS)][cyc(xii[i2][1],PointsS)][cyc(xii[i3][2]-k0_cell,PointsSS)];
            }
        pot_en = stars[s][0]*linear_interp_3D(x_compute, xii,fii, deltaX);
      }
      double pot_en_shared;  
      if(mpi_bool==true){
        MPI_Allreduce(&pot_en, &pot_en_shared, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      }
      else {
        pot_en_shared=pot_en;
      }
      stars[s][7] = pot_en_shared;
    }
    if(world_rank==0){
      print2(stars, stars_filename);
      stars_filename<<"\n"<<","<<flush;
      cout<< "Stars outputted" << endl;
    }
  }

  /**
  * @brief Reads star data from a backup file.
  * 
  * This function reads star data from a backup file and initializes the star array.
  */
  void get_star_backup(){
    string star_string_backup = outputname+"star_backup.txt";
    ifstream infile_star(star_string_backup);
    int l = 0;
    int star_i = 0;
    string temp;
    while (std::getline(infile_star, temp, ' ')) {
      double num = stod(temp);
      if(l<8){ // loop over single star feature
        stars[star_i][l] = num;
        l++;
      }
      if(l==8){ // loop over
        star_i++;
        l=0;
      }
    }
    int Nx=stars.shape()[0];
    num_stars_eff = Nx;
    for(int i =0; i<Nx; i++)
      if(stars[i][0]==0){
        num_stars_eff = i+1;
        break;
      }
  }

  /**
  * @brief Outputs star data to a backup file.
  * 
  * This function outputs the current star data to a backup file.
  */
  void out_star_backup(){
    if(world_rank==0){
      ofstream file_star_backup;
      file_star_backup.open(outputname+"star_backup.txt"); 
      file_star_backup.setf(ios_base::fixed);
      int Nx=stars.shape()[0];
      int Ny=stars.shape()[1];
      for(int s = 0;s < Nx; s++){
        for(int l = 0;l < Ny; l++){
          file_star_backup<< scientific << stars[s][l];
          if(s!=(Nx-1) || l!=(Ny-1)) //If it is not the last one
            file_star_backup<< " ";
        }
      }
      file_star_backup.close();
    }
  }

  /**
  * @brief Opens the file for star data output.
  * 
  * This function opens the file for star data output, either in write or append mode.
  */
  void open_filestars(){
    if(world_rank==0 && start_from_backup == false){
      stars_filename.open(outputname+"stars.txt");
      stars_filename<<"{";   
      stars_filename.setf(ios_base::fixed);
      
      timesfile_profile_stars.open(outputname+"times_profile_stars.txt");
      timesfile_profile_stars << "{";
      timesfile_profile_stars.setf(ios_base::fixed);
      
      profilefile_stars.open(outputname+"profile_stars.txt");
      profilefile_stars << "{";
      profilefile_stars.setf(ios_base::fixed);
    }
    else if (world_rank==0 && start_from_backup == true){
      stars_filename.open(outputname+"stars.txt", ios_base::app); 
      stars_filename.setf(ios_base::fixed);
      timesfile_profile_stars.open(outputname+"times_profile_stars.txt", ios_base::app);
      timesfile_profile_stars.setf(ios_base::fixed);
      profilefile_stars.open(outputname+"profile_stars.txt", ios_base::app);
      profilefile_stars.setf(ios_base::fixed);
    }
  }

/**
 * @brief Generates an array of random stars with positions and velocities.
 * 
 * This function generates a multi-dimensional array of stars, where each star has a mass,
 * position (x, y, z), and velocity (vx, vy, vz). The positions are randomly generated within
 * the range [0, Length], and the velocities are randomly generated within the range [-vel_max, vel_max].
 * 
 * @param vel_max The maximum absolute value for the velocity components.
 * @return A multi-dimensional array of stars, where each row represents a star and contains
 *         the following columns:
 *         - Mass (fixed at 1)
 *         - Position x
 *         - Position y
 *         - Position z
 *         - Velocity vx
 *         - Velocity vy
 *         - Velocity vz
 */
multi_array<double,2> generate_random_stars(double vel_max){
  multi_array<double,2> stars_arr(extents[num_stars_eff][8]);
  for(int i=0; i<num_stars_eff; i++){
    stars_arr[i][0] = 1; // Mass
    for(int j=1; j<4; j++){
      stars_arr[i][j] = fRand(0,Length );
    }
    for(int j=4; j<7; j++){
      stars_arr[i][j] = fRand(-vel_max,vel_max);
    }
  }
  return stars_arr;
}

  /**
  * @brief Generates stars using the Eddington procedure.
  * 
  * @param eddington Pointer to the Eddington object.
  * @param rmin Minimum radius.
  * @param rmax Maximum radius.
  * @param xmax Maximum value of the density location vector.
  * @param vel_cm Center of mass velocity.
  * @param start_star Index of the first star to generate.
  * @param end_star Index of the last star to generate.
  * @return Multi-array containing the generated star data.
  * 
  * This function generates stars using the Eddington procedure, which involves
  * random sampling from the density and velocity distributions.
  */
  void generate_stars(Eddington * eddington, double rmin, double rmax,
    vector<double> xmax, vector<double> vel_cm, int start_star, int end_star){
    // multi_array<double, 2> stars_arr(extents[num_stars_eff][8]);
    random_device rd;
    default_random_engine generator(rd()); 
    normal_distribution<double> distribution(0, 1);
    vector<double> r_arr; // Interpolating array, random uniform variable
    vector<double> cumulative_x; // Cumulative of p(x) array
    int npoints = PointsS;
    int numpoints_int = 50; // Number of points for the integration
    int npoints_vel = 500; // Number of points for the velocity cumulative computation
    r_arr.push_back(0); 
    cumulative_x.push_back(0); 
    rmin = 0.9*rmin; // Ensure that the maximum energy is never surpassed in the actual run
    for(int i=0; i< npoints+1; i++){ // avoid very first bin, which is zero
      double r = pow(10 ,(log10(rmax) -log10(rmin))/(npoints)* i + log10(rmin));
      r_arr.push_back(r);
      double bin = 0;
      double rmin_int;
      if(i==0) rmin_int = 0.1*rmin;
      else rmin_int = r_arr[i];
      for(int j=0; j< numpoints_int; j++){
        double r1 = pow(10 ,(log10(r) -log10(rmin_int))/(numpoints_int)* j + log10(rmin_int));
        double r2 =pow(10 ,(log10(r) -log10(rmin_int))/(numpoints_int)* (j+1) + log10(rmin_int));
        double dx = r2 - r1;
        double dy = eddington->profile_density(r1)*r1*r1 + eddington->profile_density(r2)*r2*r2;
        bin+= 0.5*dx*dy;
      }
      cumulative_x.push_back( cumulative_x[i] + bin*4*M_PI/eddington->profiles_massMax(rmax));
    }
      cout<< "Show x cumulative"<<endl;
      for(int ishow=0;ishow<cumulative_x.size();ishow++)
        cout<< r_arr[ishow] << " " <<cumulative_x[ishow]<<endl;
    // #pragma omp parallel for collapse(1)
    for(int i=start_star; i<end_star; i++){
      double rand = fRand(0,1);
      double x_rand = interpolant(rand, cumulative_x, r_arr);
      if(x_rand<rmin) x_rand = rmin;
      if(x_rand > rmax) x_rand = rmax - rmax*1E-8;
      vector<double> v_arr; // Interpolating array, random uniform variable
      vector<double> cumulative_v; // Cumulative of p(v|x) array
      // Avoid to go beyond vmax
      double vmax =sqrt(2*eddington->psi_potential(x_rand));
      double vmin = 0.001*vmax;
      v_arr.push_back(0); 
      cumulative_v.push_back(0);
      for(int i=0; i< npoints_vel+1; i++){ 
        double v = pow(10 ,(log10(vmax) -log10(vmin))/(npoints_vel)* i + log10(vmin));
        v_arr.push_back(v);
        double bin = 0;
        double vmin_int;
        if(i==0) vmin_int = 0.1*vmin;
        else vmin_int = v_arr[i];
        for(int j=0; j< numpoints_int; j++){
          double v1 = pow(10 ,(log10(v) -log10(vmin_int))/(numpoints_int)* j + log10(vmin_int));
          double v2 =pow(10 ,(log10(v) -log10(vmin_int))/(numpoints_int)* (j+1) + log10(vmin_int));
          double dx = v2 - v1;
          double E1 = eddington->psi_potential(x_rand)- v1*v1/2;
          double E2 = eddington->psi_potential(x_rand)- v2*v2/2;
          double dy = eddington->fE_func(E1)*v1*v1 + eddington->fE_func(E2)*v2*v2;
          bin+= 0.5*dx*dy;
        }
        cumulative_v.push_back(cumulative_v[i] + bin*4*M_PI/eddington->profile_density(x_rand));
      }
      if (cumulative_v.back() < 0.95 || cumulative_v.back() > 1.05){
        cout<< "Error in the cumulative distribution"<<endl;
        cout<< "x_rand: "<<scientific<< x_rand << " vmax: "<< vmax << " vmin: "<< vmin << " cumulative_v: "<< cumulative_v.back()<<endl;
        cout<< "Show v cumulative (final points)"<<endl;
        for(int ishow=cumulative_v.size()-10;ishow<cumulative_v.size();ishow++)
          cout<<scientific << v_arr[ishow] << " " <<cumulative_v[ishow]<<endl;
        // sleep(2);
      }
      double rand2 = fRand(0,1);
      double v_rand = interpolant(rand2, cumulative_v, v_arr);
      double star[6];
      for (int k=0;k<6;k++)
        star[k] = distribution(generator);
      double mod_x = sqrt(star[0]*star[0] +star[1]*star[1] +star[2]*star[2]);
      double mod_v = sqrt(star[3]*star[3] +star[4]*star[4] +star[5]*star[5]);
      stars[i][0]=1; // Set the mass to 1
      for (int k=1;k<4;k++){
        double value = x_rand*star[k-1]/mod_x +xmax[k-1];
        stars[i][k] = cyc_double(value,Length);
      }
      for (int k=4;k<7;k++){
        stars[i][k] = v_rand*star[k-1]/mod_v + vel_cm[k-4];
      }
    }
    // #pragma omp barrier
    cout<<fixed;
    // return stars_arr;
  }

  /**
  * @brief Generate stars on a disk using the surface density for sampling the radius and
  * radial circular velocity for sampling the velocity.
  * 
  * @param eddington Pointer to the Eddington object.
  * @param rmin Minimum radius.
  * @param rmax Maximum radius of random draw, it does not have to correspond to the maximum Eddington radius.
  * @return Multi-array containing the generated star data.
  * 
  * This function generates stars on a disk by sampling the radius using the surface density
  * and sampling the velocity using the radial circular velocity. Random eccentricity is included
  */
  multi_array<double, 2> generate_stars_disk(Eddington * eddington, double rmin, double rmax){
    multi_array<double, 2> stars_arr(extents[num_stars_eff][8]);
    // Random seed
    vector<double> r_arr; // Interpolating array, random uniform variable
    vector<double> cumulative_x; // Cumulative of p(x) array
    int npoints = PointsS;
    int numpoints_int = 100; // Number of points for the integration

    // The first bin should be zero
    r_arr.push_back(0); 
    cumulative_x.push_back(0); 
    rmin = 0.9*rmin; // Ensure that the maximum energy is never surpassed in the actual run
    for(int i=0; i< npoints+1; i++){ // avoid very first bin, which is zero
      double r = pow(10 ,(log10(rmax) -log10(rmin))/(npoints)* i + log10(rmin));
      r_arr.push_back(r);
      double bin = 0;
      double rmin_int;
      if(i==0) rmin_int = 0.1*rmin;
      else rmin_int = r_arr[i];
      for(int j=0; j< numpoints_int; j++){
        double r1 = pow(10 ,(log10(r) -log10(rmin_int))/(numpoints_int)* j + log10(rmin_int));
        double r2 =pow(10 ,(log10(r) -log10(rmin_int))/(numpoints_int)* (j+1) + log10(rmin_int));
        double dx = r2 - r1;
        // Use trapezoid integration
        double dy = eddington->profile_surface_density(r1)*r1 + eddington->profile_surface_density(r2)*r2;
        bin+= 0.5*dx*dy;
      }
      cumulative_x.push_back( cumulative_x[i] + bin*2*M_PI/eddington->profiles_massMax(rmax));
    }

    for(int i=0; i<num_stars_eff; i++){
      double rand = fRand(0,1);
      double x_rand = interpolant(rand, cumulative_x, r_arr);
      // Randomize point on the circle
      double theta= fRand(0,2*M_PI);
      // Use the potential profiles to get the circular velocity, case where
      // density is just Plummer but potential dominated by dark matter
      double vel = sqrt(eddington->profiles_massMax_pot(x_rand)/(4*M_PI*x_rand));
      // Now I want a random eccentricity, randomize vhat_phi and vhat_r
      // This gives a maximum eccentricity of 0.23
      double vhat_phi = fRand(0.9,1.1); // Should be close to 1
      double vhat_r = fRand(-0.1,0.1); // Should be close to 0
      stars_arr[i][0] = 1; // Mass
      stars_arr[i][1] = x_rand*cos(theta)+ Length/2; // x coordinate
      stars_arr[i][2] = x_rand*sin(theta)+ Length/2; // y coordinate
      stars_arr[i][3] = Length/2; // z coordinate
      stars_arr[i][4] = vel*(-vhat_phi*sin(theta) + vhat_r*cos(theta) ); // vx
      stars_arr[i][5] = vel*(vhat_phi*cos(theta) + vhat_r*sin(theta)); // vy
      stars_arr[i][6] = 0; // vz
    }
    return stars_arr;
  }


void snapshot_profile_stars(double stepCurrent){
  cout.setf(ios_base::fixed);
  if(world_rank==0)
    profilefile_stars<<"{";
  for(int l=0;l<nfields;l++){
    maxdensity[l] = find_maximum(l);
    find_center_mass(l);
    multi_array<double,2> profile = profile_density_star_center(l);
    if(world_rank==0){
      print2(profile,profilefile_stars);
      if(l<nfields-1)
        profilefile_stars<<","<<flush;
    }
  }
  if(world_rank==0){
    profilefile_stars<<"}\n" <<","<<flush;
    timesfile_profile_stars<<"{"<<scientific<<tcurrent;
      timesfile_profile_stars<<","<<scientific<<center_mass_stars[0]<<"," 
      <<center_mass_stars[1]<<"," <<center_mass_stars[2];
    timesfile_profile_stars<<"}\n"<<","<<flush;
    cout<<"Output profile results stars center"<<endl;
  }
  
}



  /**
  * @brief Solves the convection-diffusion equation.
  * 
  * This function performs the main simulation loop, solving the convection-diffusion equation
  * for the given number of steps. It handles adaptive time-stepping based on energy conservation
  * and outputs snapshots and profiles at specified intervals.
  */
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
    // Compute Phi so that I have the initial potential energy
    // I am not taking Phi from backup anymore, I am computing it from psi now
    fgrid.inputPhi(psi,nghost,nfields);                                     //inputs |psi^2|
    fgrid.calculateFT();                                              //calculates its FT, FT(|psi^2|)
    fgrid.kfactorPhi(Length);                                         //calculates -1/k^2 FT(|psi^2|)
    fgrid.calculateIFT();                                             //calculates the inverse FT
    fgrid.transferPhi(Phi,1./pow(PointsS,3));                     //transfers the result into the xytzgrid Phi and multiplies it by 1/PS^3
    if (start_from_backup == false){
      tcurrent = 0.0;
      snapshot(stepCurrent); // I want the snapshot of the initial conditions
      snapshot_profile(stepCurrent);
      snapshot_profile_stars(stepCurrent);
      exportValues(); // for backup purposes
      ofstream psi_final;
      outputfullPsi(psi_final,true,1);
      output_stars(); 
      // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
      if(world_rank==0){
        cout<<"current time = "<< tcurrent << " step " << stepCurrent << " / " << numsteps<<" dt "<<dt<<endl;
        cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
      }
      makestep(stepCurrent,dt);
      tcurrent=tcurrent+dt;
      step_stars_fourier();
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
      step_stars_fourier();
      stepCurrent=stepCurrent+1;
      double etot_current = 0;
      for(int i=0;i<nfields;i++){
        etot_current += e_kin_full1(i) + full_energy_pot(i);
      }
      if (adaptive_timestep == true){
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
      }
      
      if(stepCurrent%numoutputs==0 || stepCurrent==numsteps) {
        if (mpi_bool==true){ 
          sortGhosts(); // Should be called, to do derivatives in real space
        }
        snapshot(stepCurrent);
      }
      if(stepCurrent%numoutputs_profile==0 || stepCurrent==numsteps) {
        if (mpi_bool==true){ 
          sortGhosts(); // Should be called, to do derivatives in real space
        }
        snapshot_profile(stepCurrent);
        snapshot_profile_stars(stepCurrent);
        output_stars();
        out_star_backup();
        exportValues(); // for backup purposes
        ofstream psi_final;
        outputfullPsi(psi_final,true,1);
      }
    }
    closefiles();
    stars_filename.close();
    timesfile_profile_stars.close();
    profilefile_stars.close();
    cout<<"end"<<endl;
  }
};


#endif

