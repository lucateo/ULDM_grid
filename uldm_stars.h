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
    int num_stars_eff; ///< Effective number of stars

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
              pointsm, WR, WS, Nghost, mpi_flag}, stars(extents[num_stars][8])
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
      
      multi_array<double,1> x_compute(extents[3]);
      multi_array<int,2> xii(extents[2][3]);
      multi_array<double,3> fii(extents[2][2][2]);
      multi_array<double,1> gradPhi(extents[3]);
      int k0_cell = PointsSS*world_rank - nghost;
      for (int s = 0; s <num_stars_eff; s++){
        double x = stars[s][1]+ 0.5*dt*stars[s][4];
        double y = stars[s][2]+ 0.5*dt*stars[s][5];
        double z = stars[s][3]+ 0.5*dt*stars[s][6];
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
        stars[s][4]= stars[s][4] - ax_shared*dt;
        stars[s][5]= stars[s][5] - ay_shared*dt;
        stars[s][6]= stars[s][6] - az_shared*dt;
        stars[s][1] = cyc_double(x_compute[0] + dt*stars[s][4]/2, Length);
        stars[s][2] = cyc_double(x_compute[1] + dt*stars[s][5]/2, Length);
        stars[s][3] = cyc_double(x_compute[2] + dt*stars[s][6]/2, Length);
      }
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
    }
    else if (world_rank==0 && start_from_backup == true){
      stars_filename.open(outputname+"stars.txt", ios_base::app); 
      stars_filename.setf(ios_base::fixed);
    }
  }

  /**
  * @brief Generates stars using the Eddington procedure.
  * 
  * @param eddington Pointer to the Eddington object.
  * @param rmin Minimum radius.
  * @param rmax Maximum radius.
  * @param xmax Maximum value of the density location vector.
  * @param vel_cm Center of mass velocity.
  * @return Multi-array containing the generated star data.
  * 
  * This function generates stars using the Eddington procedure, which involves
  * random sampling from the density and velocity distributions.
  */
  multi_array<double, 2> generate_stars(Eddington * eddington, double rmin, double rmax,
    vector<double> xmax, vector<double> vel_cm){
    multi_array<double, 2> stars_arr(extents[num_stars_eff][8]);
    random_device rd;
    default_random_engine generator(rd()); 
    normal_distribution<double> distribution(0, 1);
    vector<double> r_arr; // Interpolating array, random uniform variable
    vector<double> cumulative_x; // Cumulative of p(x) array
    int npoints = PointsS;
    int numpoints_int = 50; // Number of points for the integration

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
    #pragma omp parallel for collapse(1)
    for(int i=0; i<num_stars_eff; i++){
      double rand = fRand(0,1);
      double x_rand = interpolant(rand, cumulative_x, r_arr);
      if(x_rand<rmin) x_rand = rmin;
      if(x_rand == rmax) x_rand = rmax - rmax*1E-8;
      vector<double> v_arr; // Interpolating array, random uniform variable
      vector<double> cumulative_v; // Cumulative of p(v|x) array
      double vmax =0.999999999*sqrt(2*eddington->psi_potential(x_rand));
      double vmin = 0.001*vmax;
      v_arr.push_back(0); 
      cumulative_v.push_back(0);
      for(int i=0; i< npoints+1; i++){ 
        double v = pow(10 ,(log10(vmax) -log10(vmin))/(npoints)* i + log10(vmin));
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
      double rand2 = fRand(0,1);
      double v_rand = interpolant(rand2, cumulative_v, v_arr);
      double star[6];
      for (int k=0;k<6;k++)
        star[k] = distribution(generator);
      double mod_x = sqrt(star[0]*star[0] +star[1]*star[1] +star[2]*star[2]);
      double mod_v = sqrt(star[3]*star[3] +star[4]*star[4] +star[5]*star[5]);
      stars_arr[i][0]=1; // Set the mass to 1
      for (int k=1;k<4;k++){
        double value = x_rand*star[k-1]/mod_x +xmax[k-1];
        if(value<Length)
          stars_arr[i][k] = value;
        else
          stars_arr[i][k] = value - Length;
      }
      for (int k=4;k<7;k++){
        stars_arr[i][k] = v_rand*star[k-1]/mod_v + vel_cm[k-4];
      }
    }
    #pragma omp barrier
    return stars_arr;
  }

  /**
  * @brief Generate stars on a disk using the surface density for sampling the radius and
  * radial circular velocity for sampling the velocity.
  * 
  * @param eddington Pointer to the Eddington object.
  * @param rmin Minimum radius.
  * @param rmax Maximum radius.
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
    if (start_from_backup == false){
      tcurrent = 0.0;
      // Compute Phi so that I have the initial potential energy
      fgrid.inputPhi(psi,nghost,nfields);                                     //inputs |psi^2|
      fgrid.calculateFT();                                              //calculates its FT, FT(|psi^2|)
      fgrid.kfactorPhi(Length);                                         //calculates -1/k^2 FT(|psi^2|)
      fgrid.calculateIFT();                                             //calculates the inverse FT
      fgrid.transferPhi(Phi,1./pow(PointsS,3));                     //transfers the result into the xytzgrid Phi and multiplies it by 1/PS^3
      snapshot(stepCurrent); // I want the snapshot of the initial conditions
      snapshot_profile(stepCurrent);
      exportValues(); // for backup purposes
      ofstream psi_final;
      outputfullPsi(psi_final,true,1);
      output_stars(); 
      // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
      // (Phi is not computed yet), so store the initial energy after one step
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
        output_stars();
        out_star_backup();
        exportValues(); // for backup purposes
        ofstream psi_final;
        outputfullPsi(psi_final,true,1);
      }
    }
    closefiles();
    stars_filename.close();
    cout<<"end"<<endl;
  }

};


#endif

