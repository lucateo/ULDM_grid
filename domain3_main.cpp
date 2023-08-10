#include "uldm_mpi_2field.h"

// domain3::domain3(size_t PS,size_t PSS, double L, double r_m, int Numsteps, double DT, int Nout, int Nout_profile, 
//         string Outputname, int pointsm, int WR, int WS, int Nghost) :
// psi(extents[4][PS][PS][PSS+2*Nghost]),
// Phi(extents[PS][PS][PSS])
//   {
//       nghost=Nghost;
//       ratio_mass=r_m;
//       fgrid = Fourier(PS,PSS,WR,WS); // class for Fourier trasform, defined above
//       ca=extents[8];
//       da=extents[8];
//       PointsS=PS;
//       PointsSS=PSS;
//       Length=L;
//       dt=DT;
//       // tf(Tf),
//       numsteps=Numsteps;
//       pointsmax=pointsm;
//       numoutputs=Nout;
//       numoutputs_profile=Nout_profile;
//       outputname=Outputname;
//       jumps=extents[Nout+1]; // vector whose length corresponds to the outputs in time
//       jumps_profile= extents[Nout_profile+1]; // vector whose length corresponds to the outputs in time for profile
//       world_rank=WR;
//       world_size=WS;
//     deltaX=Length/PointsS;
//     // stepping numbers, as defined in axionyx documentation
//     ca[0]=0.39225680523878;   ca[1]=0.51004341191846;   ca[2] =-0.47105338540976; ca[3]=0.06875316825251;    ca[4]=0.06875316825251;    ca[5]=-0.47105338540976;  ca[6]=0.51004341191846;   ca[7]=0.39225680523878;
//     da[0]=0.784513610477560;  da[1]=0.235573213359359;  da[2]=-1.17767998417887;  da[3]=1.3151863206839023;  da[4]=-1.17767998417887;   da[5]=0.235573213359359;  da[6]=0.784513610477560;  da[7]=0;
//   }; // constructor
        domain3::domain3(size_t PS,size_t PSS, double L, double r_m, int Numsteps, double DT, int Nout, int Nout_profile, 
            string Outputname, int pointsm, int WR, int WS, int Nghost):
          nghost(Nghost),
          psi(extents[4][PS][PS][PSS+2*Nghost]), //real and imaginary parts are stored consequently, i.e. psi[0]=real part psi1 and psi[1]=imaginary part psi1, then the same for psi2
          Phi(extents[PS][PS][PSS]),
          ratio_mass(r_m),
          fgrid(PS,PSS,WR,WS), // class for Fourier trasform, defined above
          ca(extents[8]),
          da(extents[8]),
          PointsS(PS),
          PointsSS(PSS),
          Length(L),
          dt(DT),
          // tf(Tf),
          numsteps(Numsteps),
          pointsmax(pointsm),
          numoutputs(Nout),
          numoutputs_profile(Nout_profile),
          outputname(Outputname),
          jumps(extents[Nout+1]), // vector whose length corresponds to the outputs in time
          jumps_profile(extents[Nout_profile+1]), // vector whose length corresponds to the outputs in time for profile
          world_rank(WR),
          world_size(WS)
          {
            deltaX=Length/PointsS;
            // stepping numbers, as defined in axionyx documentation
            ca[0]=0.39225680523878;   ca[1]=0.51004341191846;   ca[2] =-0.47105338540976; ca[3]=0.06875316825251;    ca[4]=0.06875316825251;    ca[5]=-0.47105338540976;  ca[6]=0.51004341191846;   ca[7]=0.39225680523878;
            da[0]=0.784513610477560;  da[1]=0.235573213359359;  da[2]=-1.17767998417887;  da[3]=1.3151863206839023;  da[4]=-1.17767998417887;   da[5]=0.235573213359359;  da[6]=0.784513610477560;  da[7]=0;
          }; // constructor
domain3::domain3() {};
domain3::~domain3() {};
void domain3::setoutputs(double t_ini){// Set the indices of the steps when output file function will be called
  for(int inde=0; inde<numoutputs+1; inde++){
    jumps[inde]=int(numsteps/numoutputs*inde);
  }
  for(int inde=0; inde<numoutputs_profile+1; inde++){
    jumps_profile[inde]=int(numsteps/numoutputs_profile*inde);
  }
}

    long double domain3::psisqmean(int whichPsi){// Computes the mean |psi|^2 of field i=0,1
      long double totV=0;
        #pragma omp parallel for collapse(3) reduction(+:totV)
        for(size_t i=0;i<PointsS;i++)
          for(size_t j=0;j<PointsS;j++)
            for(size_t k=0;k<PointsSS;k++)
              totV=totV+pow(psi[2*whichPsi][i][j][k],2)+pow(psi[2*whichPsi+1][i][j][k],2);
        #pragma omp barrier

	    long double totVshared; // total summed up across all nodes
	    MPI_Allreduce(&totV, &totVshared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        totVshared=totVshared/(PointsS*PointsS*PointsS);
        return totVshared;
      }
 
      double domain3::total_mass(int whichPsi){ // Computes the total mass
        return psisqmean(whichPsi) * pow(Length, 3);
    }

	// cycs not needed in the z direction, but do no harm
        // note k is counted including ghosts
        double domain3::energy_kin(const int & i, const int & j, const int & k, int whichPsi){// it computes the kinetic energy density at grid point (i,j,k)

         int find=2*whichPsi; // to save typing

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
          return 0.5* (pow(der_psi1im,2) + pow(der_psi1re,2) + pow(der_psi2im,2) + pow(der_psi2re,2)+ pow(der_psi3im,2) + pow(der_psi3re,2));
        }
        
        // it computes the potential energy density at grid point (i,j,k)
        // virtual for inheritance (external potential)
        // note that Psi fields have ghost points Phi doesn't, k includes ghosts
        double domain3::energy_pot(const int & i, const int & j, const int & k, int whichPsi){
            return 0.5*(pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2))*Phi[i][j][k-nghost];
        }

        double domain3::e_kin_full1(int whichPsi){//Full kinetic energy with Fourier
          double locV= pow(Length,3)*fgrid.e_kin_FT(psi,Length,nghost,whichPsi);
          double totV;
          // gather all the locV for all processes, sums them (MPI_SUM) and returns totV, but is the order correct? 
          // TODO, I changed the order
          MPI_Allreduce(&locV, &totV, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
          return totV;
        }

      long double domain3::full_energy_kin(int whichPsi){// it computes the kinetic energy of the whole box with derivatives
        long double total_energy = 0;
        #pragma omp parallel for collapse (3) reduction(+:total_energy)
        for(int i=0;i<PointsS;i++)
          for(int j=0; j<PointsS;j++)
            for(int k=nghost; k<PointsSS+nghost;k++){
              total_energy = total_energy + energy_kin(i,j,k,whichPsi);
            }
        long double total_energy_shared; // total summed up across all nodes
        MPI_Allreduce(&total_energy, &total_energy_shared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
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
        long double total_energy_shared; // total summed up across all nodes
        MPI_Allreduce(&total_energy, &total_energy_shared, 1, MPI_LONG_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
        return total_energy_shared*pow(Length/PointsS,3);
      }


	double domain3::find_maximum(int whichPsi){ // Sets maxx, maxy, maxz equal to the maximum, it just checks for one global maximum
    double maxdensity = 0;
    for(int i=0;i<PointsS;i++)
      for(int j=0; j<PointsS;j++)
        for(int k=nghost; k<PointsSS+nghost;k++){
          double density_current = pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2);
          if (density_current > maxdensity)
          {maxx=i; maxy=j; maxz=k; maxdensity =density_current;}       // convention is that maxz counts inluding the ghost points
        }

    // now compare across nodes (there's probably a better way to do this, but it's ok for now)
    maxNode=0;
    if(world_rank!=0){  // collect onto node 0
      vector<float> sendup {(float) 1.0000001*maxx,(float)1.00000001* maxy,(float)1.000001*maxz,(float) maxdensity };
      MPI_Send(&sendup.front(), sendup.size(), MPI_FLOAT, 0, 9090, MPI_COMM_WORLD);
    }
    if(world_rank==0){
        for(int lpb=1;lpb<world_size;lpb++){ // recieve from every other node
            vector<float> reci(4,0); //vector to recieve data into
            MPI_Recv(&reci.front(),reci.size(), MPI_FLOAT, lpb, 9090, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            if(reci[3]> maxdensity){maxdensity=reci[3]; maxx=reci[0];  maxy=reci[1];  maxz=reci[2];maxNode=lpb; }
        }
    }

    // now correct on node 0, but now we need to tell all the nodes
    MPI_Bcast(&maxNode, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxx, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxy, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxz, 1, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Bcast(&maxdensity, 1, MPI_DOUBLE,0, MPI_COMM_WORLD);
    return maxdensity;
  }

      void domain3::sortGhosts(){
          for(int ii=0;ii<4;ii++){
            transferghosts(psi,ii, world_rank, world_size, nghost);
          }
    }
      void domain3::expiPhi(double tstep, double da, int whichPsi, double r){
      // Insert the ratio of mass for the second field
      if(whichPsi==0){r=1;}
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
          void domain3::makestep(double stepCurrent, double tstep){ // makes a step in a dt
      // loop over the 8 values of alpha
      for(int alpha=0;alpha<8;alpha++){
        //1: For any fixed value of alpha, perform the operation exp(-i c_\alpha dt k^2/2)\psi(t) in Fourier space
        for(int whichF=0;whichF<2;whichF++){
          fgrid.inputpsi(psi,nghost,whichF);
          fgrid.calculateFT();                                                  //calculates the FT
          fgrid.kfactorpsi(tstep,Length,ca[alpha],whichF,ratio_mass);                             //multiples the psi by exp(-i c_\alpha dt k^2/2)
          fgrid.calculateIFT();                                                 //calculates the inverse FT
          fgrid.transferpsi(psi,1./pow(PointsS,3),nghost,whichF);                             //divides it by 1/PS^3 (to get the correct normalizaiton of the FT)
        }
        //2: Then perform the operation exp(-i d_\alpha dt \Phi(x)) applied to the previous output
        if(alpha!=7){  //da[7]=0 so no evolution here anyway
          //first, calculate Phi(x) by solving nabla^2Phi = |psi^2|-<|psi^2|> in Fourier space
//                    double psisqavg= psisqmean();                   //calculates the number |psi^2| (it should be conserved)
          fgrid.inputPhi(psi,nghost);                                     //inputs |psi^2|
          fgrid.calculateFT();                                              //calculates its FT, FT(|psi^2|)
          fgrid.kfactorPhi(Length);                                         //calculates -1/k^2 FT(|psi^2|)
          fgrid.calculateIFT();                                             //calculates the inverse FT
          fgrid.transferPhi(Phi,1./pow(PointsS,3));                     //transfers the result into the xytzgrid Phi and multiplies it by 1/PS^3
          //second, multiply the previous output of psi by exp(-i d_\alpha dt \Phi(x)+Phi_ext), in coordinate space
          expiPhi(dt, da[alpha],0,ratio_mass);
          expiPhi(dt, da[alpha],1,ratio_mass);
        }
      }
      long double psisqmean0 = psisqmean(0);
      long double psisqmean1 = psisqmean(1);
      if(world_rank==0){ cout<<"mean value of field 0 "<<psisqmean0<<"  and field 1 "<<psisqmean1<<endl;}
    }

    void domain3::solveConvDif(){
      int beginning=time(NULL);
      if (start_from_backup == true)
        openfiles_backup();
      else
        openfiles();
      setoutputs();
      int stepCurrent=0;
      if (start_from_backup == false){
        tcurrent = 0;
        
        snapshot(stepCurrent); // I want the snapshot of the initial conditions
        snapshot_profile(stepCurrent); 
        
        // First step, I need its total energy (for adaptive time step, the Energy at 0 does not have the potential energy
        // (Phi is not computed yet), so store the initial energy after one step
        if(world_rank==0){
          cout<<"current time = "<< tcurrent << " step " << stepCurrent <<endl;
          cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
        }
        makestep(stepCurrent,dt);
        tcurrent=tcurrent+dt;
        stepCurrent=stepCurrent+1;
        E_tot_initial = e_kin_full1(0) + full_energy_pot(0) + e_kin_full1(1) + full_energy_pot(1);
      }
      else if (start_from_backup == true){
        ifstream infile(outputname+"runinfo.txt");
        string temp;
        size_t i = 0;
        while (std::getline(infile, temp, ' ') && i<1){ // convoluted way to read just the first character
          tcurrent = stod(temp);
          i++;
        }
        E_tot_initial = e_kin_full1(0) + full_energy_pot(0)+ e_kin_full1(1) + full_energy_pot(1);
      }

    while(stepCurrent<=numsteps){
      if(world_rank==0){
        cout<<"current time = "<< tcurrent  << " step " << stepCurrent <<endl;
        cout<<"elapsed computing time (s) = "<< time(NULL)-beginning<<endl;
      }
      makestep(stepCurrent,dt);
      tcurrent=tcurrent+dt;
      stepCurrent=stepCurrent+1;
      for(int index=0;index<=numoutputs;index++)
          if(stepCurrent==jumps[index]) { snapshot(stepCurrent); }
      for(int index=0;index<=numoutputs_profile;index++)
        if(stepCurrent==jumps_profile[index]) {
          snapshot_profile(stepCurrent);
          double etot_current = e_kin_full1(0) + full_energy_pot(0)+ e_kin_full1(1) + full_energy_pot(1);
          // Criterium for dropping by half the time step if energy is not conserved well enough
          if(abs(etot_current-E_tot_initial)/abs(etot_current + E_tot_initial) > 0.001 ){
            dt = dt/2;
            E_tot_initial = etot_current;
          }
          exportValues(); // for backup purposes
          ofstream phi_final;
          outputfullPhi(phi_final);
          ofstream psi_final;
          outputfullPsi(psi_final);
        }
      }
      closefiles();
      cout<<"end"<<endl;
    }

    void domain3::initial_cond_from_backup(){
      multi_array<double,1> Arr1D(extents[PointsS*PointsS*PointsSS]);
      ifstream infile(outputname+"phi_final_"+to_string(world_rank)+".txt");
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
      for(size_t i =0; i<PointsS; i++)
        for(size_t j =0; j<PointsS; j++)
          for(size_t k =0; k<PointsS; k++){
            // cout<<i<<endl;
            Phi[i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS];
          }
      infile.close();
      infile = ifstream(outputname+"psi_final_"+to_string(world_rank)+".txt");
      l = 0;
      multi_array<double,1> Arr1Dpsi(extents[PointsS*PointsS*PointsSS*4]);
      while (std::getline(infile, temp, ' ')) {
        // Add to the list of output strings
        num = stod(temp);
        Arr1Dpsi[l] = num;
        l++;
      }
      for(size_t m =0; m<4; m++)
        for(size_t i =0; i<PointsS; i++)
          for(size_t j =0; j<PointsS; j++)
            for(size_t k =0; k<PointsS; k++){
              // cout<<i<<endl;
              psi[m][i][j][k] = Arr1D[i+j*PointsS+k*PointsS*PointsS+m*PointsS*PointsS*PointsSS];
            }
      infile.close();
    }