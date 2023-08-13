#ifndef ULDM_SIM_NFW_H
#define ULDM_SIM_NFW_H
#include "uldm_mpi_2field.h"

double nfw_potential(double r, double Rs, double normalization){
  // normalization = G \rho_0/(\gamma^2 m^2); \gamma is an adimensional rescaling parameter, which is \gamma = v_0^2 for Levkov initial waves
  double result;
  if(r !=0){result = -4*M_PI * normalization * pow(Rs, 3) / r * log(1 + r/Rs);}
  else{result = -4*M_PI * normalization * pow(Rs, 2);}
  return result;
}

class domain_ext_nfw: public domain3
{public:
  double Rs;
  double normalization;
  domain_ext_nfw(size_t PS,size_t PSS, double L, int nfields, int Numsteps, double DT, int Nout, int Nout_profile, 
          string Outputname, int pointsm, int WR, int WS, int Nghost, bool mpi_flag, double Rnfw, double norm_nfw):
    domain3{PS, PSS, L, nfields, Numsteps, DT, Nout, Nout_profile, 
           Outputname, pointsm, WR, WS, Nghost, mpi_flag},
    Rs(Rnfw),
    normalization(norm_nfw)  {};
  domain_ext_nfw() { }; // Default constructor
  ~domain_ext_nfw() { };

  void set_nfw_params(double rs, double const_nfw){
    Rs = rs;
    normalization = const_nfw;
  }
// psi -> exp(-i tstep d_alpha (Phi + Phi_ext)) psi; does a step forward or with the opposite sign by changing the sign of tstep
virtual void expiPhi(double tstep, double da, int whichPsi){
  double r = ratio_mass[whichPsi];
  int extrak = world_rank*PointsSS - nghost;
  #pragma omp parallel for collapse(3)
  for(size_t i=0;i<PointsS;i++)
    for(size_t j=0;j<PointsS;j++)
      for(size_t k=nghost;k<PointsSS+nghost;k++){
        double Repsi=psi[2*whichPsi][i][j][k];
        double Impsi=psi[2*whichPsi+1][i][j][k]; // caution: Phi doesn't have ghost cells so there's a relative shift in k
        // Locate the center of the nfw potential in the center of the grid
        int Dx=PointsS/2 -(int)i; if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        int Dy=PointsS/2 -(int)j; if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // periodic boundary conditions!
        int Dz=PointsS/2 -(int)k -extrak; if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // periodic boundary conditions!
        double distance=pow(Dx*Dx+Dy*Dy+Dz*Dz, 0.5) * deltaX;
        double potential = Phi[i][j][k-nghost] + nfw_potential(distance, Rs, normalization);

        psi[2*whichPsi][i][j][k]=cos(-tstep*da*r*potential)*Repsi - sin(-tstep*da*r*potential)*Impsi;   //real part
        psi[2*whichPsi+1][i][j][k]=sin(-tstep*da*r*potential)*Repsi + cos(-tstep*da*r*potential)*Impsi;   //im part
  }
  #pragma omp barrier
}
virtual void exportValues(){
  if(world_rank==0){
    runinfo.open(outputname+"runinfo.txt");
    runinfo.setf(ios_base::fixed);
    // runinfo<<"{"<<Length<<","<<tf<<","<<PointsS<<","<<numoutputs<<","<<numoutputs_profile<<"," << Rs << "," << normalization << "}"<<endl;
    runinfo<<tcurrent<<" "<<E_tot_initial<<" "<< Length<<" "<<numsteps<<" "<<PointsS<<" "
      <<numoutputs<<" "<<numoutputs_profile<< " "<< Rs<<" "<< normalization;
    for(int i=0; i<nfields; i++)
    {
      runinfo<<" "<< ratio_mass[i];
    }
    runinfo<<endl;
    runinfo.close();
  }
}
virtual double energy_pot(const int & i, const int & j, const int & k, int whichPsi){// it computes the potential energy density at grid point (i,j,k), adding the external potrential contribution
  double r = ratio_mass[whichPsi];
  int extrak = world_rank*PointsSS - nghost;
  int Dx=PointsS/2 -(int)i; if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
  int Dy=PointsS/2 -(int)j; if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // periodic boundary conditions!
  int Dz=PointsS/2 -(int)k -extrak; if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // periodic boundary conditions!
  double distance=pow(Dx*Dx+Dy*Dy+Dz*Dz, 0.5) * deltaX;
  return (pow(psi[2*whichPsi][i][j][k],2) + pow(psi[2*whichPsi+1][i][j][k],2))*(0.5*Phi[i][j][k-nghost] + nfw_potential(distance, Rs, normalization));
}

        // virtual multi_array<double,2> profile_density(double density_max){ // It computes the averaged density and energy as function of distance from soliton
        //                                                                    // It computes the averages from the center of the grid
        //                                                                    // I keep the dependence on density_max variable to not change also other functions
        //     multi_array<double,2> binned(extents[6][pointsmax]); //vector that cointains the binned density profile {d, rho(d), E_kin(d), E_pot(d), Phi(d), phase(d)}
        //     multi_array<int,1>   count(extents[pointsmax]);     //auxiliary vector to count the number of points in each bin, needed for average
        //     #pragma omp parallel for collapse(3)
        //     for(int i=0;i<PointsS;i++)
        //       for(int j=0; j<PointsS;j++)
        //         for(int k=0; k<PointsS;k++){
        //             int Dx=PointsS/2-(int)i; if(abs(Dx)>PointsS/2){Dx=abs(Dx)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        //             int Dy=PointsS/2-(int)j; if(abs(Dy)>PointsS/2){Dy=abs(Dy)-(int)PointsS;} // periodic boundary conditions!
        //             int Dz=PointsS/2-(int)k; if(abs(Dz)>PointsS/2){Dz=abs(Dz)-(int)PointsS;} // periodic boundary conditions!
        //             int distance=pow(Dx*Dx+Dy*Dy+Dz*Dz, 0.5);
        //             if(distance<pointsmax){
        //                 //adds up all the points in the 'shell' with radius = distance
        //                 binned[1][distance] = binned[1][distance] + pow(psi[0][i][j][k],2) + pow(psi[1][i][j][k],2);
        //                 binned[2][distance] = binned[2][distance] + energy_kin(i,j,k)*pow(Length/PointsS,3);
        //                 binned[3][distance] = binned[3][distance] + energy_pot(i,j,k)*pow(Length/PointsS,3);
        //                 binned[4][distance] = binned[4][distance] + Phi[i][j][k];
        //                 // binned[5][distance] = binned[5][distance] + atan2(psi[1][i][j][k], psi[0][i][j][k]);
        //                 count[distance]=count[distance]+1; //counts the points that fall in that shell
        //         }
        //     }
        //     // For the phase, I take only one ray
        //     for(int i=0;i<PointsS;i++){
        //       int distance =PointsS/2-(int)i; if(abs(distance)>PointsS/2){distance=abs(distance)-(int)PointsS;} // workaround which takes into account the periodic boundary conditions
        //       distance = abs(distance);
        //       if(distance<pointsmax){
        //           //Takes only one ray passing from the center of the possible soliton
        //           binned[5][distance] = atan2(psi[1][i][PointsS/2][PointsS/2], psi[0][i][PointsS/2][PointsS/2]);
        //       }
        //     }

        //     #pragma omp parallel for
        //     for(int lp=0;lp<pointsmax;lp++){
        //         if(count[lp]>0){
        //           binned[0][lp]=(lp+0.5)*Length/PointsS;
        //           binned[1][lp]=binned[1][lp]/count[lp];// the second component is the average density at that distance
        //           binned[2][lp]=binned[2][lp]/count[lp];// Kinetic energy
        //           binned[3][lp]=binned[3][lp]/count[lp];// Potential energy
        //           binned[4][lp]=binned[4][lp]/count[lp];// Phi (radial)
        //           // binned[5][lp]=binned[5][lp]/count[lp];// phase (radial)
        //         }
        //     }
        //     return binned;
        // }
};
#endif

