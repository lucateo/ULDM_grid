#ifndef EDDINGTON_H
#define EDDINGTON_H
#include "uldm_mpi_2field.h"
#include <cmath>
#include <math.h>

#define MSUN 1.989E30 // In kg
#define EV2JOULE 1.6021E-19
#define HBAR 6.62E-34 / (2* M_PI)
#define PC2METER 3.086E16
#define ARCSEC2RAD 4.84814E-6 
#define CLIGHT 3E8
#define GCONST 6.67E-11
#define YR2SEC 365.25 * 24 *3600
#define EV2KG EV2JOULE/CLIGHT/CLIGHT
// Other constants
#define HBAR_EV 4.135667696E-15 / (2*M_PI) // In eV*sec
#define M_PLANCK 1.22E19 // in GeV
#define G_ASTRO GCONST/CLIGHT/CLIGHT*MSUN/1E3/PC2METER // G_Newton in kpc/Msun

using namespace std;
using namespace boost;

class Profile{
  public:
    bool analytic_Eddington;
    vector<double> params; // Vector containing all the parameters of the profile
    vector<string> params_name; // Vector containing string with name of the parameters
    string name_profile; // Name of the profile
    Profile() {};
    ~Profile() {};
    // Sintax to define a pure virtual function, to be declared afterwards
    virtual double potential(double r) =0;    
    virtual double density(double r) =0;
    virtual double Psi(double r) =0;
    virtual double mass_rmax(double rmax) =0;
    virtual double analytic_small_radius(double psi) = 0;
    virtual double analytic_d2rho_dpsi(double psi) = 0;
    virtual double analytic_fE(double psi) = 0;
};

class NFW: public Profile{
  protected:
    double Rs;
    double Rhos;
    double Rmax;
    bool dimensionless_units = false; // If true, put G -> 1/(4\pi) for dimensionless units
    double G_eff = G_ASTRO;
  public:
    NFW(double rs, double rhos, double rmax, bool dimless_bool): Profile{}, Rs(rs), Rhos(rhos), Rmax(rmax), dimensionless_units(dimless_bool) {
        analytic_Eddington = false;
        name_profile= "NFW";
        params.push_back(rs); params.push_back(rhos); params.push_back(rmax);
        params_name.push_back("Rs"); params_name.push_back("rhos"); params_name.push_back("Rmax");
        if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
      };
    NFW() {};
    ~NFW() {};
    double potential(double r){
      double result; 
      if (r!=0)
        result = -4*M_PI*G_eff*Rhos*pow(Rs,3)/r * log(1 + r/Rs);
      else
        result= -4*M_PI*G_eff*Rhos*pow(Rs,2);
      return result;
    }
    double density(double r){ // rhos in M_sun/kpc^3, rs in kpc, r in kpc
      return Rhos*pow(Rs,3)/(r*pow((r+Rs),2) );
    }
    double Psi(double r){
      return -potential(r) + potential(Rmax);
    }
    double mass_rmax(double rmax){
      return 4*M_PI*Rhos*pow(Rs,3)*(log(1+rmax/Rs) + Rs/(Rs + rmax) -1);
    }
    double analytic_small_radius(double psi){//Analitic second derivative d^2\rho/dpsi^2 for small radius
      double convert = 4*M_PI*G_eff*pow(Rs,2)*Rhos;
      return -convert*Rhos/pow(psi-convert-potential(Rmax),3);
    }
    double analytic_d2rho_dpsi(double psi) {return -1;} // No analytic
    double analytic_fE(double psi) {return -1;} // No analytic
};

class Plummer: public Profile{
  protected:
    double M0; // Mass associated
    double Rs; // Scale radius
    double Rmax;
    bool dimensionless_units = false; // If true, put G -> 1/(4\pi) for dimensionless units
    double G_eff = G_ASTRO;
  public:
    Plummer(double rs, double m0, double rmax, bool dimless_bool): Profile{}, Rs(rs), M0(m0), Rmax(rmax), dimensionless_units(dimless_bool) {
      analytic_Eddington = true;
      name_profile = "Plummer";
      params.push_back(rs); params.push_back(m0); params.push_back(rmax);
      params_name.push_back("Rs"); params_name.push_back("M0"); params_name.push_back("Rmax");
      if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
    };
    Plummer() {};
    ~Plummer() {};
    double potential(double r){
      return -G_eff*M0/sqrt(r*r + Rs*Rs);
    }
    double density(double r){ // rhos in M_sun/kpc^3, rs in kpc, r in kpc
      return 3*M0/(4*M_PI*pow(Rs,3))*pow(1+pow(r/Rs,2) ,-5./2);
    }
    double Psi(double r){
      return -potential(r) + potential(Rmax);
    }
    double mass_rmax(double rmax){
      return M0*pow(rmax,3)/pow(rmax*rmax + Rs*Rs,3./2);
    }
    // double analytic_small_radius(double psi){//Analitic second derivative d^2\rho/dpsi^2 for small radius
    //   return -; 
    // }
    double analytic_d2rho_dpsi(double psi) {
      double phimax = potential(Rmax);
      return 15*Rs*Rs/(pow(G_eff,5)*pow(M0,4)*M_PI) * pow(psi-phimax,3);
    }
    double analytic_small_radius(double psi){//Analitic second derivative d^2\rho/dpsi^2 for small radius; but analytic result here is known
      return analytic_d2rho_dpsi(psi);
    }
    double analytic_fE(double psi) {
      double phimax = potential(Rmax);
      double prefact = 3*sqrt(psi)* Rs*Rs/( 7*sqrt(2)*pow(M_PI,3)*pow(M0,4)*pow(G_eff,5) );
      double result = prefact*(16 * pow(psi,3)-56*psi*psi*phimax +70*psi*phimax*phimax -35*pow(phimax,3)) ;
      return result;
    }
};

class Eddington{
  protected:
    Profile * profile; // I need pointers if I am using pure virtual functions
    int numpoints;
    vector<double> d2rhodpsi2_arr; // Interpolating array
    vector<double> psi_arr; // Interpolating array, psi
    vector<double> FE_arr; // Interpolating array, fE
  public:
    bool analytic_Eddington;// I want to access it
    Eddington(Profile * profile_):profile(profile_) {analytic_Eddington = profile->analytic_Eddington;};
    Eddington() {};
    ~Eddington() {};
    void compute_d2rho_dpsi2_arr(int numpoints, double rmin, double rmax){
      vector<double> rho_arr;
      vector<double> psiarr;
      rmin = 0.9*rmin; // Ensure that the maximum energy is never surpassed in the actual run
      double radius = pow(10, (log10(rmax) -log10(rmin))/(numpoints-1) * (numpoints-1) +log10(rmin) ); // Log spaced
      rho_arr.push_back(profile->density(radius));
      psiarr.push_back(profile->Psi(radius));
      int j = 0; // Index of the just filled array, it can be different from i, defined next
      for(int i = 1; i < numpoints;i++){
        // double radius = (1E3 - 1E-3)/numpoints * i + 1E-3; // Linear
        // Go with inverse order, ordering psi from smallest to largest
        double radius = pow(10, (log10(rmax) -log10(rmin))/(numpoints-1) * (numpoints-1-i) +log10(rmin) ); // Log spaced
        double psi = profile->Psi(radius);
        if(abs((psi - psiarr[j])/(psi + psiarr[j])) > 1E-7){ // If the relativ change is larger than 1e-7, accept the point (to avoid very tiny changes in psi)
          rho_arr.push_back(profile->density(radius));
          psiarr.push_back(profile->Psi(radius));
          j++;
        }
      }
      d2rhodpsi2_arr = num_second_derivative(psiarr,rho_arr);
      psi_arr = psiarr;
    }
    vector<double>  get_psiarr() {return psi_arr;}
    vector<double>  get_fE_arr() {return FE_arr;}
    vector<double>  get_d2rho_arr() {return d2rhodpsi2_arr;}
    Profile* get_profile() {return profile;}

    void compute_fE_arr(){
      int Ndim = psi_arr.size();
      double Qmin = min(1e-4*psi_arr[1], psi_arr[1]); // psi_arr[0] is zero
      vector<double> result;
      result.push_back(0); // the first bin should be zero
      for(int i=1; i< Ndim; i++){ // avoid very first bin, which is zero
        double Qmax = sqrt(psi_arr[i]);
        double E = psi_arr[i];
        int numpoints_int = 100+50*i;
        double bin = 0;
        for(int j=0; j< numpoints_int; j++){
          double Q1 = pow(10 ,(log10(Qmax) -log10(Qmin))/(numpoints_int)* j + log10(Qmin));
          double Q2 =pow(10 ,(log10(Qmax) -log10(Qmin))/(numpoints_int)* (j+1) + log10(Qmin));
          double dx = Q2 - Q1;
          double dy = d2rho_dpsi2(E-pow(Q1,2)) + d2rho_dpsi2(E-pow(Q2,2));
          bin+= 0.5*dx*dy;
        }
        result.push_back( bin*2/(M_PI*M_PI*sqrt(8)));
    }
    FE_arr= result;
  }

    double d2rho_dpsi2(double psi){
      double result;
      int Nx = psi_arr.size();
      if(analytic_Eddington ==true) {result=profile->analytic_d2rho_dpsi(psi);}
      else if (psi>psi_arr[Nx-1])
        result=profile->analytic_small_radius(psi);
      else
        result=interpolant(psi, psi_arr, d2rhodpsi2_arr);
      return result;
    }
    
    double fE_func(double E){
      double result;
      int Nx = psi_arr.size();
      if(analytic_Eddington ==true) {result=profile->analytic_fE(E);}
      else if (E<=psi_arr[Nx-1] && E>=psi_arr[0])
        result=interpolant(E, psi_arr, FE_arr);
      // else if (E<=psi_arr[0])
      //   result = E*FE_arr[0]/psi_arr[0];
      else // If it is greater than the maximum, error
        result=-1;
      return result;
    }
};

#endif