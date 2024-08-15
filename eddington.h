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
    virtual double surface_density(double r) =0;
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
    double surface_density(double r){ 
      // from astro-ph/0102341
      double xr = r/Rs;
      double result = 0;
      if (xr ==1) xr = 1.000000001; // Avoid division by zero
      double prefactor = 2*Rhos*Rs/(xr*xr-1);
      if (xr>1)
        result = prefactor*(1-1/sqrt(xr*xr-1)*atan(sqrt(xr*xr-1)));
      else if (xr<1)
        result = prefactor*(1-1/sqrt(1-xr*xr)*atanh(sqrt(1-xr*xr)));
      return result;
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
    // Surface density
    double surface_density(double r){
      return M0*pow(Rs,2)/(M_PI*pow(Rs*Rs + r*r,2));
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

class Hernquist: public Profile{
  protected:
    double Rh;
    double Mh;
    double Rmax;
    bool dimensionless_units = false; // If true, put G -> 1/(4\pi) for dimensionless units
    double G_eff = G_ASTRO;
  public:
    Hernquist(double rh, double mh, double rmax, bool dimless_bool): Profile{}, Rh(rh), Mh(mh), Rmax(rmax), dimensionless_units(dimless_bool) {
        analytic_Eddington = false; // Actually, there is an analytic version, but not implemented yet
        name_profile= "Hernquist";
        params.push_back(rh); params.push_back(mh); params.push_back(rmax);
        params_name.push_back("Rh"); params_name.push_back("Mh"); params_name.push_back("Rmax");
        if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
      };
    Hernquist() {};
    ~Hernquist() {};
    double potential(double r){
      double result; 
      result = -G_eff*Mh/(r + Rh);
      return result;
    }
    double density(double r){ // rhos in M_sun/kpc^3, rs in kpc, r in kpc
      return Mh*Rh/(2*M_PI)/(r*pow((r+Rh),3) );
    }
    double surface_density(double r){ 
      // from astro-ph/0102341
      double xr = r/Rh;
      double result = 0;
      if (xr ==1) xr = 1.000000001; // Avoid division by zero
      double prefactor = Mh/(2*M_PI*Rh*Rh)/pow(xr*xr-1,2);
      if (xr>1)
        result = prefactor*(-3+(2+xr*xr)/sqrt(xr*xr-1)*atan(sqrt(xr*xr-1)));
      else if (xr<1)
        result = prefactor*(-3+(2+xr*xr)/sqrt(1-xr*xr)*atanh(sqrt(1-xr*xr)));
      return result;
    }
    double Psi(double r){
      return -potential(r) + potential(Rmax);
    }
    double mass_rmax(double rmax){
      return Mh*rmax*rmax/pow(rmax+Rh,2);
    }
    double analytic_small_radius(double psi){//Analitic second derivative d^2\rho/dpsi^2 for small radius
      double factor1  = (Rmax+ 2*Rh)/(G_eff*G_eff*Mh*M_PI*Rh*(Rh-Rmax));
      double factor2=G_eff*Mh*Mh*pow(Rmax-Rh,3)/(M_PI*Rh*pow(G_eff*Mh*(Rmax- 2*Rh) + Rh*(Rh-Rmax)*psi,3));
      return factor1+factor2 - 3*psi/(pow(G_eff,3)*Mh*Mh*M_PI);
    }
    double analytic_d2rho_dpsi(double psi) {return -1;} // No analytic
    double analytic_fE(double psi) {return -1;} // No analytic
};

class Eddington{
  protected:
    // I need pointers if I am using pure virtual functions
    vector<Profile *> profiles_potential; // Profiles for the potential
    vector<Profile *> profiles_density; // Profiles for the density
    int numpoints;
    vector<double> d2rhodpsi2_arr; // Interpolating array
    vector<double> psi_arr; // Interpolating array, psi
    vector<double> FE_arr; // Interpolating array, fE
    
                               
  public:
    // I want to access these bools; 
    // If true, the potential is entirely sourced by the density profile_density;
    // In certain situations, I might want to discuss the case where the potential is not (at least entirely)
    // sourced by profile_density
    bool same_profile_den_pot; 
    // if true, it means there are analytic formulas for profile_potential, which will be used
    // if same_profile_den_pot is true as well   
    vector<bool> analytic_Edd;
    
    Eddington(bool same_prof_den):
      same_profile_den_pot(same_prof_den) 
      {};
    Eddington() {};
    ~Eddington() {};
    void set_profile_den(Profile * Profile) { profiles_density.push_back(Profile);}
    void set_profile_pot(Profile * Profile) { 
      profiles_potential.push_back(Profile);
      analytic_Edd.push_back(Profile->analytic_Eddington);
    }
    int profile_den_size() {return profiles_density.size();}
    int profile_pot_size() {return profiles_potential.size();}
    // Returns the i-th profile used for the total potential
    Profile* get_profile_pot(int i) {return profiles_potential[i];}
    // Returns the i-th profile used for the relevant density
    Profile* get_profile_den(int i) {return profiles_density[i];}
    double profile_density(double radius){
      double density = 0;
      for(int i=0; i<profiles_density.size();i++){
        density += profiles_density[i]->density(radius);
      }
      return density;
    }
    double psi_potential(double radius){
      double psi = 0;
      for(int i=0; i<profiles_potential.size();i++){
        psi = psi+ profiles_potential[i]->Psi(radius);
      }
      return psi;
    }
    
    double profiles_massMax(double rmax){
      double mass = 0;
      for(int i=0; i<profiles_density.size();i++){
        mass += profiles_density[i]->mass_rmax(rmax);
      }
      return mass;
    }
    
    // Uses the profiles_potential to compute the mass inside rmax
    double profiles_massMax_pot(double rmax){
      double mass = 0;
      for(int i=0; i<profiles_potential.size();i++){
        mass += profiles_potential[i]->mass_rmax(rmax);
      }
      return mass;
    }

    double profile_surface_density(double radius){
      double density_surface = 0;
      for(int i=0; i<profiles_density.size();i++){
        density_surface += profiles_density[i]->surface_density(radius);
      }
      return density_surface;
    }

    void compute_d2rho_dpsi2_arr(int numpoints, double rmin, double rmax){
      vector<double> rho_arr;
      vector<double> psiarr;
      rmin = 0.9*rmin; // Ensure that the maximum energy is never surpassed in the actual run
      // First point
      double radius = pow(10, (log10(rmax) -log10(rmin))/(numpoints-1) * (numpoints-1) +log10(rmin) ); // Log spaced
      rho_arr.push_back(profile_density(radius));
      psiarr.push_back(psi_potential(radius));
      int j = 0; // Index of the just filled array, it can be different from i, defined next
      // The remaining points
      for(int i = 1; i < numpoints;i++){
        // double radius = (1E3 - 1E-3)/numpoints * i + 1E-3; // Linear
        // Go with inverse order, ordering psi from smallest to largest
        double radius = pow(10, (log10(rmax) -log10(rmin))/(numpoints-1) * (numpoints-1-i) +log10(rmin) ); // Log spaced
        double psi = psi_potential(radius);
        if(abs((psi - psiarr[j])/(psi + psiarr[j])) > 1E-7){ // If the relative change is larger than 1e-7, accept the point (to avoid very tiny changes in psi)
          rho_arr.push_back(profile_density(radius));
          psiarr.push_back(psi_potential(radius));
          j++;
        }
      }
      d2rhodpsi2_arr = num_second_derivative(psiarr,rho_arr);
      psi_arr = psiarr;
    }
    vector<double>  get_psiarr() {return psi_arr;}
    vector<double>  get_fE_arr() {return FE_arr;}
    vector<double>  get_d2rho_arr() {return d2rhodpsi2_arr;}

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
          // Use trapezoid integration
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
      // Use analytic results only if I am dealing with a single profile, and if the potential is entirely sourced by the target density
      if((analytic_Edd[0] ==true && same_profile_den_pot==true) && analytic_Edd.size()==1) 
      {
        result=profiles_potential[0]->analytic_d2rho_dpsi(psi);
      }
      else if (psi>psi_arr[Nx-1] && same_profile_den_pot==true)
        result=profiles_potential[0]->analytic_small_radius(psi);
      else
        result=interpolant(psi, psi_arr, d2rhodpsi2_arr);
      return result;
    }
    
    double fE_func(double E){
      double result;
      int Nx = psi_arr.size();
      // Use analytic results only if I am dealing with a single profile, and if the potential is entirely sourced by the target density
      if((analytic_Edd[0] ==true && same_profile_den_pot==true) && analytic_Edd.size()==1) 
      {
        result=profiles_potential[0]->analytic_fE(E);
      }
      else if (E<=psi_arr[Nx-1] && E>=psi_arr[0])
        result=interpolant(E, psi_arr, FE_arr);
      else // If it is greater than the maximum, error
        result=-1;
      return result;
    }
  
  // Computes the array to make interpolation of f(E)
  void generate_fE_arr(int numpoints, double radmin, double radmax){
    if ((analytic_Edd[0] == false || same_profile_den_pot ==false) || analytic_Edd.size()>1){ 
      compute_d2rho_dpsi2_arr(numpoints, radmin, radmax);
      compute_fE_arr();
    }
  }

};

#endif