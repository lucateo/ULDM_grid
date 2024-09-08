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

/**
 * @brief Base class for different profiles.
 */
class Profile {
  public:
    bool analytic_Eddington; ///< Indicates if the profile has an analytic Eddington inversion.
    vector<double> params; ///< Vector containing all the parameters of the profile.
    vector<string> params_name; ///< Vector containing string with name of the parameters.
    string name_profile; ///< Name of the profile.

    Profile() {};
    ~Profile() {};

    /**
     * @brief Pure virtual function to calculate the potential at a given radius.
     * @param r Radius.
     * @return Potential at radius r.
     */
    virtual double potential(double r) = 0;

    /**
     * @brief Pure virtual function to calculate the density at a given radius.
     * @param r Radius.
     * @return Density at radius r.
     */
    virtual double density(double r) = 0;

    /**
     * @brief Pure virtual function to calculate the surface density at a given radius.
     * @param r Radius.
     * @return Surface density at radius r.
     */
    virtual double surface_density(double r) = 0;

    /**
     * @brief Pure virtual function to calculate the potential Psi at a given radius.
     * @param r Radius.
     * @return Psi at radius r.
     */
    virtual double Psi(double r) = 0;

    /**
     * @brief Pure virtual function to calculate the mass within a given radius.
     * @param rmax Maximum radius.
     * @return Mass within radius rmax.
     */
    virtual double mass_rmax(double rmax) = 0;

    /**
     * @brief Pure virtual function to calculate the analytic small radius.
     * @param psi Potential.
     * @return Analytic small radius.
     */
    virtual double analytic_small_radius(double psi) = 0;

    /**
     * @brief Pure virtual function to calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    virtual double analytic_d2rho_dpsi(double psi) = 0;

    /**
     * @brief Pure virtual function to calculate the distribution function f(E).
     * @param psi Potential.
     * @return Distribution function f(E).
     */
    virtual double analytic_fE(double psi) = 0;
};

/**
 * @brief Class for NFW profile.
 */
class NFW: public Profile {
  protected:
    double Rs; ///< Scale radius.
    double Rhos; ///< Characteristic density.
    double Rmax; ///< Maximum radius.
    bool dimensionless_units = false; ///< If true, use dimensionless units (G -> 1/(4π)).
    double G_eff = G_ASTRO; ///< Effective gravitational constant.

  public:
    /**
     * @brief Constructor for NFW profile.
     * @param rs Scale radius.
     * @param rhos Characteristic density.
     * @param rmax Maximum radius.
     * @param dimless_bool If true, use dimensionless units.
     */
    NFW(double rs, double rhos, double rmax, bool dimless_bool): Profile{}, Rs(rs), Rhos(rhos), Rmax(rmax), dimensionless_units(dimless_bool) {
        analytic_Eddington = false;
        name_profile = "NFW";
        params.push_back(rs); params.push_back(rhos); params.push_back(rmax);
        params_name.push_back("Rs"); params_name.push_back("rhos"); params_name.push_back("Rmax");
        if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
    };

    NFW() {};
    ~NFW() {};

    /**
     * @brief Calculate the potential at a given radius.
     * @param r Radius.
     * @return Potential at radius r.
     */
    double potential(double r) {
      double result; 
      if (r != 0)
        result = -4 * M_PI * G_eff * Rhos * pow(Rs, 3) / r * log(1 + r / Rs);
      else
        result = -4 * M_PI * G_eff * Rhos * pow(Rs, 2);
      return result;
    }

    /**
     * @brief Calculate the density at a given radius.
     * @param r Radius.
     * @return Density at radius r.
     */
    double density(double r) {
      double result = Rhos / (r / Rs * pow(1 + r / Rs, 2));
      return result;
    }

    /**
     * @brief Calculate the surface density at a given radius.
     * @param r Radius.
     * @return Surface density at radius r.
     */
    double surface_density(double r) {
      double xr = r / Rs;
      double result = 0;
      if (xr == 1) xr = 1.000000001; // Avoid division by zero
      double prefactor = 2 * Rhos * Rs / (xr * xr - 1);
      if (xr > 1)
        result = prefactor * (1 - 1 / sqrt(xr * xr - 1) * atan(sqrt(xr * xr - 1)));
      else if (xr < 1)
        result = prefactor * (1 - 1 / sqrt(1 - xr * xr) * atanh(sqrt(1 - xr * xr)));
      return result;
    }

    /**
     * @brief Calculate the potential Psi at a given radius.
     * @param r Radius.
     * @return Psi at radius r.
     */
    double Psi(double r) {
      return -potential(r) + potential(Rmax);
    }

    /**
     * @brief Calculate the mass within a given radius.
     * @param rmax Maximum radius.
     * @return Mass within radius rmax.
     */
    double mass_rmax(double rmax) {
      return 4 * M_PI * Rhos * pow(Rs, 3) * (log(1 + rmax / Rs) + Rs / (Rs + rmax) - 1);
    }

    /**
     * @brief Calculate the analytic small radius.
     * @param psi Potential.
     * @return Analytic small radius.
     */
    double analytic_small_radius(double psi) {
      double convert = 4 * M_PI * G_eff * pow(Rs, 2) * Rhos;
      return -convert * Rhos / pow(psi - convert - potential(Rmax), 3);
    }

    /**
     * @brief Calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    double analytic_d2rho_dpsi(double psi) {
      return -1; // No analytic
    }

    /**
     * @brief Calculate the distribution function f(E).
     * @param psi Potential.
     * @return Distribution function f(E).
     */
    double analytic_fE(double psi) {
      return -1; // No analytic
    }
};



/**
 * @brief Class for Burkert profile.
 */
class Burkert: public Profile {
  protected:
    double Rs; ///< Scale radius.
    double Rhos; ///< Characteristic density.
    double Rmax; ///< Maximum radius.
    bool dimensionless_units = false; ///< If true, use dimensionless units (G -> 1/(4π)).
    double G_eff = G_ASTRO; ///< Effective gravitational constant.

  public:
    /**
     * @brief Constructor for NFW profile.
     * @param rs Scale radius.
     * @param rhos Characteristic density.
     * @param rmax Maximum radius.
     * @param dimless_bool If true, use dimensionless units.
     */
    Burkert(double rs, double rhos, double rmax, bool dimless_bool): Profile{}, Rs(rs), Rhos(rhos), Rmax(rmax), dimensionless_units(dimless_bool) {
        analytic_Eddington = false;
        name_profile = "Burkert";
        params.push_back(rs); params.push_back(rhos); params.push_back(rmax);
        params_name.push_back("R0"); params_name.push_back("rho0"); params_name.push_back("Rmax");
        if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
    };

    Burkert() {};
    ~Burkert() {};

    /**
     * @brief Calculate the potential at a given radius.
     * @param r Radius.
     * @return Potential at radius r.
     */
    double potential(double r) {
      double xr = r / Rs;
      double result = -1./4*Rhos*pow(Rs,2)*(M_PI - 2*atan(xr) + log( pow(xr+1,2)/(xr*xr+1) ) );
      return result - mass_rmax(r) / (4*M_PI*r);
    }

    /**
     * @brief Calculate the density at a given radius.
     * @param r Radius.
     * @return Density at radius r.
     */
    double density(double r) {
      double result = Rhos *pow(Rs,3) / ((r + Rs)* ( r * r + Rs * Rs));
      return result;
    }

    /**
     * @brief Calculate the surface density at a given radius.
     * @param r Radius.
     * @return Surface density at radius r.
     */
    double surface_density(double r) {
      double xr = r / Rs;
      double result = 0;
      if (xr == 1) xr = 1.000000001; // Avoid division by zero
      if (xr > 1){
        double prefactor = Rhos * Rs / (2*sqrt( pow(xr,4) - 1));
        result = prefactor * (sqrt(xr*xr-1) *(M_PI + 2*atanh(1/(sqrt(xr*xr+1)))) - 2* sqrt(xr * xr + 1) * atan(sqrt(xr * xr - 1)));
      }
      else if (xr < 1){
        double prefactor = Rhos * Rs / (2*sqrt(1- pow(xr,2) )*(xr*xr+1));
        result = prefactor * ( -2*(xr*xr+1) *atanh(sqrt(1-xr*xr)) 
          + (2+M_PI)* sqrt(1-pow(xr,4)) * atanh(1/sqrt(xr * xr + 1)));
      }
      return result;
    }

    /**
     * @brief Calculate the potential Psi at a given radius.
     * @param r Radius.
     * @return Psi at radius r.
     */
    double Psi(double r) {
      return -potential(r) + potential(Rmax);
    }

    /**
     * @brief Calculate the mass within a given radius.
     * @param rmax Maximum radius.
     * @return Mass within radius rmax.
     */
    double mass_rmax(double rmax) {
      double x = rmax / Rs;
      return M_PI * Rhos * pow(Rs, 3) * (log((x*x+1)*pow(x+1,2)) - 2*atan(x)) ;
    }

    /**
     * @brief Calculate the analytic small radius.
     * @param psi Potential.
     * @return Analytic small radius.
     */
    double analytic_small_radius(double psi) {
      return -1; // No analytic
    }

    /**
     * @brief Calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    double analytic_d2rho_dpsi(double psi) {
      return -1; // No analytic
    }

    /**
     * @brief Calculate the distribution function f(E).
     * @param psi Potential.
     * @return Distribution function f(E).
     */
    double analytic_fE(double psi) {
      return -1; // No analytic
    }
};




/**
 * @brief Class for Plummer profile.
 */
class Plummer: public Profile {
  protected:
    double M0; ///< Mass associated.
    double Rs; ///< Scale radius.
    double Rmax; ///< Maximum radius.
    bool dimensionless_units = false; ///< If true, use dimensionless units (G -> 1/(4π)).
    double G_eff = G_ASTRO; ///< Effective gravitational constant.

  public:
    /**
     * @brief Constructor for Plummer profile.
     * @param rs Scale radius.
     * @param m0 Mass associated.
     * @param rmax Maximum radius.
     * @param dimless_bool If true, use dimensionless units.
     */
    Plummer(double rs, double m0, double rmax, bool dimless_bool): Profile{}, Rs(rs), M0(m0), Rmax(rmax), dimensionless_units(dimless_bool) {
      analytic_Eddington = true;
      name_profile = "Plummer";
      params.push_back(rs); params.push_back(m0); params.push_back(rmax);
      params_name.push_back("Rs"); params_name.push_back("M0"); params_name.push_back("Rmax");
      if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
    };

    Plummer() {};
    ~Plummer() {};

    /**
     * @brief Calculate the potential at a given radius.
     * @param r Radius.
     * @return Potential at radius r.
     */
    double potential(double r) {
      return -G_eff * M0 / sqrt(r * r + Rs * Rs);
    }

    /**
     * @brief Calculate the density at a given radius.
     * @param r Radius.
     * @return Density at radius r.
     */
    double density(double r) {
      return 3 * M0 / (4 * M_PI * pow(Rs, 3)) * pow(1 + pow(r / Rs, 2), -5. / 2);
    }

    /**
     * @brief Calculate the surface density at a given radius.
     * @param r Radius.
     * @return Surface density at radius r.
     */
    double surface_density(double r) {
      return M0 * pow(Rs, 2) / (M_PI * pow(Rs * Rs + r * r, 2));
    }

    /**
     * @brief Calculate the potential Psi at a given radius.
     * @param r Radius.
     * @return Psi at radius r.
     */
    double Psi(double r) {
      return -potential(r) + potential(Rmax);
    }

    /**
     * @brief Calculate the mass within a given radius.
     * @param rmax Maximum radius.
     * @return Mass within radius rmax.
     */
    double mass_rmax(double rmax) {
      return M0 * pow(rmax, 3) / pow(rmax * rmax + Rs * Rs, 3. / 2);
    }

    /**
     * @brief Calculate the analytic small radius.
     * @param psi Potential.
     * @return Analytic small radius.
     */
    double analytic_small_radius(double psi) {
      return analytic_d2rho_dpsi(psi);
    }

    /**
     * @brief Calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    double analytic_d2rho_dpsi(double psi) {
      double phimax = potential(Rmax);
      return 15 * Rs * Rs / (pow(G_eff, 5) * pow(M0, 4) * M_PI) * pow(psi - phimax, 3);
    }

    /**
     * @brief Calculate the distribution function f(E).
     * @param psi Potential.
     * @return Distribution function f(E).
     */
    double analytic_fE(double psi) {
      double phimax = potential(Rmax);
      double prefact = 3 * sqrt(psi) * Rs * Rs / (7 * sqrt(2) * pow(M_PI, 3) * pow(M0, 4) * pow(G_eff, 5));
      double result = prefact * (16 * pow(psi, 3) - 56 * psi * psi * phimax + 70 * psi * phimax * phimax - 35 * pow(phimax, 3));
      return result;
    }
};


/**
 * @brief Class for Hernquist profile.
 */
class Hernquist: public Profile {
  protected:
    double Rh; ///< Scale radius.
    double Mh; ///< Total mass.
    double Rmax; ///< Maximum radius.
    bool dimensionless_units = false; ///< If true, use dimensionless units (G -> 1/(4π)).
    double G_eff = G_ASTRO; ///< Effective gravitational constant.

  public:
    /**
     * @brief Constructor for Hernquist profile.
     * @param rh Scale radius.
     * @param mh Total mass.
     * @param rmax Maximum radius.
     * @param dimless_bool If true, use dimensionless units.
     */
    Hernquist(double rh, double mh, double rmax, bool dimless_bool): Profile{}, Rh(rh), Mh(mh), Rmax(rmax), dimensionless_units(dimless_bool) {
        analytic_Eddington = false; ///< Actually, there is an analytic version, but not implemented yet.
        name_profile = "Hernquist";
        params.push_back(rh); params.push_back(mh); params.push_back(rmax);
        params_name.push_back("Rh"); params_name.push_back("Mh"); params_name.push_back("Rmax");
        if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
    };

    Hernquist() {};
    ~Hernquist() {};

    /**
     * @brief Calculate the potential at a given radius.
     * @param r Radius.
     * @return Potential at radius r.
     */
    double potential(double r) {
      double result; 
      result = -G_eff * Mh / (r + Rh);
      return result;
    }

    /**
     * @brief Calculate the density at a given radius.
     * @param r Radius.
     * @return Density at radius r.
     */
    double density(double r) {
      return Mh * Rh / (2 * M_PI) / (r * pow((r + Rh), 3));
    }

    /**
     * @brief Calculate the surface density at a given radius.
     * @param r Radius.
     * @return Surface density at radius r.
     */
    double surface_density(double r) {
      double xr = r / Rh;
      double result = 0;
      if (xr == 1) xr = 1.000000001; // Avoid division by zero
      double prefactor = Mh / (2 * M_PI * Rh * Rh) / pow(xr * xr - 1, 2);
      if (xr > 1)
        result = prefactor * (-3 + (2 + xr * xr) / sqrt(xr * xr - 1) * atan(sqrt(xr * xr - 1)));
      else if (xr < 1)
        result = prefactor * (-3 + (2 + xr * xr) / sqrt(1 - xr * xr) * atanh(sqrt(1 - xr * xr)));
      return result;
    }

    /**
     * @brief Calculate the potential Psi at a given radius.
     * @param r Radius.
     * @return Psi at radius r.
     */
    double Psi(double r) {
      return -potential(r) + potential(Rmax);
    }

    /**
     * @brief Calculate the mass within a given radius.
     * @param rmax Maximum radius.
     * @return Mass within radius rmax.
     */
    double mass_rmax(double rmax) {
      return Mh * rmax * rmax / pow(rmax + Rh, 2);
    }

    /**
     * @brief Calculate the analytic small radius.
     * @param psi Potential.
     * @return Analytic small radius.
     */
    double analytic_small_radius(double psi) {
      double factor1 = (Rmax + 2 * Rh) / (G_eff * G_eff * Mh * M_PI * Rh * (Rh - Rmax));
      double factor2 = G_eff * Mh * Mh * pow(Rmax - Rh, 3) / (M_PI * Rh * pow(G_eff * Mh * (Rmax - 2 * Rh) + Rh * (Rh - Rmax) * psi, 3));
      return factor1 + factor2 - 3 * psi / (pow(G_eff, 3) * Mh * Mh * M_PI);
    }

    /**
     * @brief Calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    double analytic_d2rho_dpsi(double psi) {
      return -1; // No analytic
    }

    /**
     * @brief Calculate the distribution function f(E).
     * @param psi Potential.
     * @return Distribution function f(E).
     */
    double analytic_fE(double psi) {
      return -1; // No analytic
    }
};

/**
 * @brief Class for a generic profile, to which you feed the radially averaged density profile.
 */
class Generic: public Profile {
  protected:
    vector<double> Density_rad; ///< The radially averaged density profile.
    vector<double> Radius; ///< The radius corresponding to Density_rad.
    vector<double> Potential; ///< The potential, array for interpolation, which is computed by integrating the density profile.
    double Rmax; ///< Maximum radius.
    bool dimensionless_units = false; ///< If true, use dimensionless units (G -> 1/(4π)).
    double G_eff = G_ASTRO; ///< Effective gravitational constant.
    double dr; ///< Delta r spacing, assumed to be constant.

  public:
    /**
     * @brief Constructor for Generic profile.
     * @param density_rad The radially averaged density profile.
     * @param radius The radius corresponding to density_rad.
     * @param rmax Maximum radius.
     * @param dimless_bool If true, use dimensionless units.
     */
    Generic(vector<double> &density_rad, vector<double> &radius, double rmax, bool dimless_bool): Profile{}, Rmax(rmax), dimensionless_units(dimless_bool) {
      analytic_Eddington = false;
      name_profile = "Generic";
      Density_rad = density_rad;
      Radius = radius;
      dr = Radius[1] - Radius[0];
      for(int i = 0; i < Radius.size(); i++) {
        double result = 0;
        int j_integrand = 0;
        while(Radius[j_integrand] <= Radius[i]) {
          result += Density_rad[j_integrand] * 4 * M_PI * Radius[j_integrand] * Radius[j_integrand] * dr;
          j_integrand++;
          // Make sure that you don't go beyond the last element
          if(j_integrand >= Radius.size()) break;
        }
        result = -result / (4 * M_PI * Radius[i]);
        if(j_integrand < Radius.size()) {
          while (Radius[j_integrand] <= Radius.back()) {
            result += -Density_rad[j_integrand] * Radius[j_integrand] * dr;
            j_integrand++;
            // Make sure that you don't go beyond the last element
            if(j_integrand >= Radius.size()) break;
          }
        }
        Potential.push_back(result);
      }
      if(dimensionless_units == true) G_eff = 1 /(4*M_PI);
    };

    Generic() {};
    ~Generic() {};

    /**
     * @brief Calculate the potential at a given radius.
     * @param r Radius.
     * @return Potential at radius r.
     */
    double potential(double r) {
      return interpolant(r, Radius, Potential);
    }

    /**
     * @brief Calculate the density at a given radius.
     * @param r Radius.
     * @return Density at radius r.
     */
    double density(double r) {
      return interpolant(r, Radius, Density_rad);
    }

    /**
     * @brief Calculate the surface density at a given radius.
     * @param r Radius.
     * @return Surface density at radius r.
     */
    double surface_density(double r) {
      double result = 0;
      for (int i = 0; i < Radius.size(); i++) {
        double rad_compute = sqrt(r * r + Radius[i] * Radius[i]); 
        if (rad_compute < Rmax) {
          double density_integrand = interpolant(rad_compute, Radius, Density_rad);
          result += 2 * 2 * M_PI * density_integrand * dr;
        }
      }
      return result;
    }

    /**
     * @brief Calculate the potential Psi at a given radius.
     * @param r Radius.
     * @return Psi at radius r.
     */
    double Psi(double r) {
      return -potential(r) + potential(Rmax);
    }

    /**
     * @brief Calculate the mass within a given radius.
     * @param rmax Maximum radius.
     * @return Mass within radius rmax.
     */
    double mass_rmax(double rmax) {
      double mass = 0;
      for(int i = 0; i < Radius.size(); i++) {
        if(Radius[i] < rmax) {
          mass += Density_rad[i] * 4 * M_PI * Radius[i] * Radius[i] * dr;
        } else {
          break;
        }
      }
      return mass;
    }

    /**
     * @brief Calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    double analytic_d2rho_dpsi(double psi) {
      return -1; // No analytic
    }

    /**
     * @brief Calculate the analytic small radius.
     * @param psi Potential.
     * @return Analytic small radius.
     */
    double analytic_small_radius(double psi) {
      return -1; // No analytic
    }

    /**
     * @brief Calculate the distribution function f(E).
     * @param psi Potential.
     * @return Distribution function f(E).
     */
    double analytic_fE(double psi) {
      return -1; // No analytic
    }
};


/**
 * @brief Class for Eddington inversion.
 */
class Eddington {
  protected:
    vector<Profile *> profiles_potential; ///< Profiles for the potential.
    vector<Profile *> profiles_density; ///< Profiles for the density.
    int numpoints; ///< Number of points for interpolation.
    vector<double> d2rhodpsi2_arr; ///< Interpolating array for the second derivative of density with respect to potential.
    vector<double> psi_arr; ///< Interpolating array for potential.
    vector<double> FE_arr; ///< Interpolating array for distribution function f(E).

  public:
    bool same_profile_den_pot; ///< If true, the potential is entirely sourced by the density profile_density.
    vector<bool> analytic_Edd; ///< If true, it means there are analytic formulas for profile_potential, which will be used if same_profile_den_pot is true as well.

    /**
     * @brief Constructor for Eddington class.
     * @param same_prof_den If true, the potential is entirely sourced by the density profile_density.
     */
    Eddington(bool same_prof_den) : same_profile_den_pot(same_prof_den) {}

    /**
     * @brief Default constructor for Eddington class.
     */
    Eddington() {}

    /**
     * @brief Destructor for Eddington class.
     */
    ~Eddington() {}

    /**
     * @brief Set the density profile.
     * @param Profile Pointer to the density profile.
     */
    void set_profile_den(Profile *Profile) { profiles_density.push_back(Profile); }

    /**
     * @brief Set the potential profile.
     * @param Profile Pointer to the potential profile.
     */
    void set_profile_pot(Profile *Profile) {
      profiles_potential.push_back(Profile);
      analytic_Edd.push_back(Profile->analytic_Eddington);
    }

    /**
     * @brief Get the size of the density profiles.
     * @return Size of the density profiles.
     */
    int profile_den_size() { return profiles_density.size(); }

    /**
     * @brief Get the size of the potential profiles.
     * @return Size of the potential profiles.
     */
    int profile_pot_size() { return profiles_potential.size(); }

    /**
     * @brief Get the i-th potential profile.
     * @param i Index of the profile.
     * @return Pointer to the i-th potential profile.
     */
    Profile* get_profile_pot(int i) { return profiles_potential[i]; }

    /**
     * @brief Get the i-th density profile.
     * @param i Index of the profile.
     * @return Pointer to the i-th density profile.
     */
    Profile* get_profile_den(int i) { return profiles_density[i]; }

    /**
     * @brief Calculate the total density at a given radius.
     * @param radius Radius.
     * @return Total density at the given radius.
     */
    double profile_density(double radius) {
      double density = 0;
      for (int i = 0; i < profiles_density.size(); i++) {
        density += profiles_density[i]->density(radius);
      }
      return density;
    }

    /**
     * @brief Calculate the total potential at a given radius.
     * @param radius Radius.
     * @return Total potential at the given radius.
     */
    double psi_potential(double radius) {
      double psi = 0;
      for (int i = 0; i < profiles_potential.size(); i++) {
        psi += profiles_potential[i]->Psi(radius);
      }
      return psi;
    }

    /**
     * @brief Calculate the total mass within a given radius.
     * @param rmax Maximum radius.
     * @return Total mass within the given radius.
     */
    double profiles_massMax(double rmax) {
      double mass = 0;
      for (int i = 0; i < profiles_density.size(); i++) {
        mass += profiles_density[i]->mass_rmax(rmax);
      }
      return mass;
    }

    /**
     * @brief Calculate the total mass within a given radius using potential profiles.
     * @param rmax Maximum radius.
     * @return Total mass within the given radius using potential profiles.
     */
    double profiles_massMax_pot(double rmax) {
      double mass = 0;
      for (int i = 0; i < profiles_potential.size(); i++) {
        mass += profiles_potential[i]->mass_rmax(rmax);
      }
      return mass;
    }

    /**
     * @brief Calculate the total surface density at a given radius.
     * @param radius Radius.
     * @return Total surface density at the given radius.
     */
    double profile_surface_density(double radius) {
      double density_surface = 0;
      for (int i = 0; i < profiles_density.size(); i++) {
        density_surface += profiles_density[i]->surface_density(radius);
      }
      return density_surface;
    }

    /**
     * @brief Compute the second derivative of density with respect to potential.
     * @param numpoints Number of points for interpolation.
     * @param rmin Minimum radius.
     * @param rmax Maximum radius.
     */
    void compute_d2rho_dpsi2_arr(int numpoints, double rmin, double rmax) {
      vector<double> rho_arr;
      vector<double> psiarr;
      cout << "psi max " << psi_potential(rmin) << endl;
      rmin = 0.9 * rmin; // Ensure that the maximum energy is never surpassed in the actual run
      // First point
      double radius = pow(10, (log10(rmax) - log10(rmin)) / (numpoints - 1) * (numpoints - 1) + log10(rmin)); // Log spaced
      rho_arr.push_back(profile_density(radius));
      psiarr.push_back(psi_potential(radius));
      int j = 0; // Index of the just filled array, it can be different from i, defined next
      // The remaining points
      for (int i = 1; i < numpoints; i++) {
        // double radius = (1E3 - 1E-3)/numpoints * i + 1E-3; // Linear
        // Go with inverse order, ordering psi from smallest to largest
        double radius = pow(10, (log10(rmax) - log10(rmin)) / (numpoints - 1) * (numpoints - 1 - i) + log10(rmin)); // Log spaced
        double psi = psi_potential(radius);
        if (abs((psi - psiarr[j]) / (psi + psiarr[j])) > 1E-10) { // If the relative change is larger than 1e-7, accept the point (to avoid very tiny changes in psi)
          rho_arr.push_back(profile_density(radius));
          psiarr.push_back(psi_potential(radius));
          j++;
        }
      }
      d2rhodpsi2_arr = num_second_derivative(psiarr, rho_arr);
      psi_arr = psiarr;
    }

    /**
     * @brief Get the array of potential values.
     * @return Array of potential values.
     */
    vector<double> get_psiarr() { return psi_arr; }

    /**
     * @brief Get the array of distribution function values.
     * @return Array of distribution function values.
     */
    vector<double> get_fE_arr() { return FE_arr; }

    /**
     * @brief Get the array of second derivative of density with respect to potential.
     * @return Array of second derivative of density with respect to potential.
     */
    vector<double> get_d2rho_arr() { return d2rhodpsi2_arr; }

    /**
     * @brief Compute the array of distribution function values.
     */
    void compute_fE_arr() {
      int Ndim = psi_arr.size();
      double Qmin = min(1e-4 * psi_arr[1], psi_arr[1]); // psi_arr[0] is zero
      vector<double> result;
      result.push_back(0); // the first bin should be zero
      for (int i = 1; i < Ndim; i++) { // avoid very first bin, which is zero
        double Qmax = sqrt(psi_arr[i]);
        double E = psi_arr[i];
        int numpoints_int = 100 + 50 * i;
        double bin = 0;
        for (int j = 0; j < numpoints_int; j++) {
          double Q1 = pow(10, (log10(Qmax) - log10(Qmin)) / numpoints_int * j + log10(Qmin));
          double Q2 = pow(10, (log10(Qmax) - log10(Qmin)) / numpoints_int * (j + 1) + log10(Qmin));
          double dx = Q2 - Q1;
          // Use trapezoid integration
          double dy = d2rho_dpsi2(E - pow(Q1, 2)) + d2rho_dpsi2(E - pow(Q2, 2));
          bin += 0.5 * dx * dy;
        }
        result.push_back(bin * 2 / (M_PI * M_PI * sqrt(8)));
      }
      FE_arr = result;
    }

    /**
     * @brief Calculate the second derivative of density with respect to potential.
     * @param psi Potential.
     * @return Second derivative of density with respect to potential.
     */
    double d2rho_dpsi2(double psi) {
      double result;
      int Nx = psi_arr.size();
      // Use analytic results only if I am dealing with a single profile, and if the potential is entirely sourced by the target density
      if ((analytic_Edd[0] == true && same_profile_den_pot == true) && analytic_Edd.size() == 1) {
        result = profiles_potential[0]->analytic_d2rho_dpsi(psi);
      } else if (psi > psi_arr[Nx - 1] && same_profile_den_pot == true) {
        result = profiles_potential[0]->analytic_small_radius(psi);
      } else {
        result = interpolant(psi, psi_arr, d2rhodpsi2_arr);
      }
      return result;
    }

    /**
     * @brief Calculate the distribution function f(E).
     * @param E Energy.
     * @return Distribution function f(E).
     */
    double fE_func(double E) {
      double result;
      int Nx = psi_arr.size();
      // Use analytic results only if I am dealing with a single profile, and if the potential is entirely sourced by the target density
      if ((analytic_Edd[0] == true && same_profile_den_pot == true) && analytic_Edd.size() == 1) {
        result = profiles_potential[0]->analytic_fE(E);
      } else if (E <= psi_arr[Nx - 1] && E >= psi_arr[0]) {
        result = interpolant(E, psi_arr, FE_arr);
        // When potential and energy profile are incompatible, it can happen
        // that f(E) is negative. In this case, set it to zero.
        if (result < 0) result = 0;
      } else { // If it is greater than the maximum, error
        result = -1;
      }
      return result;
    }
  /**
  * @brief Computes the array to make interpolation of f(E).
  * 
  * This function generates the array for the distribution function f(E) by computing
  * the second derivative of density with respect to potential if necessary, and then
  * computing the f(E) array. It also prints the E values and f(E) for verification.
  * 
  * @param numpoints Number of points for the array.
  * @param radmin Minimum radius.
  * @param radmax Maximum radius.
  */
  void generate_fE_arr(int numpoints, double radmin, double radmax) {
      if ((analytic_Edd[0] == false || same_profile_den_pot == false) || analytic_Edd.size() > 1) { 
          cout << "Computing d2rho_dpsi2 array" << endl;
          compute_d2rho_dpsi2_arr(numpoints, radmin, radmax);
          compute_fE_arr();
      }
      cout << "E values" << "\t" << "f(E)" << endl;
      for (int i = 0; i < psi_arr.size(); i++) {
          cout << scientific << psi_arr[i] << "\t" << FE_arr[i] << "\t"; 
          if (i > 0 && i < psi_arr.size() - 1) {
              cout << d2rhodpsi2_arr[i - 1] << endl;
          } else {
              cout << endl;
          }
      }
      cout << fixed;
  }
};


#endif