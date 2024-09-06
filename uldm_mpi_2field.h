#ifndef ULDM_MPI_2FIELDS_H
#define ULDM_MPI_2FIELDS_H

#include <cstddef>
#include <ios>
#include<iostream>
#include<stdio.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <unistd.h>
#include <limits>
#include <random>
#include <omp.h>
#include "boost/multi_array.hpp"
#include <fftw3-mpi.h>
#include <mpi.h>

using namespace std;
using namespace boost;

/**
 * @file uldm_mpi_2field.h
 * @brief Header file containing function declarations for various utilities and MPI operations.
 */


/**
 * @brief Generate a random double between fMin and fMax.
 * 
 * @param fMin The minimum value.
 * @param fMax The maximum value.
 * @return A random double between fMin and fMax.
 */
extern double fRand(double fMin, double fMax);

/**
 * @brief Shift function for Fourier transform conventions with minus signs.
 * 
 * @param i The index.
 * @param N The total number of elements.
 * @return The shifted value.
 */
extern double shift(float i, float N);

/**
 * @brief Outputs the power spectrum for Gaussian waves initial conditions.
 * 
 * @param k2 The squared momentum.
 * @param Np The number of particles.
 * @return The power spectrum value.
 */
extern double P_spec(double k2, double Np);

/**
 * @brief Apply cyclic boundary conditions.
 * 
 * @param ar The array index.
 * @param le The length of the array.
 * @return The cyclic boundary index.
 */
extern int cyc(int ar, int le);

/**
 * @brief Apply cyclic boundary conditions for doubles.
 * 
 * @param ar The array index as a double.
 * @param le The length of the array as a double.
 * @return The cyclic boundary index as a double.
 */
extern double cyc_double(double ar, double le);

/**
 * @brief Compute the 3-point order derivative, mostly for computing kinetic energy.
 * 
 * @param f1_plus The function value at x + h.
 * @param f1_minus The function value at x - h.
 * @param f2_plus The function value at x + 2h.
 * @param f2_minus The function value at x - 2h.
 * @param f3_plus The function value at x + 3h.
 * @param f3_minus The function value at x - 3h.
 * @return The 3-point order derivative.
 */
extern double derivative_3point(double f1_plus, double f1_minus, double f2_plus, 
                                double f2_minus, double f3_plus, double f3_minus);

/**
 * @brief Compute the first order derivative with 5 points-midpoint.
 * 
 * @param f2_plus The function value at x + 2h.
 * @param f_plus The function value at x + h.
 * @param f_minus The function value at x - h.
 * @param f2_minus The function value at x - 2h.
 * @param deltaX The spacing between points.
 * @return The first order derivative.
 */
extern double derivative_5midpoint(double f2_plus, double f_plus, double f_minus, double f2_minus, double deltaX);

/**
 * @brief Perform 3D linear interpolation.
 * 
 * @param x_compute The coordinates to compute the interpolation.
 * @param xii The integer coordinates of the grid points.
 * @param fii The function values at the grid points.
 * @param deltax The spacing between grid points.
 * @return The interpolated value.
 */
extern double linear_interp_3D(multi_array<double, 1> x_compute, multi_array<int, 2> xii, 
                               multi_array<double, 3> fii, double deltax);

/**
 * @brief Compute the soliton profile in grid units.
 * 
 * @param r_c The core radius.
 * @param r The radius.
 * @param ratio The mass ratio of the field with respect to reference
 * @return The soliton profile value.
 */
extern double psi_soliton(double r_c, double r, double ratio);

/**
 * @brief Send ghost cells in a 4D grid.
 * 
 * @param grid The 4D grid.
 * @param ii The index.
 * @param world_rank The rank of the process.
 * @param world_size The size of the parallel nodes.
 * @param dir The direction of the send.
 * @param nghost The number of ghost cells.
 */
extern void sendg(multi_array<double, 4> &grid, int ii, int world_rank, int world_size, int dir, int nghost);

/**
 * @brief Receive ghost cells in a 4D grid.
 * 
 * @param grid The 4D grid.
 * @param ii The index.
 * @param world_rank The rank of the process.
 * @param world_size The size of the parallel nodes.
 * @param dir The direction of the receive.
 * @param nghost The number of ghost cells.
 */


/**
 * @brief Receive ghost cells in a 4D grid.
 * 
 * @param grid The 4D grid.
 * @param ii The index.
 * @param world_rank The rank of the process.
 * @param world_size The size of the parallel nodes
 * @param dir The direction of the receive.
 * @param nghost The number of ghost cells.
 */
extern void receiveg(multi_array<double, 4> &grid, int ii, int world_rank, int world_size, int dir, int nghost);

/**
 * @brief Transfer ghost cells in a 4D grid.
 * 
 * @param gr The 4D grid.
 * @param ii The index.
 * @param world_rank The rank of the process.
 * @param world_size The size of the parallel nodes.
 * @param nghost The number of ghost cells.
 */
extern void transferghosts(multi_array<double, 4> &gr, int ii, int world_rank, int world_size, int nghost);

/**
 * @brief Compute the numerical second derivative.
 * 
 * @param xarr The x values.
 * @param yarr The y values.
 * @return A vector containing the second derivative values.
 */
extern vector<double> num_second_derivative(vector<double> &xarr, vector<double> &yarr);

/**
 * @brief Compute the integral using the trapezoidal rule.
 * 
 * @param xarr The x values.
 * @param yarr The y values.
 * @return A multi_array containing the integral values.
 */
extern multi_array<double, 1> Integral_trapezoid(multi_array<double, 1> &xarr, multi_array<double, 1> &yarr);

/**
 * @brief Interpolate a value given x, xarr, and yarr.
 * 
 * @param x The x value to interpolate.
 * @param xarr The x values.
 * @param yarr The y values.
 * @return The interpolated value.
 */
extern double interpolant(double x, vector<double> &xarr, vector<double> &yarr);

/**
 * @brief Export data for plotting.
 * 
 * @param name The name of the output file.
 * @param xarr The x values.
 * @param yarr The y values.
 */
extern void export_for_plot(string name, vector<double> &xarr, vector<double> &yarr);

/**
 * @class Eddington
 * @brief Forward declaration of the Eddington class.
 */
class Eddington;

/**
 * @class Profile
 * @brief Forward declaration of the Profile class.
 */
class Profile;

/**
 * @brief Get the dimension of a 3D multi_array.
 * 
 * @param arr The 3D multi_array.
 * @return The dimension of the array.
 */
inline int dimension(const multi_array<double, 3> &arr) { 
  return arr.shape()[arr.num_dimensions() - 1]; 
}

/**
 * @brief Print a 2D multi_array to an output file.
 * 
 * @param v1 The 2D multi_array.
 * @param filename The output file stream.
 */
void inline print2(multi_array<double, 2> &v1, ofstream &filename) {
  int Nx = v1.shape()[0];
  int Ny = v1.shape()[1];
  filename << "{";
  for (int i = 0; i < Nx; i++) {
    filename << "{";
    for (int j = 0; j < Ny; j++) {
      filename << scientific << v1[i][j];
      // Add a comma if not the last element in the second dimension
      if (j != (Ny - 1)) { filename << ","; }
    }
    // If last element, close the array in the second dimension
    filename << "}";
    // Add a comma if not the last element in the first dimension
    if (i != (Nx - 1)) { filename << ","; }
    // Array in the first dimension is not closed, to allow for more elements
    // (e.g. if one one to build a 3D array where the first dimension is the time
    // snapshots and the last two are the grid points)
  }
}


/**
 * @brief Print a 3D multi_array to an output file in a nested format.
 * 
 * This function prints a 3D array in the output file in a nested format, making it easy to read.
 * 
 * @param v1 The 3D multi_array to print.
 * @param filename The output file stream.
 */
void inline print3(multi_array<double, 3> &v1, ofstream &filename) {
    int Nx = v1.shape()[0];
    int Ny = v1.shape()[1];
    int Nz = v1.shape()[2];
    filename << "{";
    for (int i = 0; i < Nx; i++) {
        filename << "{";
        for (int j = 0; j < Ny; j++) {
            filename << "{";
            for (int k = 0; k < Nz; k++) {
                filename << scientific << v1[i][j][k];
                // Add a comma if not the last element in the third dimension
                if (k != (Nz - 1)) {
                    filename << ",";
                }
            }
            filename << "}";
            // Add a comma if not the last element in the second dimension
            if (j != (Ny - 1)) {
                filename << ",";
            }
        }
        filename << "}";
        // Add a comma if not the last element in the first dimension
        if (i != (Nx - 1)) {
            filename << ",";
        }
    }
    // Close the array in the first dimension
    filename << " } ";
}

/**
 * @brief Print a 3D multi_array to an output file in a flat format.
 * 
 * This function prints a 3D array in the output file in a flat format, making it easy to load with C++.
 * The array is arranged in a 3*Nx format such that v1[i][j][k] goes to the i + Nx*j + Nx*Nx*k position.
 * The numbers are separated by a space, except for the last one.
 * 
 * @param v1 The 3D multi_array to print.
 * @param filename The output file stream.
 */
void inline print3_cpp(multi_array<double, 3> &v1, ofstream &filename) {
    int Nx = v1.shape()[0];
    int Ny = v1.shape()[1];
    int Nz = v1.shape()[2];
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                filename << scientific << v1[i][j][k];
                // Add a space if not the last element
                if (i != (Nx - 1) || j != (Ny - 1) || k != (Nz - 1)) {
                    filename << " ";
                }
            }
        }
    }
}

/**
 * @brief Print a 1D multi_array to an output file.
 * 
 * This function prints a 1D array in the output file in a flat format, making it easy to read.
 * 
 * @param v1 The 1D multi_array to print.
 * @param filename The output file stream.
 */
void inline print1(multi_array<double, 1> &v1, ofstream &filename) {
    int Nx = v1.shape()[0];
    filename << "{";
    for (int i = 0; i < Nx; i++) {
        filename << scientific << v1[i];
        if (i != (Nx - 1)) {
            filename << ",";
        }
    }
    filename << "}";
}


/**
 * @brief Outputs the full psi with the n fields to a file.
 * 
 * This function prints a 4D multi_array to an output file in a flat format, making it easy to load with C++.
 * The array is arranged such that v1[l][i][j][k] is printed in a single line, with elements separated by spaces.
 * 
 * @param v1 The 4D multi_array to print.
 * @param filename The output file stream.
 * @param nghost The number of ghost cells.
 * @param reduce_grid The grid reduction factor (default is 1, i.e. no reduction).
 */
void inline print4_cpp(multi_array<double, 4> &v1, ofstream &filename, int nghost, int reduce_grid = 1) {
    // Dimension along the first argument (2*nfields)
    int Nl = v1.shape()[0]; 
    int Nx = v1.shape()[1];
    int Ny = v1.shape()[2];
    // Nz contains the 2*nghost cells, relevant when mpi is used
    int Nz = v1.shape()[3]; 
    for (int l = 0; l < Nl; l++) {
        // Avoid the ghost cells
        // reduce_grid is used to reduce the grid size for output
        for (int k = nghost; k < Nz - nghost; k = k + reduce_grid) {
            for (int j = 0; j < Nx; j = j + reduce_grid) {
                for (int i = 0; i < Nx; i = i + reduce_grid) {
                    filename << scientific << v1[l][i][j][k];
                    // Add a space if not the last element
                    if (l != (Nl - 1) || k != (Nz - nghost - 1) || j != (Ny - 1) || i != (Nx - 1)) // If it is not the last one
                        filename << " ";
                }
            }
        }
    }
}

/**
 * @brief Prints a 3-dimensional array stored in a 1D vector to a file.
 * 
 * This function prints a 3D array, with dimensions specified in dims, stored in a 1D vector.
 * The array is arranged such that v1[i + j * dims[0] + k * dims[0] * dims[1]] is printed in a nested format.
 * 
 * @param v1 The 1D vector containing the 3D array data.
 * @param dims The dimensions of the 3D array.
 * @param filename The output file stream.
 */
void inline print3_1dvector(vector<double> &v1, vector<int> &dims, ofstream &filename) {
    // Number of elements along each dimension, for 3 dimensions
    int ndims = dims.size(); 
    filename << "{";
    for (size_t i = 0; i < dims[0]; i++) {
        filename << "{";
        for (size_t j = 0; j < dims[1]; j++) {
            filename << "{";
            for (size_t k = 0; k < dims[2]; k++) {
                filename << scientific << v1[i + j * dims[0] + k * dims[0] * dims[1]];
                // Add a comma if not the last element in the third dimension
                if (k != (dims[2] - 1)) {
                    filename << ",";
                }
            }
            filename << "}";
            // Add a comma if not the last element in the second dimension
            if (j != (dims[1] - 1)) {
                filename << ",";
            }
        }
        filename << "}";
        // Add a comma if not the last element in the first dimension
        if (i != (dims[0] - 1)) {
            filename << ",";
        }
    }
    filename << "}";
}



/**
 * @class Fourier
 * @brief A class for performing Fourier transforms using FFTW with MPI support.
 * 
 * This class provides methods to calculate the forward and inverse Fourier transforms,
 * as well as methods to initialize and manipulate the input data.
 */
class Fourier {
    size_t Nx; ///< Number of grid points in the linear dimension.
    size_t Nz; ///< Number of points in the short grid dimension.
    fftw_plan plan; ///< FFTW plan for forward transform.
    fftw_plan planback; ///< FFTW plan for inverse transform.
    fftw_complex *rin; ///< Pointer to the input data array for FFTW.
    ptrdiff_t alloc_local; ///< Local allocation size for FFTW.
    ptrdiff_t local_n0; ///< Local size in the first dimension for FFTW.
    ptrdiff_t local_0_start; ///< Local start index in the first dimension for FFTW.
    int world_rank; ///< MPI rank of the process.
    int world_size; ///< Total number of MPI processes.
    bool mpi_bool; ///< Flag indicating if MPI is used.

public:
    /**
     * @brief Default constructor for the Fourier class.
     */
    Fourier();

    /**
     * @brief Parameterized constructor for the Fourier class.
     * 
     * @param PS Number of grid points in the linear dimension.
     * @param PSS Number of points in the short grid dimension (when mpi is used, PSS=PS/WS).
     * @param WR MPI rank of the process.
     * @param WS Total number of MPI processes.
     * @param mpi_flag Flag indicating if MPI is used.
     */
    Fourier(size_t PS, size_t PSS, int WR, int WS, bool mpi_flag);

    /**
     * @brief Destructor for the Fourier class.
     */
    ~Fourier();

    /**
     * @brief Calculate the forward Fourier transform.
     * 
     * This method calculates the forward Fourier transform of the input data.
     */
    void calculateFT();

    /**
     * @brief Calculate the unnormalized inverse Fourier transform.
     * 
     * This method calculates the unnormalized inverse Fourier transform of the input data.
     * Remember to divide the output by Nx^3 to normalize it.
     */
    void calculateIFT();

    /**
     * @brief Initialize a sample grid for testing purposes.
     * 
     * This method initializes the input data array with a sample grid for testing.
     */
    void inputsamplegrid();

    /**
     * @brief Add random phases to the input data array.
     * 
     * This method adds random phases to the already inserted input data array.
     */
    void add_phases();

    /**
     * @brief Insert initial conditions for Gaussian waves.
     * 
     * @param Length The physical length of the grid.
     * @param Npart The number of particles.
     * @param r The radius parameter.
     * 
     * This method inserts initial conditions for Levkov waves into the input data array.
     */
    void inputSpectrum(double Length, double Npart, double r);

    /**
     * @brief Initialize a Gaussian correlated field.
     * 
     * @param A_corr The amplitude of the correlation.
     * @param l_corr The correlation length.
     * @param Length The physical length of the grid.
     * 
     * This method initializes the input data array with a Gaussian correlated field.
     */
    void input_Gauss_corr(double A_corr, double l_corr, double Length);
    
    /**
     * @brief Initialize a delta function in Fourier space.
     * 
     * @param Length The physical length of the grid.
     * @param Npart The number of particles.
     * @param r The radius parameter.
     * 
     * This method initializes the input data array with a delta function in Fourier space.
     */
    void inputDelta(double Length, double Npart, double r);

    /**
     * @brief Initialize a Heaviside function in Fourier space.
     * 
     * @param Length The physical length of the grid.
     * @param Npart The number of particles.
     * @param r The radius parameter.
     * 
     * This method initializes the input data array with a Heaviside function in Fourier space.
     */
    void inputTheta(double Length, double Npart, double r);

    /**
     * @brief Input psi on the array for FFTW, considering ghost cells.
     * 
     * @param psi The psi array.
     * @param nghost The number of ghost cells.
     * @param whichPsi Indicates which field (0 or 1).
     * 
     * This method inputs psi on the array for FFTW, considering the shift due to ghost cells.
     */
    void inputpsi(multi_array<double, 4> &psi, int nghost, int whichPsi);

    /**
     * @brief Input a 1D array into the Fourier input array.
     * 
     * @param arr1d The 1D array.
     * @param whichfield Indicates which field.
     * @param which_coordinate Indicates which coordinate.
     * 
     * This method inputs a 1D array into the Fourier input array.
     */
    void input_arr(multi_array<double, 1> &arr1d, int whichfield, int which_coordinate);

    /**
     * @brief Compute the phase factor given velocity in Fourier space.
     * 
     * @param Length The physical length of the grid.
     * @param r The radius parameter.
     * @param which_coord Indicates which coordinate.
     * 
     * This method computes the phase factor given velocity by passing in Fourier space.
     */
    void kfactor_vel(double Length, double r, int which_coord);

    /**
     * @brief Apply the -k^2 factor to psi.
     * 
     * @param tstep The time step.
     * @param Length The physical length of the grid.
     * @param calpha The c_alpha coefficient.
     * @param r The mass ratio parameter.
     * 
     * This method applies the -k^2 factor to psi, more precisely, psi->exp(-i c_alpha dt k^2/2 ) psi.
     */
    void kfactorpsi(double tstep, double Length, double calpha, double r);

    /**
     * @brief Transfer the result of Fourier transforms to psi.
     * 
     * @param psi The psi array.
     * @param factor The scaling factor.
     * @param nghost The number of ghost cells.
     * @param whichPsi Indicates which field.
     * 
     * This method transfers the result of Fourier transforms to psi.
     */
    void transferpsi(multi_array<double, 4> &psi, double factor, int nghost, int whichPsi);

    /**
     * @brief Add the result of Fourier transforms to psi.
     * 
     * @param psi The psi array.
     * @param factor The scaling factor.
     * @param nghost The number of ghost cells.
     * @param whichPsi Indicates which field.
     * 
     * This method adds the result of Fourier transforms to psi, add and not replace for stacking initial conditions.
     */
    void transferpsi_add(multi_array<double, 4> &psi, double factor, int nghost, int whichPsi);

    /**
     * @brief Input |psi|^2 to the memory that will be Fourier transformed.
     * 
     * @param psi_in The input psi array.
     * @param nghost The number of ghost cells.
     * @param nfields The number of fields.
     * 
     * This method inputs |psi|^2 to the memory that will be Fourier transformed.
     * This method is virtual for inheritance, in case one wants to use an external potential.
     */
    virtual void inputPhi(multi_array<double, 4> &psi_in, int nghost, int nfields);

    /**
     * @brief Apply the -1/k^2 factor to the output of the Fourier transform.
     * 
     * @param Length The physical length of the grid.
     * 
     * This method applies the -1/k^2 factor to the output of the Fourier transform,
     * needed for solving the Poisson equation to get Phi.
     */
    void kfactorPhi(double Length);

    /**
     * @brief Transfer the result of calculating Phi from the Poisson equation to a grid.
     * 
     * @param Phi The Phi array, gravitational potential.
     * @param factor The normalization factor.
     * 
     * This method transfers the result of calculating Phi from the Poisson equation to a grid.
     * Note the normalization factor here from the inverse Fourier transform.
     */
    void transferPhi(multi_array<double, 3> &Phi, double factor);

    /**
     * @brief Calculate the total kinetic energy using the Fourier transform.
     * 
     * @param psi The psi array.
     * @param Length The physical length of the grid.
     * @param nghost The number of ghost cells.
     * @param whichPsi Indicates which field.
     * @return The total kinetic energy.
     * 
     * This method calculates the total kinetic energy using the Fourier transform.
     * Note that this does not sum up the values on different nodes.
     */
    double e_kin_FT(multi_array<double, 4> &psi, double Length, int nghost, int whichPsi);

    /**
     * @brief Calculate the center of mass velocity using the Fourier transform.
     * 
     * @param psi The psi array.
     * @param Length The physical length of the grid.
     * @param nghost The number of ghost cells.
     * @param whichPsi Indicates which field.
     * @param coordinate The coordinate index (0, 1, or 2).
     * @return The center of mass velocity.
     * 
     * This method calculates the center of mass velocity using the Fourier transform.
     * Note that this does not sum up the values on different nodes.
     */
    double v_center_mass_FT(multi_array<double, 4> &psi, double Length, int nghost, int whichPsi, int coordinate);

    /**
     * @brief Transfer the Phi gradient to an array.
     * 
     * @param arr The array to store the Phi gradient.
     * @param which_coord The coordinate index (0, 1, or 2).
     * @param factor The normalization factor.
     * 
     * This method transfers the Phi gradient to an array.
     */
    void transfer_arr(multi_array<double, 4> &arr, int which_coord, double factor);

    /**
     * @brief Input the potential Phi, which is already stored.
     * 
     * @param Phi_in The input Phi array.
     * 
     * This method inputs the potential Phi, which is already stored.
     */
    void input_potential(multi_array<double, 3> &Phi_in);

    /**
     * @brief Compute the Fourier transform of the Phi gradients and store them in an array.
     * 
     * @param Phi_in The input Phi array.
     * @param arr The array to store the Phi gradients.
     * @param Length The physical length of the grid.
     * 
     * This method computes the Fourier transform of the Phi gradients and stores them in an array.
     */
    void kPhi_FT(multi_array<double, 3> &Phi_in, multi_array<double, 4> &arr, double Length);
};


/**
 * @class domain3
 * @brief A class for managing the computational domain for solving the problem.
 * 
 * This class builds the grid with the correct parameters and manages the domain for solving the problem.
 */
class domain3 {
protected:
    multi_array<double, 4> psi; ///< Contains the n fields.
    multi_array<double, 4> psi_backup; ///< Backup of psi, used for computing the energy spectrum if spectrum_bool is true.
    int nghost; ///< Number of ghost layers above and below in the z direction.
    bool mpi_bool; ///< Flag indicating if the domain is distributed over MPI.
    int nfields; ///< Number of fields.
    multi_array<double, 3> Phi; ///< Potential field.
    multi_array<double, 1> ratio_mass; ///< Ratio between the masses of the ULDM with respect to field 0.
    size_t PointsS; ///< Number of points in the long direction.
    size_t PointsSS; ///< Number of points in the short direction, not including ghost cells.
    double Length; ///< Physical length of the domain.
    int numsteps; ///< Number of steps for adaptive timestep case.
    double dt; ///< Initial time step.
    double deltaX; ///< Grid units grid spacing.
    int numoutputs; ///< Number of outputs for the sliced or full 3D density profile (for animation).
    int numoutputs_profile; ///< Number of outputs for the radial profiles.
    string outputname; ///< Name of the output file.
    Fourier fgrid; ///< Fourier transform object for the grid.
    multi_array<double, 1> ca; ///< Vector storing the numerical values of the coefficients to step forward.
    multi_array<double, 1> da; ///< Vector storing the numerical values of the coefficients to step forward.
    ofstream runinfo; ///< Output file for the run information (grid points, length, etc.).
    ofstream profilefile; ///< Output file for radial profiles of density, energies, etc.
    ofstream profile_sliced; ///< Output file for 2D projected profile.
    ofstream phase_slice; ///< Slice for the phase.
    ofstream timesfile_grid; ///< Output file for grid.
    ofstream timesfile_profile; ///< Output file for useful information (total energies, etc.).
    ofstream info_initial_cond; ///< Output file for initial condition details.
    ofstream spectrum_energy; ///< Output file for spectrum energy.
    bool first_initial_cond; ///< Starts from true, becomes false when an initial condition is inserted; used to change to append mode on info_initial_condition file (when false).
    int snapshotcount = 0; ///< Variable to number the snapshots if Grid3D is true.
    int reduce_grid_param = 1; ///< Parameter to reduce the grid.
    bool Grid3D = false; ///< If true, outputs the full density on the 3D grid; if false (recommended), outputs the 2D projection of the density profile.
    bool phaseGrid = false; ///< If true, outputs the phase slice passing through the center.
    bool start_from_backup = false; ///< If true, starts from the backup files.
    bool adaptive_timestep = true; ///< If true, uses adaptive timesteps.
    bool spectrum_bool = false; ///< If true, outputs the energy spectrum from Levkov.
    int pointsmax = 0; ///< Maximum number of points.
    multi_array<int, 2> maxx; ///< Location x, y, z of the max density in the grid (for a certain field).
    int maxNode = 0; ///< Node that the maximum value is on.
    multi_array<double, 1> maxdensity; ///< Max density on the grid.
    double tcurrent = 0.0; ///< Current time of simulation.
    double E_tot_initial = 0; ///< Stores the initial total energy to implement the adaptive time step.
    int world_rank; ///< MPI rank of the process.
    int world_size; ///< Total number of MPI processes.

public:
    /**
     * @brief Parameterized constructor for the domain3 class.
     * 
     * @param PS Number of points in the long direction.
     * @param PSS Number of points in the short direction.
     * @param L Physical length of the domain.
     * @param nfields Number of fields.
     * @param Numsteps Number of steps for adaptive timestep case.
     * @param DT Initial time step.
     * @param Nout Number of outputs for the density profile.
     * @param Nout_profile Number of outputs for the radial profiles.
     * @param pointsm Maximum number of points.
     * @param WR MPI rank of the process.
     * @param WS Total number of MPI processes.
     * @param Nghost Number of ghost layers.
     * @param mpi_flag Flag indicating if the domain is distributed over MPI.
     */
    domain3(size_t PS, size_t PSS, double L, int nfields, int Numsteps, double DT, int Nout, int Nout_profile, 
            int pointsm, int WR, int WS, int Nghost, bool mpi_flag);

    /**
     * @brief Default constructor for the domain3 class.
     */
    domain3();

    /**
     * @brief Destructor for the domain3 class.
     */
    ~domain3();

    /**
     * @brief Calculate the mean squared value of psi for a given field.
     * 
     * @param whichPsi The field index.
     * @return The mean squared value of psi.
     */
    long double psisqmean(int whichPsi);

    /**
     * @brief Calculate the total mass for a given field.
     * 
     * @param whichPsi The field index.
     * @return The total mass.
     */
    double total_mass(int whichPsi);

    /**
     * @brief Find the position of the center of mass for a given field.
     * 
     * @param coordinate The coordinate index (0, 1, or 2).
     * @param whichPsi The field index.
     * @return The position of the center of mass.
     */
    double x_center_mass(int coordinate, int whichPsi);

    /**
     * @brief Find the velocity of the center of mass for a given field.
     * 
     * @param coordinate The coordinate index (0, 1, or 2).
     * @param whichPsi The field index.
     * @return The velocity of the center of mass.
     */
    double v_center_mass(int coordinate, int whichPsi);

    /**
     * @brief Calculate the kinetic energy at a grid point for a given field.
     * 
     * @param i The x-coordinate index.
     * @param j The y-coordinate index.
     * @param k The z-coordinate index.
     * @param whichPsi The field index.
     * @return The kinetic energy at the grid point.
     */
    double energy_kin(const int &i, const int &j, const int &k, int whichPsi);
    
    /**
     * @brief Compute the potential energy density at a grid point.
     * 
     * @param i The x-coordinate index.
     * @param j The y-coordinate index.
     * @param k The z-coordinate index.
     * @param whichPsi Indicates which field.
     * @return The potential energy density at the grid point.
     * 
     * This method is virtual for inheritance (external potential case).
     * Note that Psi fields have ghost points, Phi doesn't, k includes ghosts.
     */
    virtual double energy_pot(const int & i, const int & j, const int & k, int whichPsi);

    /**
     * @brief Calculate the full kinetic energy for a field.
     * 
     * @param whichPsi Indicates which field.
     * @return The full kinetic energy for the field.
     */
    double e_kin_full1(int whichPsi);

    /**
     * @brief Calculate the total kinetic energy for a field.
     * 
     * @param whichPsi Indicates which field.
     * @return The total kinetic energy for the field.
     */
    long double full_energy_kin(int whichPsi);

    /**
     * @brief Calculate the total potential energy for a field.
     * 
     * @param whichPsi Indicates which field.
     * @return The total potential energy for the field.
     */
    long double full_energy_pot(int whichPsi);

    /**
     * @brief Find the maximum density for a field.
     * 
     * @param whichPsi Indicates which field.
     * @return The maximum density for the field.
     */
    double find_maximum(int whichPsi);

    /**
     * @brief Update the ghost cells on the psi grids.
     * 
     * This method needs to be run before the time shots.
     */
    void sortGhosts();

    /**
     * @brief Perform a step forward in time for psi.
     * 
     * @param tstep The time step.
     * @param da The d_alpha coefficients.
     * @param whichPsi Indicates which field.
     * 
     * This method performs a step forward.
     */
    virtual void expiPhi(double tstep, double da, int whichPsi);

    /**
     * @brief Set the grid output mode.
     * 
     * @param grid_bool If false, domain3 outputs only the 2D sliced density profile.
     */
    void set_grid(bool grid_bool);

    /**
     * @brief Set the phase slice output mode.
     * 
     * @param bool_phase If false, domain3 does not output the phase slice.
     */
    void set_grid_phase(bool bool_phase);

    /**
     * @brief Set the backup flag.
     * 
     * @param bool_backup If false, the run does not start from an already existing backup.
     */
    void set_backup_flag(bool bool_backup);

    /**
     * @brief Set the adaptive time step flag.
     * 
     * @param bool_dt_adaptive If true, uses adaptive time step. Default is true.
     */
    void set_adaptive_dt_flag(bool bool_dt_adaptive);

    /**
     * @brief Set the ratio between the masses of the ULDM with respect to field 0.
     * 
     * @param ratio_mass The ratio between the masses.
     */
    void set_ratio_masses(multi_array<double,1> ratio_mass);

    /**
     * @brief Set the grid reduction parameter, relevant for full snapshots outputs.
     * 
     * @param reduce_grid The grid reduction parameter.
     */
    void set_reduce_grid(int reduce_grid);

    /**
     * @brief Set the energy spectrum computation flag.
     * 
     * @param spect If true, it does the computation of the energy spectrum.
     */
    void set_spectrum_flag(bool spect);

    /**
     * @brief Set the output file name.
     * 
     * @param name The output file name.
     */
    void set_output_name(string name);

    /**
     * @brief Set the number of steps.
     * 
     * @param numsteps The number of steps.
     */
    void set_numsteps(int numsteps);

    /**
     * @brief Perform a step in the simulation.
     * 
     * @param stepCurrent The current step.
     * @param tstep The time step.
     */
    virtual void makestep(double stepCurrent, double tstep);

    /**
     * @brief Solve the convection-diffusion equation.
     */
    virtual void solveConvDif();

    /**
     * @brief Set the initial conditions from the backup files.
     * 
     * This method should be called if the backup flag is true.
     * Note that you should call the backup from a run which uses the SAME number of cores in MPI processes.
     */
    void initial_cond_from_backup();

    // Methods for outputting data
    /**
     * @brief Open the necessary output files for the simulation.
     */
    void openfiles();

    /**
     * @brief Open the necessary backup files for the simulation.
     */
    void openfiles_backup();

    /**
     * @brief Output the full density to a file.
     * 
     * @param fileout The output file stream.
     * @param whichPsi Indicates which field.
     */
    void outputfulldensity(ofstream& fileout, int whichPsi);

    /**
     * @brief Output the full 3D Phi field for backup purposes.
     * 
     * @param fileout The output file stream.
     */
    void outputfullPhi(ofstream& fileout);

    /**
     * @brief Output the full 4D Psi field for backup purposes.
     * 
     * @param fileout The output file stream.
     * @param backup If true, outputs for backup; otherwise, outputs for snapshots.
     * @param reduce_grid The grid reduction parameter.
     */
    void outputfullPsi(ofstream& fileout, bool backup, int reduce_grid);

    /**
     * @brief Compute the radially averaged profiles
     * 
     * @param whichPsi Indicates which field.
     * @return The radial density profile.
     * 
     * Computes radially averaged profile starting from the maximum density point.
     * output is a 2D array with: radius, density, Ekin, Epot, Phi.
     * This method is virtual because for the NFW case (for example) you want to compute radial functions starting from the center of the box and not the maximum.
     */
    virtual multi_array<double, 2> profile_density(int whichPsi);

    /**
     * @brief Get the maximum density coordinate.
     * 
     * @param whichPsi Indicates which field.
     * @param coord The coordinate index (0, 1, or 2).
     * @return The maximum density coordinate.
     */
    double get_maxx(int whichPsi, int coord);

    /**
     * @brief Take a 2D or #D (if grid3D =true) density snapshot of the current state of the simulation.
     * 
     * @param stepCurrent The current step.
     */
    void snapshot(double stepCurrent);

    /**
     * @brief Take a snapshot of the radially averaged profiles of the current state of the simulation.
     * 
     * @param stepCurrent The current step.
     */
    void snapshot_profile(double stepCurrent);

    /**
     * @brief Store quantities relevant for spectrum energy computation.
     * 
     * @param spectrum_vect The vector to store spectrum quantities.
     * @param stepCurrent The current step.
     * @param tin The initial time.
     * @param tcurr The current time.
     */
    void spectrum_output(vector<vector<double>> &spectrum_vect, double stepCurrent, double tin, double tcurr);

    /**
     * @brief Write the energy spectrum to a file.
     * 
     * @param spectrum_vect The vector containing spectrum quantities.
     */
    void spectrum_write(vector<vector<double>> &spectrum_vect);

    /**
     * @brief Output the projected 2D column density for each field.
     * 
     * @param fileout The output file stream.
     */
    void outputSlicedDensity(ofstream& fileout);

    /**
     * @brief Output a 2D slice of the phase.
     * 
     * @param fileout The output file stream.
     */
    void outputPhaseSlice(ofstream& fileout);

    /**
     * @brief Close the output files.
     * 
     * This function closes files WITHOUT putting a final `}`. If one wants to read these arrays with a program (Mathematica etc.),
     * one should load those arrays and put a final `}`.
     */
    void closefiles();

    /**
     * @brief Export the run output information.
     * 
     * This method stores run output info.
     */
    virtual void exportValues();
    
    // Initial conditions
    /**
     * @brief Set test initial conditions with an empty initial conditions file.
     */
    void setTest();

    /**
     * @brief Set a uniform density sphere as initial condition.
     * 
     * @param rho0 The density of the sphere.
     * @param rad The radius of the sphere.
     */
    void uniform_sphere(double rho0, double rad);

    /**
     * @brief Set initial conditions with waves for test purposes.
     * 
     * @param whichF Indicates which field.
     */
    void initial_waves(int whichF);

    /**
     * @brief Set one soliton in the center of the box.
     * 
     * @param r_c The core radius of the soliton.
     * @param whichPsi Indicates which field.
     */
    void setInitialSoliton_1(double r_c, int whichPsi);

    /**
     * @brief Set one perturbed soliton in the center of the box.
     * 
     * @param r_c The core radius of the soliton.
     * @param c_pert The perturbation constant.
     */
    void set1Sol_perturbed(double r_c, double c_pert);

    /**
     * @brief Set many solitons as initial condition with random core radius.
     * 
     * @param num_Sol The number of solitons.
     * @param min_radius The minimum core radius.
     * @param max_radius The maximum core radius.
     * @param length_lim The length limit for the soliton centers.
     * @param whichPsi Indicates which field.
     */
    void setManySolitons_random_radius(int num_Sol, double min_radius, double max_radius, double length_lim, int whichPsi);

    /**
     * @brief Set many solitons as initial condition with the same core radius.
     * 
     * @param num_Sol The number of solitons.
     * @param r_c The core radius.
     * @param length_lim The length limit for the soliton centers.
     * @param whichPsi Indicates which field.
     */
    void setManySolitons_same_radius(int num_Sol, double r_c, double length_lim, int whichPsi);

    /**
     * @brief Set deterministic initial conditions with many solitons.
     * 
     * @param r_c The core radius.
     * @param num_sol The number of solitons.
     * @param whichPsi Indicates which field.
     */
    void setManySolitons_deterministic(double r_c, int num_sol, int whichPsi);

    /**
     * @brief Set Levkov-like initial conditions with waves.
     * 
     * @param Npart The number of particles.
     * @param whichPsi Indicates which field.
     */
    void set_waves_Levkov(double Npart, int whichPsi);

    /**
     * @brief Set delta function initial conditions in Fourier space.
     * 
     * @param Npart The number of particles.
     * @param whichPsi Indicates which field.
     */
    void set_delta(double Npart, int whichPsi);

    /**
     * @brief Set Heaviside function initial conditions in Fourier space.
     * 
     * @param Npart The number of particles.
     * @param whichPsi Indicates which field.
     */
    void set_theta(double Npart, int whichPsi);

    /**
     * @brief Set elliptic collapse initial conditions.
     * 
     * @param norm The normalization factor.
     * @param a_e The semi-major axis in the x direction.
     * @param b_e The semi-major axis in the y direction.
     * @param c_e The semi-major axis in the z direction.
     * @param whichPsi Indicates which field.
     * @param rand_phases If true, adds random phases.
     * @param Acorr The amplitude of the correlation.
     * @param lcorr The correlation length.
     */
    void setEllitpicCollapse(double norm, double a_e, double b_e, double c_e, int whichPsi, bool rand_phases, double Acorr, double lcorr);

    /**
     * @brief Set initial conditions from a file with density and velocity at each grid point.
     * 
     * @param filename_in The input file name for density.
     * @param filename_vel The input file name for velocity.
     */
    void set_initial_from_file(string filename_in, string filename_vel);

    /**
     * @brief Set a static profile with psi = sqrt(rho) without phases.
     * 
     * @param profile The profile object.
     * @param whichF Indicates which field.
     */
    void set_static_profile(Profile *profile, int whichF);

    /**
     * @brief Set Eddington initial conditions.
     * 
     * @param eddington The Eddington object.
     * @param numpoints The number of points.
     * @param radmin The minimum radius.
     * @param radmax The maximum radius.
     * @param fieldid The field ID.
     * @param ratiomass The ratio of mass.
     * @param num_k The number of k values.
     * @param simplify_k If true, simplifies k values.
     * @param center_x The x-coordinate of the center.
     * @param center_y The y-coordinate of the center.
     * @param center_z The z-coordinate of the center.
     */
    void setEddington(Eddington *eddington, int numpoints, double radmin, double radmax, int fieldid, 
                      double ratiomass, int num_k, bool simplify_k, int center_x, int center_y, int center_z);

    /**
     * @brief Re-output the initial conditions in the initial conditions file.
     * 
     * @param eddington The Eddington object.
     * @param numpoints The number of points.
     * @param radmin The minimum radius.
     * @param radmax The maximum radius.
     * @param fieldid The field ID.
     * @param ratiomass The ratio of mass.
     * @param num_k The number of k values.
     * @param simplify_k If true, simplifies k values.
     * @param center_x The x-coordinate of the center.
     * @param center_y The y-coordinate of the center.
     * @param center_z The z-coordinate of the center.
     */
    void output_Eddington_initial_cond(Eddington *eddington, int numpoints, double radmin, 
                                       double radmax, int fieldid, double ratiomass, int num_k, 
                                       bool simplify_k, int center_x, int center_y, int center_z);
};

#endif


