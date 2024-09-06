Initial Conditions explanation
-----------

Here are listed all the possible initial conditions for main_sim_mpi_2fields.cpp

### levkov
- **Description:** Sets initial conditions based on Levkov's waves.
- **Parameters:**
  - `Nparts[i]`: Number of particles for each field.
  - `ratio_mass[i]`: Ratio of masses for each field.

### delta
- **Description:** Sets initial conditions using Dirac delta on Fourier space.
- **Parameters:**
  - `Nparts[i]`: Number of particles for each field.
  - `ratio_mass[i]`: Ratio of masses for each field (for `num_fields > 1`).

### theta
- **Description:** Sets initial conditions using Heaviside function on Fourier space.
- **Parameters:**
  - `Nparts[i]`: Number of particles for each field.
  - `ratio_mass[i]`: Ratio of masses for each field (for `num_fields > 1`).

### 1Sol
- **Description:** Sets initial conditions for a single soliton.
- **Parameters:**
  - `rc`: Radius of the soliton.
  - `whichpsi`: Index of the field where the soliton is placed.
  - `ratio_mass[whichpsi]`: Ratio of mass for the specified field.

### eddington_nfw_soliton
- **Description:** Sets initial conditions using NFW Eddington profile plus a soliton.
- **Parameters:**
  - `ratio_mass[1]`: Ratio of mass for the second field.
  - `rs`: NFW scale radius.
  - `rhos`: NFW normalization.
  - `rc`: Core radius.
  - `num_k`: Number of k values for the Eddington function.

### eddington_nfw
- **Description:** Sets initial conditions using NFW Eddington profile.
- **Parameters:**
  - `ratio_mass[i]`: Ratio of masses for each field.
  - `nprofile_per_mass[i]`: Number of profiles per mass for each field.
  - `rs[index_field]`: NFW scale radius.
  - `rhos[index_field]`: NFW normalization.
  - `center_x[index_field]`: X-coordinate of the profile center.
  - `center_y[index_field]`: Y-coordinate of the profile center.
  - `center_z[index_field]`: Z-coordinate of the profile center.
  - [`num_k`]: Number of k values for the Eddington function.

### eddington_nfw_halos
- **Description:** Sets initial conditions using NFW Eddington profile for halos.
- **Parameters:**
  - `ratio_mass[i]`: Ratio of masses for each field.
  - `density_percentage[i]`: Density percentage for each field.
  - `num_halos`: Number of halos.
  - `rs[i]`: NFW scale radius.
  - `rhos[i]`: NFW normalization.
  - `center_x[i]`: X-coordinate of the halo center.
  - `center_y[i]`: Y-coordinate of the halo center.
  - `center_z[i]`: Z-coordinate of the halo center.
  - `num_k`: Number of k values for the Eddington function.

### eddington_plummer
- **Description:** Sets initial conditions using Plummer Eddington profile.
- **Parameters:**
  - `field_id`: The field where to put the Eddington generated NFW profile.
  - `ratio_mass[field_id]`: Ratio of mass for the specified field.
  - `rs`: Plummer scale radius.
  - `m0`: Plummer mass normalization.
  - `num_k`: Number of k values for the Eddington function.

### eddington_nfw_levkov
- **Description:** Sets initial conditions using NFW Eddington profile for field 1 plus Levkov for field 0.
- **Parameters:**
  - `Nparts`: Levkov initial condition parameter.
  - `ratio_mass[1]`: Ratio of mass for the second field.
  - `rs`: NFW scale radius.
  - `rhos`: NFW normalization.
  - `num_k`: Number of k values for the Eddington function.

### schive
- **Description:** Schive initial conditions.
- **Parameters:**
  - `rc`: Radius of soliton.
  - `Nsol`: Number of solitons.
  - `length_lim`: Length limit of span of solitons.
  - [`ratio_mass[0]`]: Ratio of mass for the first field.

### mocz
- **Description:** Mocz initial conditions.
- **Parameters:**
  - `min_radius`: Minimum radius of soliton.
  - `max_radius`: Maximum radius of soliton.
  - `Nsol`: Number of solitons.
  - `length_lim`: Length limit of span of solitons.

### deterministic
- **Description:** Deterministic initial conditions for tests.
- **Parameters:**
  - `rc`: Radius of soliton.
  - `Nsol`: Number of solitons (should not surpass 30).
  - `ratio_mass[0]`: Ratio of mass for the first field.

### elliptCollapse
- **Description:** Elliptical collapse initial conditions.
- **Parameters:**
  - `Norm`: Normalization of profile.
  - `a_e`: Elliptical parameter a.
  - `b_e`: Elliptical parameter b.
  - `c_e`: Elliptical parameter c.
  - `ratio_mass[0]`: Ratio of mass for the first field.
  - `rand_phases`: If true, use random phases; otherwise, use random field.
  - `A_corr`: Correlation amplitude; if 0, avoid the random procedure completely.
  - `lcorr`: Correlation length, effective if [`rand_phases`] is true.

### staticProfile_NFW
- **Description:** Static NFW initial condition, psi = sqrt(rho).
- **Parameters:**
  - `rs`: NFW scale radius.
  - `rhos`: NFW normalization.