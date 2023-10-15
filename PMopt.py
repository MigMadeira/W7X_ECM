#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving the 
permanent magnet optimization on the NCSX grid.

The script should be run as:
    mpirun -n 1 python3 QI_Magnets.py

"""

import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField, ToroidalField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import PermanentMagnetGrid, create_equally_spaced_curves, curves_to_vtk, Surface
from simsopt.field import Current, coils_via_symmetries, Coil
from simsopt.solve import relax_and_split, GPMO 
from simsopt._core import Optimizable
import pickle
import time
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.mhd.vmec import Vmec
from simsopt import load

t_start = time.time()

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
dr = 0.11
surface_flag = 'wout'
input_name = 'wout_W7-X_standard_configuration.nc'
coordinate_flag = 'toroidal'


# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# Make the output directory
OUT_DIR = 'W7X_PMopt_2_20/'
os.makedirs(OUT_DIR, exist_ok=True)

#Loading the coils
coilfile = str(TEST_DIR/"./coils/2_20/biot_savart_opt.json")
bs_wrong_currents = load(coilfile)
ncoils = len(bs_wrong_currents.coils)

if ncoils == 40:
    scaling = 17.80113401
elif ncoils == 30:
    scaling = 23.74225526
else:
    scaling = 35.69195113
    
#fix the current
coils = bs_wrong_currents.coils
base_curves = [coils[i].curve for i in range(ncoils)]
base_currents = [coils[i].current*scaling for i in range(ncoils)]
fixed_coils = []
for i in range(ncoils):
    fixed_coils.append(Coil(base_curves[i], base_currents[i]))
bs = BiotSavart(fixed_coils)

# Set up BiotSavart fields
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
bs.save(OUT_DIR + f"biot_savart_opt.json")

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

#create the inside boundary for the PMs
s_in = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_in.extend_via_projected_normal(s.quadpoints_phi, 0.1)
s_in.to_vtk(OUT_DIR + "surf_in")

if ncoils == 40:
    #create the outside boundary for the PMs
    s_out = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
    s_out.extend_via_projected_normal(s.quadpoints_phi, 0.40)
    s_out.set_rc( 0, 0, s.get_rc(0,0) + 0.09)
    s_out_filename = TEST_DIR / 's_out.vts'
    s_out.to_vtk(OUT_DIR + "surf_out")
elif ncoils == 30:
    s_out = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
    s_out.extend_via_projected_normal(s.quadpoints_phi, 0.6)
    s_out.set_rc( 0, 0, s.get_rc(0,0) + 0.2)
    s_out.set_zs( 1, 0,  s.get_zs(0,0) + 1.0)
    s_out.to_vtk(OUT_DIR + "surf_out")
else:
    #create the outside boundary for the PMs
    s_out = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
    s_out.extend_via_projected_normal(s.quadpoints_phi, 0.55)
    s_out.set_rc( 0, 0, s.get_rc(0,0) + 0.2)
    s_out.to_vtk(OUT_DIR + "surf_out")

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)


# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s, dr=dr, rz_inner_surface=s_in, rz_outer_surface=s_out,
    Bn=Bnormal,
    filename=surface_filename,
    coordinate_flag=coordinate_flag
)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
algorithm = 'baseline'
kwargs = initialize_default_kwargs('GPMO')
if ncoils == 40:
    kwargs['K'] = 7200
    kwargs['nhistory'] = 600
    dividor = 60
elif ncoils == 30:
    kwargs['K'] = 11600
    kwargs['nhistory'] = 580
    divisor = 58
else:
    kwargs['K'] = 12300
    kwargs['nhistory'] = 410
    divisor = 41

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history,Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print("t2=" ,t2)
print('GPMO took t = ', t2 - t1, ' s')

# optionally save the whole solution history
#np.savetxt(OUT_DIR + 'mhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1))
np.savetxt(OUT_DIR + 'R2history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', R2_history)

# Note backtracking uses num_nonzeros since many magnets get removed 
plt.figure()
plt.semilogy(R2_history)
#plt.semilogy(pm_opt.num_nonzeros, R2_history[1:])
plt.grid(True)
plt.xlabel('K')
plt.ylabel('$f_B$')
plt.savefig(OUT_DIR + 'GPMO_MSE_history.png')

mu0 = 4 * np.pi * 1e-7
Bmax = 1.465
vol_eff = np.sum(np.sqrt(np.sum(m_history ** 2, axis=1)), axis=0) * mu0 * 2 * s.nfp / Bmax
np.savetxt(OUT_DIR + 'eff_vol_history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', vol_eff)

# Plot the MSE history versus the effective magnet volume
plt.figure()
print(len(R2_history),len(vol_eff),len(m_history))
plt.semilogy(vol_eff, R2_history)
#plt.semilogy(vol_eff[:len(pm_opt.num_nonzeros) + 1], R2_history) #for backtracking
plt.grid(True)
plt.xlabel('$V_{eff}$')
plt.ylabel('$f_B$')
plt.savefig(OUT_DIR + 'GPMO_Volume_MSE_history.png')

# Solution is the m vector that minimized the fb
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])
print("best result = ", 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
np.savetxt(OUT_DIR + 'best_result_m=' + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind )) + '.txt', m_history[:, :, min_ind ].reshape(pm_opt.ndipoles * 3))
b_dipole = DipoleField(pm_opt)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)))
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)))
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)), extra_data=pointData)
    
# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

# Plot the SIMSOPT GPMO solution 
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

for k in range(0, kwargs["nhistory"] + 1, dividor):
    pm_opt.m = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
    np.savetxt(OUT_DIR + 'm' + str(int(kwargs['K'] / (kwargs['nhistory']) * k)) + '.txt', m_history[:, :, k].reshape(pm_opt.ndipoles * 3))
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    print("Total fB = ",
          0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))

    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles
    # For plotting Bn on the full torus surface at the end with just the dipole fields
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)), extra_data=pointData)

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

# write solution to FAMUS-type file
write_pm_optimizer_to_famus(OUT_DIR, pm_opt)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
    from simsopt.mhd.vmec import Vmec
    mpi = MpiPartition(ngroups=4)
    comm = MPI.COMM_WORLD

    # Make the QFM surfaces
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface
    t2 = time.time()
    print("Making the QFM surface took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

t_end = time.time()
print('Total time = ', t_end - t_start)
#plt.show()
#plt.savefig("teste")
