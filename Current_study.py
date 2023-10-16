#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving the 
permanent magnet optimization on the NCSX grid.

The script should be run as:
    mpirun -n 1 python3 QI_Magnets.py

"""
import sys
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

def write_float_line(file, float1, float2, float3):
    line = "{:.{}f} {:.{}f} {:.{}f}\n".format(float1, sys.float_info.dig, float2, sys.float_info.dig, float3, sys.float_info.dig)
    file.write(line)

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
dr = 0.055
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
OUT_DIR = './W7X_PMopt_2_20_dr=0.055/'
os.makedirs(OUT_DIR, exist_ok=True)

#Loading the coils
coilfile = str(TEST_DIR/"./coils/2_20/biot_savart_opt.json")
bs_wrong_currents = load(coilfile)
ncoils = len(bs_wrong_currents.coils)

if ncoils == 40:
    scaling = np.linspace(1, 17.80113401, 10)
elif ncoils == 30:
    scaling = np.linspace(1, 23.74225526, 10)
else:
    scaling = np.linspace(1, 35.69195113, 10)

for x in scaling: 
    OUT_DIR2 = OUT_DIR + f'scaling={x}/'
    os.makedirs(OUT_DIR2, exist_ok=True)
       
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
    bs.save(OUT_DIR2 + f"biot_savart_opt.json")

    # check after-optimization average on-axis magnetic field strength
    B0avg = calculate_on_axis_B(bs, s)

    # Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
    make_Bnormal_plots(bs, s_plot, OUT_DIR2, "biot_savart_initial")

    #create the inside boundary for the PMs
    s_in = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
    s_in.extend_via_projected_normal(s.quadpoints_phi, 0.1)
    s_in.to_vtk(OUT_DIR2 + "surf_in")

    if ncoils == 40:
        #create the outside boundary for the PMs
        s_out = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_out.extend_via_projected_normal(s.quadpoints_phi, 0.40)
        s_out.set_rc( 0, 0, s.get_rc(0,0) + 0.09)
        s_out_filename = TEST_DIR / 's_out.vts'
        s_out.to_vtk(OUT_DIR2 + "surf_out")
    elif ncoils == 30:
        s_out = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_out.extend_via_projected_normal(s.quadpoints_phi, 0.6)
        s_out.set_rc( 0, 0, s.get_rc(0,0) + 0.2)
        s_out.set_zs( 1, 0,  s.get_zs(0,0) + 1.0)
        s_out.to_vtk(OUT_DIR2 + "surf_out")
    else:
        #create the outside boundary for the PMs
        s_out = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_out.extend_via_projected_normal(s.quadpoints_phi, 0.55)
        s_out.set_rc( 0, 0, s.get_rc(0,0) + 0.2)
        s_out.to_vtk(OUT_DIR2 + "surf_out")

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
        kwargs['K'] = 32100 
        kwargs['nhistory'] = 321 
    elif ncoils == 30:
        kwargs['K'] = 49700 
        kwargs['nhistory'] = 497
    else:
        kwargs['K'] = 52800 
        kwargs['nhistory'] = 528

    # Optimize the permanent magnets greedily
    t1 = time.time()
    R2_history,Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
    t2 = time.time()
    print("t2=" ,t2)
    print('GPMO took t = ', t2 - t1, ' s')

    # optionally save the whole solution history
    #np.savetxt(OUT_DIR + 'mhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1))
    np.savetxt(OUT_DIR2 + 'R2history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', R2_history)
    np.savetxt(OUT_DIR2 + 'Bnhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', Bn_history)

    # Note backtracking uses num_nonzeros since many magnets get removed 
    plt.figure()
    plt.semilogy(R2_history)
    #plt.semilogy(pm_opt.num_nonzeros, R2_history[1:])
    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('$f_B$')
    plt.savefig(OUT_DIR2 + 'GPMO_MSE_history.png')

    mu0 = 4 * np.pi * 1e-7
    Bmax = 1.465
    vol_eff = np.sum(np.sqrt(np.sum(m_history ** 2, axis=1)), axis=0) * mu0 * 2 * s.nfp / Bmax
    np.savetxt(OUT_DIR2 + 'eff_vol_history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', vol_eff)

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
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    Bnormal_total = Bnormal + Bnormal_dipoles
    # For plotting Bn on the full torus surface at the end with just the dipole fields
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR2, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)))
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(OUT_DIR2 + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)), extra_data=pointData)
        
    # Plot the SIMSOPT GPMO solution 
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

    k=10
    pm_opt.m = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
    np.savetxt(OUT_DIR2 + 'm' + str(int(kwargs['K'] / (kwargs['nhistory']) * k)) + '.txt', m_history[:, :, k].reshape(pm_opt.ndipoles * 3))
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR2 + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))

    Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles
    # For plotting Bn on the full torus surface at the end with just the dipole fields
    make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)))
    pointData = {"B_N": Bnormal_total[:, :, None]}
    s_plot.to_vtk(OUT_DIR2 + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * k)), extra_data=pointData)

    # write solution to FAMUS-type file
    write_pm_optimizer_to_famus(OUT_DIR2, pm_opt)
    
    with open('current_study.txt', 'a') as f:
        write_float_line(f, scaling, B0avg, R2_history[min_ind])
