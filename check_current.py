import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.geo import SurfaceRZFourier, curves_to_vtk, PermanentMagnetGrid
from simsopt.field import Current, Coil, BiotSavart, DipoleField
import simsoptpp as sopp
from simsopt.mhd.vmec import Vmec
from simsopt import load
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.solve import GPMO
from simsopt.objectives import SquaredFlux

# Set some parameters
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above

input_name = './wout_W7-X_standard_configuration.nc'
algorithm = "baseline"

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR/input_name)

s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

#Loading the coils
coilfile = str(TEST_DIR/"./coils/3_13/biot_savart_opt.json")
bs = load(coilfile)
ncoils = len(bs.coils)

#fix the current
coils = bs.coils
base_curves = [coils[i].curve for i in range(ncoils)]
base_currents = [coils[i].current*23.74225526 for i in range(ncoils)]
fixed_coils = []
for i in range(ncoils):
    fixed_coils.append(Coil(base_curves[i], base_currents[i]))
bs_fixed = BiotSavart(fixed_coils)

# Set up BiotSavart fields
bs_fixed.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs_fixed.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

calculate_on_axis_B(bs_fixed, s)