# -*- coding: utf-8 -*-

import os
import sys
import pde
import numpy as np
from scipy.io import wavfile as siowav
for path in [
        'models/vocal_fold',
        'solvers/ode_solvers'
]:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))),
            path))
from vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from adjoint_model_displacement import adjoint_model
from ode_solver import ode_solver, dae_solver

# Data
wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Normal/001.8k.wav'

sample_rate, samples = siowav.read(wav_file)
assert sample_rate == 8000, "{}: incompatible sample rate"\
    " need 8000 but got {}".format(wav_file, sample_rate)

# Inverse filter
# temporarily use glottal flow computed by covarep toolbox in Matlab
glottal_flow = np.loadtxt('filename_to_glottal_flow')
u0_m = A0 / (rho * c) * glottal_flow

# Solve vocal fold displacement model
X, dX = ode_solver(vdp_coupled, vdp_jacobian, model_params, init_state, init_t,
                   solver='lsoda', dt=0.1, t_max=1000)

# Solve adjoint model
M, dM = dae_solver(adjoint_model, model_params, init_state, init_t)

# Update parameters


# Results
