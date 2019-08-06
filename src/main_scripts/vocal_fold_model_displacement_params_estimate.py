# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import logging.config
import numpy as np
import scipy.io
import scipy.io.wavfile
from matplotlib import pyplot as plt
import pdb
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
from ode_solver import ode_solver
from dae_solver import dae_solver

# TODO: put constants and others into configure file
if len(sys.argv) < 2:
    print('python {} configure.json'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    args = json.load(f)

# Log
if not os.path.exists(args['log_dir']):
    os.makedirs(args['log_dir'])

log_file = os.path.join(args['log_dir'], args['log_file'])
if os.path.isfile(log_file):  # remove existing log
    os.remove(log_file)

logging.config.dictConfig(args['log'])  # setup logger
logger = logging.getLogger('main')

# Data
wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Normal/001.8k.wav'

fs, samples = scipy.io.wavfile.read(wav_file)
assert fs == 8000, "{}: incompatible sample rate"\
    "--need 8000 but got {}".format(wav_file, fs)

# Inverse filter
# temporarily use glottal flow computed by covarep toolbox in Matlab
glottal_flow = scipy.io.loadmat('./Normal_001_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
glottal_flow = glottal_flow / np.linalg.norm(glottal_flow)  # normalize

assert len(glottal_flow) == len(samples), "Inconsistent length: glottal flow ({:d}) / wav file ({:d})".format(len(glottal_flow), len(samples))

# Initial conditions
# alpha = 0.8  # if > 0.5 delta, stable-like oscillator
# beta = 0.32
# delta = 1.  # asymmetry parameter

alpha = 0.6679  # if > 0.5 delta, stable-like oscillator
beta = 0.32
delta = 1.3376  # asymmetry parameter

vdp_init_t = 0.
# vdp_init_state = [0.1, 0., 0.1, 0.]  # (xl, dxl, xr, dxr)
vdp_init_state = [0., 0.1, 0., 0.1]  # (xl, dxl, xr, dxr)

# Some constants
x0 = 0.1  # half glottal width at rest position, cm
tau = 1e-3  # time delay for surface wave to travel half glottal height T, 1 ms
# Ps = 8000  # subglottal pressure, dyne/cm^2, 800Pa
eta = 1.  # nonlinear factor for energy dissipation at large amplitude
c = 5000  # air particle velocity, cm/s
d = 1.75  # length of vocal folds, cm
M = 0.5  # mass, g/cm^2
B = 100  # damping, dyne s/cm^3
# B = (tau * Ps * beta) / (x0 * alpha)
# K = 200000  # stiffness, dyne/cm^3
# K = (tau * Ps / x0) ** 2 / (alpha ** 2 * M)

R = 1e16

logger.info('Initial parameters: alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}'.
            format(alpha, beta, delta))
logger.info('-' * 110)

# TODO: stop criterion tolerance, early stop
while np.linalg.norm(R) > 1:  # norm of the difference between predicted and real glottal flow

    logger.info('Solving vocal fold displacement model')

    vdp_params = [alpha, beta, delta]

    K = B ** 2 / (beta ** 2 * M)
    Ps = (alpha * x0 * np.sqrt(M * K)) / tau

    time_scaling = np.sqrt(K / float(M))  # t -> s
    x_scaling = np.sqrt(eta)

    logger.debug('stiffness K = {:.4f} dyne/cm^3    subglottal Ps = {:.4f} dyne/cm^2    '
                 'time_scaling = {:.4f}'.format(K, Ps, time_scaling))

    # Solve vocal fold displacement model
    sol = ode_solver(vdp_coupled, vdp_jacobian, vdp_params,
                     vdp_init_state, vdp_init_t,
                     solver='lsoda', ixpr=0,
                     dt=(time_scaling / float(fs)),  # dt -> ds
                     tmax=(time_scaling * len(samples) / float(fs))
                     )

    # Plot xl, xr for 0.1s
    # plt.figure()
    # plt.plot(sol[:int(0.1 * fs), 0], sol[:int(0.1 * fs), 1], 'b.-')
    # plt.plot(sol[:int(0.1 * fs), 0], sol[:int(0.1 * fs), 3], 'r.-')
    # plt.legend(['right', 'left'])
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.show()

    # Plot states
    # plt.figure()
    # plt.subplot(121)
    # plt.plot(sol[:, 1], sol[:, 3], 'b.-')
    # plt.xlabel('u')
    # plt.ylabel('v')
    # plt.subplot(122)
    # plt.plot(sol[:, 2], sol[:, 4], 'b.-')
    # plt.xlabel('du')
    # plt.ylabel('dv')
    # plt.show()

    logger.info('Solving adjoint model')

    # Calculate some terms
    # BUG: len > 1
    if len(sol) > len(samples):
        sol = sol[:-1]
    assert len(sol) == len(samples), "Inconsistent length: ODE sol ({:d}) / wav file ({:d})".format(len(sol), len(samples))

    X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
    dX = sol[:, [2, 4]]  # cm/s
    u0 = c * d * np.sum(X, axis=1)  # volume velocity flow, cm^3/s
    u0 = u0 / np.linalg.norm(u0)  # normalize
    R = u0 - glottal_flow

    # Plot glottal flow
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(sol[:int(0.1 * fs), 0], glottal_flow[:int(0.1 * fs)], 'b.-')
    # plt.ylabel('glottal flow')
    # plt.subplot(312)
    # plt.plot(sol[:int(0.1 * fs), 0], u0[:int(0.1 * fs)], 'b.-')
    # plt.ylabel('u0')
    # plt.subplot(313)
    # plt.plot(sol[:int(0.1 * fs), 0], R[:int(0.1 * fs)], 'b.-')
    # plt.ylabel('R')
    # plt.xlabel('t')
    # plt.show()

    # plt.figure()
    # plt.plot(sol[:int(0.1 * fs), 0], glottal_flow[:int(0.1 * fs)], 'k.-')
    # plt.plot(sol[:int(0.1 * fs), 0], u0[:int(0.1 * fs)], 'b.-')
    # plt.plot(sol[:int(0.1 * fs), 0], R[:int(0.1 * fs)], 'r.-')
    # plt.xlabel('t')
    # plt.legend(['glottal flow', 'u0', 'R'])
    # plt.show()
    # plt.figure()
    # plt.plot(sol[:, 0], glottal_flow[:], 'k.-')
    # plt.plot(sol[:, 0], u0[:], 'b.-')
    # plt.plot(sol[:, 0], R[:], 'r.-')
    # plt.xlabel('t')
    # plt.legend(['glottal flow', 'u0', 'R'])
    # plt.show()

    # Solve adjoint model
    residual, jac = adjoint_model(alpha, beta, delta, X, dX, R, fs)
    M_T = [0., 0., 0., 0.]  # initial states of the adjoint model at T
    dM_T = [0., -R[-1], 0., -R[-1]]  # initial ddL = ddE = -R(T)
    # dM_T = [0., 0.01, 0., 0.01]  # BUG: inconsistent initial conditions?
    T = len(samples) / float(fs)
    adjoint_sol = dae_solver(residual, M_T, dM_T, T,
                             solver='IDA', algvar=[0, 1, 0, 1], suppress_alg=True, atol=1e-6, rtol=1e-6,
                             usejac=True, jac=jac, usesens=False,
                             backward=True, tfinal=0., ncp=len(samples),  # simulate (T --> 0)s backwards
                             display_progress=True, report_continuously=False, verbosity=50)  # NOTE: report_continuously should be False

    # Update parameters
    L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
    E = adjoint_sol[1][:, 2][::-1]

    # BUG: length +1
    L = L[:-1]
    E = E[:-1]

    d_alpha = -np.dot((dX[:, 0] + dX[:, 1]), (L + E))
    d_beta = np.sum(L * (1 + np.square(X[:, 0])) * dX[:, 0] + E * (1 + np.square(X[:, 1])) * dX[:, 1])
    d_delta = np.sum(0.5 * (X[:, 1] * E - X[:, 0] * L))

    # TODO: stepsize, adaptive adjust stepsize
    alpha = alpha - 0.1 * d_alpha
    # beta = beta - 0.1 * d_beta
    delta = delta - 0.1 * d_delta

    logger.info('norm(R) = {:.4f} | alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}'.
                format(np.linalg.norm(R), alpha, beta, delta))
    logger.info('-' * 110)

# Results
logger.info('*' * 110)

logger.info('alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}  norm(R) = {:.4f}'.
            format(alpha, beta, delta, np.linalg.norm(R)))
