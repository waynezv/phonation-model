# -*- coding: utf-8 -*-

import os
import sys
import json
import wave
import struct
import logging
import logging.config
import numpy as np
import scipy.io
import scipy.io.wavfile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pdb
for path in [
        'models/vocal_fold',
        'solvers/ode_solvers',
        'solvers/pde_solvers'
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
from fem_solver import vocal_tract_solver, vocal_tract_solver_backward


def pcm16_to_float(wav_file):
    '''
    Convert 16-bit signed integer PCM wav samples to
    float array between [-1, 1].
    '''
    w = wave.open(wav_file)

    num_frames = w.getnframes()
    num_channels = w.getnchannels()
    num_samples = num_frames * num_channels
    fs = w.getframerate()  # sample rate

    # Convert binary chunks to short ints
    fmt = "%ih" % num_samples
    raw_data = w.readframes(num_frames)
    a = struct.unpack(fmt, raw_data)

    # Convert short ints to floats
    a = [float(val) / pow(2, 15) for val in a]

    return np.array(a, dtype=float), fs


# TODO: put constants into configure file
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
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Normal/001.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Neoplasm/001.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Phonotrauma/001.8k.wav'
wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Vocal_palsy/001.8k.wav'
samples, fs = pcm16_to_float(wav_file)
assert fs == 8000, "{}: incompatible sample rate"\
    "--need 8000 but got {}".format(wav_file, fs)

# fig = plt.figure()
# plt.plot(np.linspace(0, len(samples) / fs, len(samples)), samples)
# plt.show()

# TODO: trim
# samples = samples[int(4 * fs): int(6 * fs)]  # for normal, neoplasm, phonotrauma
samples = samples[int(1 * fs): int(3 * fs)]  # for vocal_palsy

# samples = (samples - samples.min()) / (samples.max() - samples.min())
samples = samples / np.linalg.norm(samples)  # normalize

# Initial conditions
alpha = 0.8  # if > 0.5 delta, stable-like oscillator
beta = 0.32
delta = 1.  # asymmetry parameter

# alpha = 0.85
# beta = 0.65
# delta = 1.

vdp_init_t = 0.
vdp_init_state = [0., 0.1, 0., 0.1]  # (xl, dxl, xr, dxr), xl=xr=0

# Some constants
x0 = 0.1  # half glottal width at rest position, cm
d = 1.75  # length of vocal folds, cm
tau = 1e-3  # time delay for surface wave to travel half glottal height T, 1 ms
eta = 1.  # nonlinear factor for energy dissipation at large amplitude
c = 5000  # air particle velocity, cm/s
M = 0.5  # mass, g/cm^2
B = 100  # damping, dyne s/cm^3

c_sound = 34000.  # speed of sound, cm/s
tau_f = 1.  # parameter for Updating f
gamma_f = 1.  # parameter for updating f


R = 1e16  # prediction residual w.r.t L2 norm
logger.info('Initial parameters: alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}'.
            format(alpha, beta, delta))
logger.info('-' * 110)

# Define some constants
np.random.seed(1749)

Nx = 64  # number of uniformly spaced cells in mesh
BASIS_DEGREE = 2  # degree of the basis functional space
length = 17.5  # spatial dimension, length of vocal tract, cm
divisions = (Nx,)  # mesh size
num_dof = Nx * BASIS_DEGREE + 1  # degree of freedom

num_tsteps = len(samples)  # number of time steps
T = len(samples) / float(fs)  # total time, s
dt = T / num_tsteps  # time step size
print('Total time: {:.4f}s  Stepsize: {:.4g}s'.format(T, dt))
f_data = np.zeros((num_tsteps, num_dof))  # init f

samples = samples[:num_tsteps]
iteration = 0
while R > 0.1:

    # Step 1: solve vocal fold displacement model
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

    # Step 2: solve forward vocal tract model
    logger.info('Solving forward vocal tract model')

    # Calculate some terms
    # BUG: len > 1
    if len(sol) > len(samples):
        sol = sol[:-1]
    assert len(sol) == len(samples),\
        "Inconsistent length: ODE sol ({:d}) / wav file ({:d})".\
        format(len(sol), len(samples))

    X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
    dX = sol[:, [2, 4]]  # cm/s
    u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
    u0 = u0 / np.linalg.norm(u0)  # normalize
    u0 = u0[:num_tsteps]

    # plt.figure()
    # plt.plot(np.linspace(0, T, len(u0)), u0)
    # plt.show()

    uL_k, U_k = vocal_tract_solver(f_data, u0, samples, c_sound,
                                   length, Nx, BASIS_DEGREE,
                                   T, num_tsteps, iteration)

    # Step 3: calculate difference signal  # BUG: should be ~nil by boundary condition, remove right bc?
    logger.info('Calculating difference signal')
    uL_k = uL_k / np.linalg.norm(uL_k)  # normalize
    r_k = samples - uL_k

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, T, len(samples)), samples, 'b')
    ax.plot(np.linspace(0, T, len(samples)), uL_k, 'r')
    ax.set_xlabel('t')
    plt.legend(['samples', 'uL_k'])
    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/wzhao/ProJEX/phonation-model/src/main_scripts/outputs/vocal_tract_estimate/plots_run_0920_4/vocal_tract_estimate_uL_iter{}.png'.format(iteration))

    # Step 4: solve backward vocal tract model
    logger.info('Solving backward vocal tract model')
    Z_k = vocal_tract_solver_backward(f_data, r_k, c_sound,
                                      length, Nx, BASIS_DEGREE,
                                      T, num_tsteps, iteration)

    # Step 5: update f
    logger.info('Updating f^k')
    f_data = f_data + (tau_f / gamma_f) *\
        (Z_k[::-1, ...] / (c_sound ** 2) + U_k)

    iteration = iteration + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(np.linspace(0, T, f_data.shape[0]), np.linspace(0, length, f_data.shape[1]))
    ax.plot_surface(XX, YY, f_data.T, cmap='coolwarm')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f')

    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/wzhao/ProJEX/phonation-model/src/main_scripts/outputs/vocal_tract_estimate/plots_run_0920_4/vocal_tract_estimate_f_iter{}.png'.format(iteration))

    # Step 6: solve adjoint model
    logger.info('Solving adjoint model')

    dHf_u0 = uL_k / (u0 + 1E-14)  # derivative of operator Hf w.r.t. u0
    R_diff = 2 * c * d * (-r_k) * dHf_u0  # term c.r.t. difference signal

    residual, jac = adjoint_model(alpha, beta, delta, X, dX, R_diff,
                                  num_tsteps/T)

    M_T = [0., 0., 0., 0.]  # initial states of the adjoint model at T
    dM_T = [0., -R_diff[-1], 0., -R_diff[-1]]  # initial ddL = ddE = -R_diff(T)

    adjoint_sol = dae_solver(residual, M_T, dM_T, T,
                             solver='IDA', algvar=[0, 1, 0, 1], suppress_alg=True, atol=1e-6, rtol=1e-6,
                             usejac=True, jac=jac, usesens=False,
                             backward=True, tfinal=0., ncp=(len(samples)-1),  # simulate (T --> 0)s backwards
                             display_progress=True, report_continuously=False, verbosity=50)  # NOTE: report_continuously should be False

    # Step 7: update vocal fold model parameters
    logger.info('Updating parameters')

    L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
    E = adjoint_sol[1][:, 2][::-1]
    assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"
    L = L / np.linalg.norm(L)
    E = E / np.linalg.norm(E)

    d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
    d_beta = np.sum(L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0] + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1])
    d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))

    # TODO: stepsize, adaptive adjust stepsize
    # stepsize = 0.1
    stepsize = 0.01 / np.max([d_alpha, d_beta, d_delta])
    alpha = alpha - stepsize * d_alpha
    beta = beta - stepsize * d_beta
    delta = delta - stepsize * d_delta

    R = np.sqrt(np.sum(r_k ** 2))

    logger.info('L2 Residual = {:.4f} | alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}'.
                format(R, alpha, beta, delta))
    logger.info('-' * 110)

# Results
logger.info('*' * 110)

logger.info('alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}  norm(R) = {:.4f}'.
            format(alpha, beta, delta, np.linalg.norm(R)))
