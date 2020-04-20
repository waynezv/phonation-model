# -*- coding: utf-8 -*-

import os
import sys
import json
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
        'utils'
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
from utils import voice_activity_detection,\
    optim_grad_step, optim_adapt_step, optim_adam
from sigproc import framesig


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
data_root = '../../data/COVID-19'
wav_dir = 'data_for_wayne'
flw_dir = 'glottal_flow'
sub_dirs = ['covid_negative', 'covid_positive']

# wav_file = os.path.join(data_root, wav_dir, sub_dirs[0], 'audio_aaa_0ujacne.16k.wav')
# flw_file = os.path.join(data_root, flw_dir, sub_dirs[0], 'audio_aaa_0ujacne.16k.wav.mat')
wav_file = os.path.join(data_root, wav_dir, sub_dirs[1], 'audio_aaa_0SyzEZp.16k.wav')
flw_file = os.path.join(data_root, flw_dir, sub_dirs[1], 'audio_aaa_0SyzEZp.16k.wav.mat')
# wav_file = os.path.join(data_root, wav_dir, sub_dirs[1], 'audio_aaa_0TxJdJL.16k.wav')
# flw_file = os.path.join(data_root, flw_dir, sub_dirs[1], 'audio_aaa_0TxJdJL.16k.wav.mat')
# wav_file = os.path.join(data_root, wav_dir, sub_dirs[1], 'audio_ooo_ZqLYCVI.16k.wav')
# flw_file = os.path.join(data_root, flw_dir, sub_dirs[1], 'audio_ooo_ZqLYCVI.16k.wav.mat')

fs, raw_samples = scipy.io.wavfile.read(wav_file)
raw_glottal_flow = scipy.io.loadmat(flw_file)['gf_iaif_ola_sync'].reshape((-1,))

# VAD
voiced_segments = voice_activity_detection(raw_samples, fs,
                                           window_duration=30,
                                           bytes_per_sample=2,
                                           vad_mode=3)

wav_samples = np.concatenate([raw_samples[segment['start']:segment['stop']]
                              for segment in voiced_segments if segment['is_speech']])
glottal_flow = np.concatenate([raw_glottal_flow[segment['start']:segment['stop']]
                               for segment in voiced_segments if segment['is_speech']])

wav_samples = wav_samples / float(pow(2, 15))  # to float

assert len(glottal_flow) == len(wav_samples),\
    f"Inconsistent length: glottal flow ({len(glottal_flow):d}) / wav samples ({len(wav_samples):d})"

# Normalize
wav_samples = wav_samples / np.linalg.norm(wav_samples)
glottal_flow = glottal_flow / np.linalg.norm(glottal_flow)

# Frame
# sample_frames = framesig(wav_samples, 0.025 * fs, 0.01 * fs, winfunc=lambda x: np.ones((x,)), stride_trick=True)
# flow_frames = framesig(glottal_flow, 0.025 * fs, 0.01 * fs, winfunc=lambda x: np.ones((x,)), stride_trick=True)
sample_frames = framesig(wav_samples, 0.5 * fs, 0.5 * fs, winfunc=lambda x: np.ones((x,)), stride_trick=True)
flow_frames = framesig(glottal_flow, 0.5 * fs, 0.5 * fs, winfunc=lambda x: np.ones((x,)), stride_trick=True)

# Some constants
x0 = 0.1  # half glottal width at rest position, cm
tau = 1e-3  # time delay for surface wave to travel half glottal height T, 1 ms
eta = 1.  # nonlinear factor for energy dissipation at large amplitude
c = 5000  # air particle velocity, cm/s
d = 1.75  # length of vocal folds, cm
M = 0.5  # mass, g/cm^2
B = 100  # damping, dyne s/cm^3

# Initial conditions
alpha = 0.8  # if > 0.5 delta, stable-like oscillator
beta = 0.32
delta = 1.  # asymmetry parameter
vdp_init_t = 0.
vdp_init_state = [0., 0.1, 0., 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
logger.info(f'Initial parameters: alpha = {alpha:.4f}   beta = {beta:.4f}   '
            f'delta = {delta:.4f}')
logger.info('-' * 110)

# Loop over frames
best_results = {
    'iteration': [],
    'Rk': [],
    'alpha': [],
    'beta': [],
    'delta': [],
    'sol': [],
    'u0': [],
    'R': []
}
for Tk, (samples, flow) in enumerate(zip(sample_frames, flow_frames)):

    # fig = plt.figure()
    # plt.plot(np.linspace(0, len(samples) / fs, len(samples)), samples)
    # plt.plot(np.linspace(0, len(flow) / fs, len(flow)), flow)
    # plt.legend(['speech sample', 'glottal flow'])
    # plt.show()

    num_tsteps = len(samples)  # number of time steps
    T = len(samples) / float(fs)  # total time, s

    iteration = 0  # optimize iter
    Rk = 1e16  # prediction residual w.r.t L2 norm @ k
    Rk_best = 1e16
    patience = 0  # number of patient iterations of no improvement before stopping optim
    while patience < 100:

        # Solve vocal fold displacement model
        logger.info('Solving vocal fold displacement model')

        vdp_params = [alpha, beta, delta]

        K = B ** 2 / (beta ** 2 * M)
        Ps = (alpha * x0 * np.sqrt(M * K)) / tau
        time_scaling = np.sqrt(K / float(M))  # t -> s
        x_scaling = np.sqrt(eta)
        logger.debug(f'stiffness K = {K:.4f} dyne/cm^3    subglottal Ps = {Ps:.4f} '
                     f'dyne/cm^2    time_scaling = {time_scaling:.4f}')

        sol = ode_solver(vdp_coupled, vdp_jacobian, vdp_params,
                        vdp_init_state, (time_scaling * (vdp_init_t + Tk * T)),
                        solver='lsoda', ixpr=0,
                        dt=(time_scaling / float(fs)),  # dt -> ds
                        tmax=(time_scaling * (Tk + 1) * T)
                        )
        if len(sol) > len(samples):
            sol = sol[:-1]
        assert len(sol) == len(samples), f"Inconsistent length: ODE sol ({len(sol):d}) / wav file ({len(samples):d})"

        # Calculate some terms
        X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
        dX = sol[:, [2, 4]]  # cm/s
        u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
        u0 = u0 / np.linalg.norm(u0) * np.linalg.norm(flow)  # normalize
        u0 = u0[:num_tsteps]

        # Prediction residual
        R = u0 - flow

        # Plot glottal flow
        # plt.figure()
        # plt.plot(sol[:, 0], flow, 'k.-')
        # plt.plot(sol[:, 0], u0, 'b.-')
        # plt.plot(sol[:, 0], R, 'r.-')
        # plt.xlabel('t')
        # plt.legend(['glottal flow', 'u0', 'R'])
        # plt.show()

        # Solve adjoint model
        logger.info('Solving adjoint model')

        residual, jac = adjoint_model(alpha, beta, delta, X, dX, R, num_tsteps/T, (Tk * T), (Tk + 1) * T)

        M_T = [0., 0., 0., 0.]  # initial states of the adjoint model at T
        dM_T = [0., -R[-1], 0., -R[-1]]  # initial ddL = ddE = -R(T)

        adjoint_sol = dae_solver(residual, M_T, dM_T, (Tk + 1) * T,
                                backward=True, tfinal=(Tk * T), ncp=len(samples),  # simulate (tfinal-->t0)s backward
                                solver='IDA', algvar=[0, 1, 0, 1], suppress_alg=True, atol=1e-6, rtol=1e-6,
                                usejac=True, jac=jac, usesens=False,
                                display_progress=True, report_continuously=False, verbosity=50)  # NOTE: report_continuously should be False

        # Update parameters
        logger.info('Updating parameters')
        L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
        E = adjoint_sol[1][:, 2][::-1]
        assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"
        L = L / np.linalg.norm(L)  # normalize
        E = E / np.linalg.norm(E)

        alpha_k = alpha  # record parameters@current step
        beta_k = beta
        delta_k = delta
        Rk = np.sqrt(np.sum(R ** 2))
        logger.info(f'[{patience:d}:{iteration:d}][{Tk:d}/{len(sample_frames):d}] L2 Residual = {Rk:.4f} | alpha = {alpha_k:.4f}   beta = {beta_k:.4f}   '
                    f'delta = {delta_k:.4f}')

        if Rk < Rk_best:  # has improvement
            # Record best
            iteration_best = iteration
            Rk_best = Rk
            alpha_best = alpha_k
            beta_best = beta_k
            delta_best = delta_k
            pv_best = np.array([alpha_best, beta_best, delta_best])  # param vec
            sol_best = sol
            u0_best = u0
            R_best = R

            # Compute gradients
            d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
            d_beta = np.sum(L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0] + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1])
            d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vec
            dpv = dpv / np.linalg.norm(dpv)
            d_alpha, d_beta, d_delta = dpv

            # Update
            iteration += 1
            # alpha, beta, delta = optim_grad_step(alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.1)
            alpha, beta, delta = optim_adapt_step(alpha, beta, delta, d_alpha, d_beta, d_delta, default_step=0.1)

        else:  # no improvement
            patience = patience + 1

            # Compute conjugate gradients
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vec
            dpv = dpv / np.linalg.norm(dpv)
            ov = np.random.randn(len(dpv))  # ortho vec
            ov = ov - (np.dot(ov, dpv) / np.dot(dpv, dpv)) * dpv  # orthogonalize
            ov = ov / np.linalg.norm(ov)
            d_alpha, d_beta, d_delta = ov

            # Reverse previous update & update in conjugate direction
            iteration += 1
            # alpha, beta, delta = optim_grad_step(alpha_best, beta_best, delta_best, d_alpha, d_beta, d_delta, stepsize=0.1)
            alpha, beta, delta = optim_adapt_step(alpha_best, beta_best, delta_best, d_alpha, d_beta, d_delta, default_step=0.1)

        while (alpha <= 0.01) or (beta <= 0.01) or (delta <= 0.01):  # if below 0
            rv = np.random.randn(len(pv_best))  # radius
            rv = rv / np.linalg.norm(rv)
            pv = pv_best + 0.01 * rv  # perturb within a 0.01 radius ball
            alpha, beta, delta = pv

        logger.info(f'[{patience:d}:{iteration:d}][{Tk:d}/{len(sample_frames):d}] NEW: alpha = {alpha:.4f}   beta = {beta:.4f}   '
                    f'delta = {delta:.4f}')
        logger.info('-' * 110)

    # Record best
    logger.info('-' * 110)
    best_results['iteration'].append(iteration_best)
    best_results['Rk'].append(Rk_best)
    best_results['alpha'].append(alpha_best)
    best_results['beta'].append(beta_best)
    best_results['delta'].append(delta_best)
    best_results['sol'].append(sol_best)
    best_results['u0'].append(u0_best)
    best_results['R'].append(R_best)
    logger.info(f'[{Tk:d}/{len(sample_frames):d}] BEST@{iteration_best:d}: L2 Residual = {Rk_best:.4f} | alpha = {alpha_best:.4f}   beta = {beta_best:.4f}   delta = {delta_best:.4f}')
    logger.info('*' * 110)

    # plt.figure()
    # plt.plot(sol_best[:, 0], flow, 'k.-')
    # plt.plot(sol_best[:, 0], u0_best, 'b.-')
    # plt.plot(sol_best[:, 0], R_best, 'r.-')
    # plt.xlabel('t')
    # plt.legend(['glottal flow', 'u0', 'R'])
    # plt.figure()
    # plt.subplot(121)
    # plt.plot(sol_best[:, 1], sol_best[:, 3], 'b.-')
    # plt.xlabel(r'$\xi_r$')
    # plt.ylabel(r'$\xi_l$')
    # plt.subplot(122)
    # plt.plot(sol_best[:, 2], sol_best[:, 4], 'b.-')
    # plt.xlabel(r'$\dot{\xi}_r$')
    # plt.ylabel(r'$\dot{\xi}_l$')
    # plt.tight_layout()
    # plt.show()

logger.info('*' * 110)
logger.info('*' * 110)
logger.info(best_results)
