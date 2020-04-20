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
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Normal/001.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Neoplasm/001.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Phonotrauma/001.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Vocal_palsy/001.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Normal/034.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Neoplasm/033.8k.wav'
# wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Phonotrauma/056.8k.wav'
wav_file = '../../data/FEMH_Data/processed/resample_8k/Training_Dataset/Pathological/Vocal_palsy/035.8k.wav'
samples, fs = pcm16_to_float(wav_file)
assert fs == 8000, "{}: incompatible sample rate"\
    "--need 8000 but got {}".format(wav_file, fs)

# Trim
# samples = samples[int(4 * fs): int(6 * fs)]  # for normal, neoplasm, phonotrauma
# samples = samples[int(4 * fs): int(4.1 * fs)]  # for normal, neoplasm, phonotrauma
# samples = samples[int(1 * fs): int(3 * fs)]  # for vocal_palsy
samples = samples[int(1 * fs): int(1.1 * fs)]  # for vocal_palsy

samples = samples / np.linalg.norm(samples)  # normalize

# Inverse filter
# temporarily use glottal flow computed by covarep toolbox in Matlab
# glottal_flow = scipy.io.loadmat('./Normal_001_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
# glottal_flow = scipy.io.loadmat('./Neoplasm_001_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
# glottal_flow = scipy.io.loadmat('./Phonotrauma_001_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
# glottal_flow = scipy.io.loadmat('./Vocal_palsy_001_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
# glottal_flow = scipy.io.loadmat('./new_tests_icassp/Normal_034_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
# glottal_flow = scipy.io.loadmat('./new_tests_icassp/Neoplasm_033_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
# glottal_flow = scipy.io.loadmat('./new_tests_icassp/Phonotrauma_056_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))
glottal_flow = scipy.io.loadmat('./new_tests_icassp/Vocal_palsy_035_8k_glottal_volume_flow_iaif_ola_sync.mat')['gf_iaif_ola_sync'].reshape((-1,))

# Trim
# glottal_flow = glottal_flow[int(4 * fs): int(6 * fs)]  # for normal, neoplasm, phonotrauma
# glottal_flow = glottal_flow[int(4 * fs): int(4.1 * fs)]  # for normal, neoplasm, phonotrauma
# glottal_flow = glottal_flow[int(1 * fs): int(3 * fs)]  # for vocal_palsy
glottal_flow = glottal_flow[int(1 * fs): int(1.1 * fs)]  # for vocal_palsy

glottal_flow = glottal_flow / np.linalg.norm(glottal_flow)  # normalize

assert len(glottal_flow) == len(samples), "Inconsistent length: glottal flow ({:d}) / wav file ({:d})".format(len(glottal_flow), len(samples))


# fig = plt.figure()
# plt.plot(np.linspace(0, len(samples) / fs, len(samples)), samples)
# plt.plot(np.linspace(0, len(samples) / fs, len(samples)), glottal_flow)
# plt.legend(['speech sample', 'glottal flow'])
# plt.show()

# Initial conditions
# alpha = 0.8  # if > 0.5 delta, stable-like oscillator
# beta = 0.32
# delta = 1.  # asymmetry parameter

alpha = 0.45
beta = 0.28
delta = 0.75

vdp_init_t = 0.
vdp_init_state = [0., 0.1, 0., 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

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

Rk = 1e16  # prediction residual w.r.t L2 norm @ k

num_tsteps = len(samples)  # number of time steps
T = len(samples) / float(fs)  # total time, s

logger.info('Initial parameters: alpha = {:.4f}   beta = {:.4f}   '
            'delta = {:.4f}'.format(alpha, beta, delta))
logger.info('-' * 110)

iteration = 0

# Adam optimizer
adam_beta_1 = 0.9  # decay for grad
adam_beta_2 = 0.999  # decay for grad^2
adam_eps = 1e-8
eta_alpha = 0.01  # learning rate
m_t_alpha = 0  # weighted average of grad
v_t_alpha = 0  # weighted average of grad^2
eta_beta = 0.01
m_t_beta = 0
v_t_beta = 0
eta_delta = 0.01
m_t_delta = 0
v_t_delta = 0

while Rk > 0.1:
    iteration += 1

    logger.info('Solving vocal fold displacement model')

    vdp_params = [alpha, beta, delta]

    K = B ** 2 / (beta ** 2 * M)
    Ps = (alpha * x0 * np.sqrt(M * K)) / tau

    time_scaling = np.sqrt(K / float(M))  # t -> s
    x_scaling = np.sqrt(eta)

    logger.debug('stiffness K = {:.4f} dyne/cm^3    subglottal Ps = {:.4f} '
                 'dyne/cm^2    time_scaling = {:.4f}'.
                 format(K, Ps, time_scaling))

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
    plt.figure()
    plt.subplot(121)
    plt.plot(sol[:, 1], sol[:, 3], 'b.-')
    plt.xlabel(r'$\xi_r$')
    plt.ylabel(r'$\xi_l$')
    plt.subplot(122)
    plt.plot(sol[:, 2], sol[:, 4], 'b.-')
    plt.xlabel(r'$\dot{\xi}_r$')
    plt.ylabel(r'$\dot{\xi}_l$')
    plt.tight_layout()
    # plt.show()
    # plt.savefig('/home/wzhao/ProJEX/phonation-model/src/main_scripts/outputs/icassp_plots/phase_plot_vocalpalsy.png')

    logger.info('Solving adjoint model')

    # Calculate some terms
    # BUG: len > 1
    if len(sol) > len(samples):
        sol = sol[:-1]
    assert len(sol) == len(samples), "Inconsistent length: ODE sol ({:d}) / "\
        "wav file ({:d})".format(len(sol), len(samples))

    X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
    dX = sol[:, [2, 4]]  # cm/s
    u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
    u0 = u0 / np.linalg.norm(u0)  # normalize
    u0 = u0[:num_tsteps]

    R = u0 - glottal_flow

    # Plot glottal flow
    # plt.figure()
    # plt.plot(np.linspace(0, T, len(u0)), u0)
    # plt.xlabel('t')
    # plt.ylabel('u0')

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(sol[int(0.9 * fs):int(1.05 * fs), 0], glottal_flow[int(0.9 * fs):int(1.05 * fs)], 'b.-')
    # plt.ylabel('glottal flow')
    # plt.subplot(312)
    # plt.plot(sol[int(0.9 * fs):int(1.05 * fs), 0], u0[int(0.9 * fs):int(1.05 * fs)], 'b.-')
    # plt.ylabel('u0')
    # plt.subplot(313)
    # plt.plot(sol[int(0.9 * fs):int(1.05 * fs), 0], R[int(0.9 * fs):int(1.05 * fs)], 'b.-')
    # plt.ylabel('R')
    # plt.xlabel('t')

    # plt.figure()
    # plt.plot(sol[int(0.9 * fs):int(1.05 * fs), 0], glottal_flow[int(0.9 * fs):int(1.05 * fs)], 'k.-')
    # plt.plot(sol[int(0.9 * fs):int(1.05 * fs), 0], u0[int(0.9 * fs):int(1.05 * fs)], 'b.-')
    # # plt.plot(sol[int(0.9 * fs):int(1.05 * fs), 0], R[int(0.9 * fs):int(1.05 * fs)], 'r.-')
    # plt.xlabel('t')
    # # plt.legend(['glottal flow', 'u0', 'R'])
    # plt.legend(['glottal flow', 'u0'])

    plt.figure()
    # plt.plot(np.linspace(4, 4.1, len(glottal_flow)), glottal_flow[:], 'k--', lw=1.5)
    # plt.plot(np.linspace(4, 4.1, len(glottal_flow)), u0[:], 'b-', lw=1.5)
    plt.plot(np.linspace(1, 1.1, len(glottal_flow)), glottal_flow[:], 'k--', lw=1.5)
    plt.plot(np.linspace(1, 1.1, len(glottal_flow)), u0[:], 'b-', lw=1.5)
    # plt.plot(sol[:, 0], R[:], 'r.-')
    plt.xlabel('t')
    # plt.legend(['glottal flow', 'u0', 'R'])
    plt.legend(['inverse filter', 'ADLES'])
    plt.tight_layout()
    plt.show()
    # plt.savefig('/home/wzhao/ProJEX/phonation-model/src/main_scripts/outputs/icassp_plots/glottal_flow_vocalpalsy.png')
    pdb.set_trace()

    # Solve adjoint model
    residual, jac = adjoint_model(alpha, beta, delta, X, dX, R,
                                  num_tsteps/T)

    M_T = [0., 0., 0., 0.]  # initial states of the adjoint model at T
    dM_T = [0., -R[-1], 0., -R[-1]]  # initial ddL = ddE = -R(T)

    adjoint_sol = dae_solver(residual, M_T, dM_T, T,
                             solver='IDA', algvar=[0, 1, 0, 1], suppress_alg=True, atol=1e-6, rtol=1e-6,
                             usejac=True, jac=jac, usesens=False,
                             backward=True, tfinal=0., ncp=(len(samples)-1),  # simulate (T --> 0)s backwards
                             display_progress=True, report_continuously=False, verbosity=50)  # NOTE: report_continuously should be False

    # Update parameters
    L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
    E = adjoint_sol[1][:, 2][::-1]
    assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"

    L = L / np.linalg.norm(L)
    E = E / np.linalg.norm(E)

    # Gradients
    d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
    d_beta = np.sum(L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0] + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1])
    d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))

    # Adaptive stepsize
    # stepsize = 0.1
    # stepsize = 0.01 / np.max([d_alpha, d_beta, d_delta])
    # if (alpha - stepsize * d_alpha) > 0 and (alpha - stepsize * d_alpha) < 5:  # (0, 5)
        # alpha = alpha - stepsize * d_alpha
    # else:
        # continue
    # if (beta - stepsize * d_beta) > 0 and (beta - stepsize * d_beta) < 5:  # (0, 5)
        # beta = beta - stepsize * d_beta
    # else:
        # continue
    # if (delta - stepsize * d_delta) > 0 and (delta - stepsize * d_delta) < 2:  # (0, 2)
        # delta = delta - stepsize * d_delta
    # else:
        # continue

    # Adam optimize
    m_t_alpha = adam_beta_1 * m_t_alpha + (1 - adam_beta_1) * d_alpha
    v_t_alpha = adam_beta_2 * v_t_alpha + (1 - adam_beta_2) * (d_alpha * d_alpha)
    m_cap_alpha = m_t_alpha / (1 - (adam_beta_1 ** iteration))  # correct bias
    v_cap_alpha = v_t_alpha / (1 - (adam_beta_2 ** iteration))
    alpha_prev = alpha
    alpha = alpha - (eta_alpha * m_cap_alpha) / (np.sqrt(v_cap_alpha) + adam_eps)
    if(alpha == alpha_prev):  #checks if it is converged or not
        print('Alpha not converging!')
        break

    m_t_beta = adam_beta_1 * m_t_beta + (1 - adam_beta_1) * d_beta
    v_t_beta = adam_beta_2 * v_t_beta + (1 - adam_beta_2) * (d_beta * d_beta)
    m_cap_beta = m_t_beta / (1 - (adam_beta_1 ** iteration))  # correct bias
    v_cap_beta = v_t_beta / (1 - (adam_beta_2 ** iteration))
    beta_prev = beta
    beta = beta - (eta_beta * m_cap_beta) / (np.sqrt(v_cap_beta) + adam_eps)
    if(beta == beta_prev):  #checks if it is converged or not
        print('Beta not converging!')
        break

    m_t_delta = adam_beta_1 * m_t_delta + (1 - adam_beta_1) * d_delta
    v_t_delta = adam_beta_2 * v_t_delta + (1 - adam_beta_2) * (d_delta * d_delta)
    m_cap_delta = m_t_delta / (1 - (adam_beta_1 ** iteration))  # correct bias
    v_cap_delta = v_t_delta / (1 - (adam_beta_2 ** iteration))
    delta_prev = delta
    delta = delta - (eta_delta * m_cap_delta) / (np.sqrt(v_cap_delta) + adam_eps)
    if(delta == delta_prev):  #checks if it is converged or not
        print('Delta not converging!')
        break

    Rk = np.sqrt(np.sum(R ** 2))

    logger.info('L2 Residual = {:.4f} | alpha = {:.4f}   beta = {:.4f}   '
                'delta = {:.4f}'.format(Rk, alpha, beta, delta))
    logger.info('-' * 110)

# Results
logger.info('*' * 110)

logger.info('alpha = {:.4f}   beta = {:.4f}   delta = {:.4f}  '
            'L2 Residual = {:.4f}'.format(alpha, beta, delta, Rk))
