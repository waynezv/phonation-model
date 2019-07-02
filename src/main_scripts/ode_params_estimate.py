# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import numpy as np
from scipy.signal import hilbert, correlate, fftconvolve
from scipy.io import wavfile as siowav
from scipy import integrate, interpolate
from scipy import optimize
import matplotlib
# matplotlib.use('Agg')
from matplotlib.mlab import find
from matplotlib import pyplot as plt
from ode_model import vdp_coupled, vdp_jacobian
import pdb
for path in [
        'utils'
]:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))),
            path))
from sigproc import framesig


# Data
wav_root = '../../data/FEMH_Data/processed/resample_8k'
wav_list = './FEMH_data_8k.lst'

# Initial settings
y0 = [0, 0.1, 0, 0.1]
params0 = (0.25, 0.32, 0.30)


def envelope(x):
    '''
    Compute the analytic signal and the envelope.

    Parameters
    ----------
    x: np.array[float], shape (N,)
        The input signal.

    Returns
    -------
    xa: np.array[complex], shape (N,)
        The analytic signal.
    envelope: np.array[float], shape (N,)
        The envelope.
    '''
    x = x.astype('float')
    xa = hilbert(x)  # analytic signal
    envelope = np.abs(xa)
    return xa, envelope


def cross_correlation(x, y):
    '''
    Compute the cross-correlation of the two signal x and y.

    Parameters
    ----------
    x: np.array, shape (N,)
    y: np.array, shape (M,), N >= M

    Returns
    -------
    Rxy: np.array, shape (N,)
    '''
    Rxy = np.correlate(x, y, 'full')
    return Rxy


def generate_flow(params, t):
    '''
    Generate the glottal area flow via the ODE system.

    Parameters
    ----------
    params: np.array[float]
        Parameters to ODE system.
    t: np.array[float]
        Time sequence.

    Returns
    -------
    flow:
        Glottal area flow.
    '''
    r = integrate.odeint(vdp_coupled, y0, t,
                         args=tuple(params), Dfun=vdp_jacobian)

    flow = r[:, 0] + r[:, 2]
    return r, flow


# Plot glottal area flow
# t = np.linspace(0, 250, 1000)
# pm = (0.4, 0.32, 1.6)
# r, flow = generate_flow(pm, t)
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(211)
# ax1.plot(t, flow, c='b', ls='-', lw=1.5)
# ax1.set_xlabel('t', fontsize=12)
# ax1.set_ylabel(r'$x_1 + x_2$', fontsize=12)
# ax1 = fig.add_subplot(212)
# ax1.plot(t, r[:, 0], c='b', ls='-', lw=1.5)
# ax1.plot(t, r[:, 2], c='r', ls='--', lw=1.5)
# ax1.set_xlabel('t', fontsize=12)
# ax1.set_ylabel(r'$x$', fontsize=12)
# ax1.legend([r'$x_1$', r'$x_2$'], loc='best')
# ax1.axis('auto')
# plt.tight_layout()
# plt.savefig('glottal_area_flow_C2_{}_{}_{}.pdf'.
            # format(pm[0], pm[1], pm[2]))


def estimate_pitch(params, t, pitch_true, fs):
    '''
    Estimate the pitch via the ODE system.

    Parameters
    ----------
    params: np.array[float]
        Parameters to ODE system.
    t: np.array[float]
        Time sequence.
    pitch_true: np.array[float]
        True pitch.
    fs: int
        Sample rate.

    Returns
    -------
    pitch: List[float]
        Estimated pitch.
    '''
    flow = generate_flow(params, t)

    flow = flow - np.mean(flow)

    pitch = []
    for i in range(len(pitch_true)):
        flow_seg = flow[i * 1000: (i + 1) * 1000]
        # Find frequency via zero crossing
        indices, = np.nonzero(np.ravel(
            (flow_seg[1:] >= 0) & (flow_seg[:-1] < 0)))
        crossings = [j - flow_seg[j] / (flow_seg[j + 1] - flow_seg[j])
                     for j in indices]  # linear interpolate
        cycle_len = np.mean(np.diff(crossings))  # avg len per cycle
        p = fs / float(cycle_len)  # cycle/s
        p = p * 10  # TODO: decide time unit
        pitch.append(p)

    return pitch


# Least-squares residual function
def residual(params, y_true, tmax, tlen, fs):
    t = np.linspace(0, tmax, tlen)
    yh = estimate_pitch(params, t, y_true, fs)
    return y_true - np.array(yh)


def fit():
    alphas = []
    betas = []
    deltas = []

    wavfiles = [l.rstrip('\n').split()[0] for l in open(wav_list)]

    FNULL = open(os.devnull, 'w')

    for f in wavfiles:
        # Read wav
        sample_rate, samples = siowav.read(os.path.join(wav_root, f))
        assert sample_rate == 8000, "{}: incompatible sample rate"\
            " need 8000 but got {}".format(f, sample_rate)

        # Plot signal
        # fig = plt.figure(figsize=(6, 6))
        # ax1 = fig.add_subplot(111)
        # ax1.plot(samples, c='b', ls='-', lw=2)
        # ax1.set_xlabel('t', fontsize=12)
        # ax1.set_ylabel('amp', fontsize=12)
        # ax1.axis('auto')
        # plt.tight_layout()
        # plt.savefig('signal_{}.pdf'.format(f.replace('/', '_').rstrip('.wav')))

        # Plot peak of envelope correlation
        # plt.figure()
        # plt.subplot(511)
        # plt.plot(range(len(samples)), samples, 'b--')
        # plt.title('signal')
        # sig_seg = framesig(samples, sample_rate * 0.01, sample_rate * 0.01)
        # peaks = []  # peaks of envelope correlation
        # for seg in sig_seg:
            # _, env = envelope(seg)
            # Rxy = cross_correlation(env[:len(seg)//2], env[len(seg)//2:])
            # peaks.append(np.max(Rxy))
        # plt.subplot(512)
        # tk = np.linspace(0, len(samples), len(peaks))
        # plt.plot(tk, peaks, 'k-')
        # plt.title('peaks of envelope correlation (0.01s segment)')
        # sig_seg = framesig(samples, sample_rate * 0.02, sample_rate * 0.02)
        # peaks = []  # peaks of envelope correlation
        # for seg in sig_seg:
            # _, env = envelope(seg)
            # Rxy = cross_correlation(env[:len(seg)//2], env[len(seg)//2:])
            # peaks.append(np.max(Rxy))
        # plt.subplot(513)
        # tk = np.linspace(0, len(samples), len(peaks))
        # plt.plot(tk, peaks, 'k-')
        # plt.title('peaks of envelope correlation (0.02s segment)')
        # sig_seg = framesig(samples, sample_rate * 0.03, sample_rate * 0.03)
        # peaks = []  # peaks of envelope correlation
        # for seg in sig_seg:
            # _, env = envelope(seg)
            # Rxy = cross_correlation(env[:len(seg)//2], env[len(seg)//2:])
            # peaks.append(np.max(Rxy))
        # plt.subplot(514)
        # tk = np.linspace(0, len(samples), len(peaks))
        # plt.plot(tk, peaks, 'k-')
        # plt.title('peaks of envelope correlation (0.03s segment)')
        # sig_seg = framesig(samples, sample_rate * 0.04, sample_rate * 0.04)
        # peaks = []  # peaks of envelope correlation
        # for seg in sig_seg:
            # _, env = envelope(seg)
            # Rxy = cross_correlation(env[:len(seg)//2], env[len(seg)//2:])
            # peaks.append(np.max(Rxy))
        # plt.subplot(515)
        # tk = np.linspace(0, len(samples), len(peaks))
        # plt.plot(tk, peaks, 'k-')
        # plt.title('peaks of envelope correlation (0.04s segment)')
        # plt.tight_layout()
        # plt.savefig('envelope_correlation_{}.pdf'.format(f.replace('/', '_').rstrip('.wav')))

        # Extract pitch
        subprocess.run(["./histopitch -in {} -out tmp.pit -srate {}".
                        format(os.path.join(wav_root, f),
                               sample_rate)
                        ],
                       shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        pitch = np.loadtxt("tmp.pit")

        # Estimate ODE params
        tmax = 2000 * len(samples) / float(sample_rate)
        tlen = len(pitch) * 1000
        params_best = optimize.leastsq(residual, params0,
                                       args=(pitch, tmax, tlen, sample_rate))

        alphas.append(params_best[0][0])
        betas.append(params_best[0][1])
        deltas.append(params_best[0][2])

        print(f)
        print("parameter values are ", params_best)

        # Plot pitch
        #  t = np.linspace(0, tmax, tlen)
        #  pitch_est = estimate_pitch(params_best, t, pitch, sample_rate)
        #  fig = plt.figure(figsize=(12, 6))
        #  ax1 = fig.add_subplot(121)
        #  ax1.plot(pitch, c='b', ls='-', lw=2)
        #  ax1.set_xlabel('t', fontsize=12)
        #  ax1.set_ylabel('pitch', fontsize=12)
        #  ax1.set_ylim(bottom=0)
        #  ax1.axis('auto')
        #  ax1 = fig.add_subplot(122)
        #  ax1.plot(pitch_est, c='b', ls='-', lw=2)
        #  ax1.set_xlabel('t', fontsize=12)
        #  ax1.set_ylabel('pitch', fontsize=12)
        #  ax1.set_ylim(bottom=0)
        #  ax1.axis('auto')
        #  plt.tight_layout()
        #  plt.savefig('pitch_{}.pdf'.format(f.replace('/', '_').rstrip('.wav')))

    # Plot bifurcation
    # fig = plt.figure(figsize=(6, 6))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(np.abs(deltas), alphas, s=2, c='b', marker='.')
    # ax1.scatter(np.abs(deltas), betas, s=2, c='r', marker='o')
    # ax1.set_xlabel(r'$| \Delta |$', fontsize=12)
    # ax1.set_ylabel(r'$\alpha$', fontsize=12)
    # # ax1.set_xlim(0, 2)
    # # ax1.set_ylim(0, 1.5)
    # ax1.axis('auto')
    # plt.tight_layout()
    # plt.savefig('bifurcation_plot.pdf')


fit()
