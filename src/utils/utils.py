# -*- coding: utf-8 -*-

import wave
import struct
import numpy as np
import webrtcvad
from matplotlib import pyplot as plt
import pdb

def pcm16_to_float(wav_file):
    '''
    Convert 16-bit signed integer PCM wav samples to
    float array between [-1, 1].

    Parameters
    ----------
    wav_file: string
        Path to wav file.

    Returns
    -------
    samples: np.array[float]
        Float samples.
    fs: int
        Sample rate.
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
    samples = np.array(a, dtype=float)

    return samples, fs


def voice_activity_detection(raw_samples, sample_rate,
                             window_duration=30,
                             bytes_per_sample=2,
                             vad_mode=3
                             ):
    '''
    VAD using webrtcvad.

    Parameters
    ----------
    raw_samples: np.array[int16]
        Raw 16-bit signed integer PCM wav samples.
    sample_rate: int
        Support 8000, 16000, 32000, or 48000 Hz.
    window_duration: float
        In milisecond. Support 10, 20, or 30 ms.
    bytes_per_sample: int
        2 bytes, 16 bits.
    vad_mode: int
        Aggressiveness [0, 3], 0 is the least aggressive about filtering out non-speech,
        3 is the most aggressive.

    Returns
    -------
    segments: List[dict]
        List of segments marking start, stop, is_speech.
    '''

    assert sample_rate in (8000, 16000, 32000, 48000), "Non-supported sample rate!"
    assert window_duration in (10, 20, 30), "Non-supported window duration!"
    assert bytes_per_sample == 2, "Non-supported bytes_per_sample!"
    assert vad_mode in (0, 1, 2, 3), "Non-supported vad mode!"

    # Convert samples to raw 16 bit per sample stream needed by webrtcvad
    bit_stream = struct.pack("%dh" % len(raw_samples), *raw_samples)

    # Start vad
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)

    segments = []
    samples_per_window=int(window_duration / float(1000) * sample_rate + 0.5)

    for start in np.arange(0, len(raw_samples), samples_per_window):
        stop = start + samples_per_window
        if stop > len(raw_samples):
            break
        is_speech = vad.is_speech(bit_stream[start * bytes_per_sample:stop * bytes_per_sample],
                                  sample_rate=sample_rate)
        segments.append(dict(
            start=start,
            stop=stop,
            is_speech=is_speech))

    # Plot segments identifed as speech
    # ymax = max(raw_samples)
    # plt.figure()
    # plt.plot(raw_samples, 'b-.')
    # for segment in segments:
        # if segment['is_speech']:
            # plt.plot([segment['start'], segment['stop'] - 1], [ymax * 1.1, ymax * 1.1], color='orange')
    # plt.grid()
    # plt.show()

    return segments


def optim_grad_step(alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.01):
    '''
    Perform one step of gradient descent for model parameters.
    '''
    alpha = alpha - stepsize * d_alpha
    beta = beta - stepsize * d_beta
    delta = delta - stepsize * d_delta
    return alpha, beta, delta


def optim_adapt_step(alpha, beta, delta, d_alpha, d_beta, d_delta, default_step=0.01):
    '''
    Perform one step of gradient descent for model parameters.
    Stepsize is adaptive.
    '''
    stepsize = default_step / np.max([d_alpha, d_beta, d_delta])

    if (alpha - stepsize * d_alpha) > 0 and (alpha - stepsize * d_alpha) < 2:
        alpha = alpha - stepsize * d_alpha

    if (beta - stepsize * d_beta) > 0 and (beta - stepsize * d_beta) < 2:
        beta = beta - stepsize * d_beta

    if (delta - stepsize * d_delta) > 0 and (delta - stepsize * d_delta) < 2:
        delta = delta - stepsize * d_delta

    return alpha, beta, delta


def optim_adam(p, dp, m_t, v_t, itr, eta=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8):
    '''
    Perform Adam update.

    Parameters
    ----------
    p: float
        Parameter.
    dp: float
        Gradient.
    m_t: float
        Moving average of gradient.
    v_t: float
        Moving average of gradient squared.
    itr: float
        Iteration.
    eta: float
        Learning rate.
    beta_1: float
        Decay for gradient.
    beta_2: float
        Decay for gradient squared.
    eps: float
        Tolerance.

    Returns
    -------
    p: float
        Updated parameter.
    '''
    m_t = beta_1 * m_t + (1 - beta_1) * dp
    v_t = beta_2 * v_t + (1 - beta_2) * (dp * dp)
    m_cap = m_t / (1 - (beta_1 ** itr))  # correct bias
    v_cap = v_t / (1 - (beta_2 ** itr))
    p = p - (eta * m_cap) / (np.sqrt(v_cap) + eps)
    return p
