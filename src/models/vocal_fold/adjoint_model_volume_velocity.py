# -*- coding: utf-8 -*-

import numpy as np


def adjoint_model(t, L, dL, alpha, beta, delta, Ru_Sm_duR):
    '''
    Adjoint model for vocal fold model.
    Used to solve derivatives of volume velocity to model parameters.

    Parameters
    ----------
    t: time
    L: variables [l1(t), l2(t)]
    dL: derivative of variables [dl1(t), dl2(t)]
    alpha: glottal pressure coupling parameter
    beta: mass, elastic, damping parameter
    delta: asymmetry parameter
    Ru_Sm_duR: extra terms required

    Returns
    -------
    res: numpy.array
        Residual vector.
    '''
    res_1 = dL[1] + (delta + 1) * L[0] + 2 * Ru_Sm_duR

    res_2 = (beta - 2 * alpha) * L[0]
    return np.array([res_1, res_2])
