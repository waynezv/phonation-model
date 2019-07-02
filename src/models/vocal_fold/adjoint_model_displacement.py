# -*- coding: utf-8 -*-

import numpy as np


def adjoint_model(t, M, dM, alpha, beta, delta, X, dX, R):
    '''
    Adjoint model for vocal fold model.
    Used to solve derivatives of left/right vocal fold displacements
        w.r.t. model parameters (alpha, beta, delta).

    Parameters
    ----------
    t: time
    M: state variables [L, dL, E, dE]
    dM: derivative of state variables [dL, ddL, dE, ddE]
    alpha: glottal pressure coupling parameter
    beta: mass, elastic, damping parameter
    delta: asymmetry parameter
    X: vocal fold displacements [x_l, x_r]
    dX: vocal fold velocity [dx_l, dx_r]
    R: extra terms required

    Returns
    -------
    res: numpy.array
        Residual vector.
    '''
    res_1 = dM[1] + (2 * beta * X[1] * dX[1] + 1 - 0.5 * delta) * M[0] + R

    res_2 = dM[3] + (2 * beta * X[0] * dX[0] + 1 + 0.5 * delta) * M[2] + R

    res_3 = beta * (M[2] * (1 + X[0]) ** 2 - M[0] * (1 + X[1]) ** 2)
    return np.array([res_1, res_2, res_3])
