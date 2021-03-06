# -*- coding: utf-8 -*-

import numpy as np
import pdb


def adjoint_model(alpha, beta, delta, X, dX, R, fs, t0, tf):
    '''
    Adjoint model for the 1-d vocal fold oscillation model.
    Used to solve derivatives of left/right vocal fold displacements
        w.r.t. model parameters (alpha, beta, delta).

    Parameters
    ----------
    alpha: float
        Glottal pressure coupling parameter.
    beta: float
        Mass, elastic, damping parameter.
    delta: float
        Asymmetry parameter.
    X: List[float]
        Vocal fold displacements [x_r, x_l].
    dX: List[fliat]
        Vocal fold velocity [dx_r, dx_l].
    R: List[float]
        Extra term c.r.t. the difference between
        predicted and real volume velocity flows.
    fs: int
        Sample rate.
    t0: float
        Start time.
    tf: float
        Stop time.

    Returns
    -------
    residual: function
        Defines the adjoint model.
    jac: function
        Jacobian of the adjoint model.
    '''

    def residual(t, M, dM):
        '''
        Defines the implicit problem, which should be of the form
            0 <-- res = F(t, M, dM).

        Parameters
        ----------
        t: float
            Time.
        M: List[float]
            State variables [L, dL, E, dE].
        dM: List[float]
            Derivative of state variables [dL, ddL, dE, ddE].

        Returns
        -------
        res: np.array[float], shape (len(M),)
            Residual vector.
        '''
        # Convert t to [0, T]
        t = t - t0
        # Convert t(s) --> idx(#sample)
        idx = int(round(t * fs) - 1)
        if idx < 0:
            idx = 0
        # print(f't: {t:.4f}    adjoint idx: {idx:d}')

        x = X[idx]
        dx = dX[idx]
        r = R[idx]

        res_1 = dM[1] + (2 * beta * x[0] * dx[0] + 1 - 0.5 * delta) * M[0] + r

        res_2 = beta * M[0] * (1 + x[0] ** 2) - alpha * (M[0] + M[2])

        res_3 = dM[3] + (2 * beta * x[1] * dx[1] + 1 + 0.5 * delta) * M[2] + r

        res_4 = beta * M[2] * (1 + x[1] ** 2) - alpha * (M[0] + M[2])

        res = np.array([res_1, res_2, res_3, res_4])
        return res

    def jac(c, t, M, Md):
        '''
        Defines the Jacobian, which should be of the form
            J = dF/dM + c*dF/d(dM).

        Parameters
        ----------
        c: float
            Constant.
        t: float
            Time.
        M: List[float]
            State variables [L, dL, E, dE].
        dM: List[float]
            Derivative of state variables [dL, ddL, dE, ddE].

        Returns
        -------
        jacobian: np.array[float], shape (len(M), len(M))
            Jacobian matrix.
        '''
        # Convert t to [0, T]
        T = tf - t0
        t = (t - t0) / (tf - t0) * T
        # Convert t(s) --> idx(#sample)
        idx = int(round(t * fs) - 1)
        if idx < 0:
            idx = 0

        x = X[idx]
        dx = dX[idx]

        jacobian = np.zeros((len(M), len(M)))

        # jacobian[0, 0] = 2 * beta * x[0] * dx[0] + 1 - 0.5 * delta
        # jacobian[0, 1] = c
        # jacobian[1, 2] = 2 * beta * x[1] * dx[1] + 1 + 0.5 * delta
        # jacobian[1, 3] = c
        # jacobian[2, 0] = beta * (1 + x[0] ** 2) - alpha
        # jacobian[2, 2] = -alpha
        # jacobian[3, 0] = -alpha
        # jacobian[3, 2] = beta * (1 + x[1] ** 2) - alpha

        jacobian[0, 0] = 2 * beta * x[0] * dx[0] + 1 - 0.5 * delta
        jacobian[0, 1] = c

        jacobian[1, 0] = beta * (1 + x[0] ** 2) - alpha
        jacobian[1, 2] = -alpha

        jacobian[2, 2] = 2 * beta * x[1] * dx[1] + 1 + 0.5 * delta
        jacobian[2, 3] = c

        jacobian[3, 0] = -alpha
        jacobian[3, 2] = beta * (1 + x[1] ** 2) - alpha

        return jacobian

    return residual, jac
