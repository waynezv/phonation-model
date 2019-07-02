# -*- coding: utf-8 -*-
import pdb


def vocal_fold_model(U, t, alpha, beta, delta):
    '''
    Model of volume velocity flow through vocal folds.

    Parameters
    ----------
    t: time
    U: variables [u1(t), u2(t)]
    alpha: glottal pressure coupling parameter
    beta: mass, elastic, damping parameter
    delta: asymmetry parameter

    Returns
    -------
    dU: system derivative [du1, du2]
    '''
    du1 = U[1]

    du2 = -(beta - 2 * alpha) * U[1] - (delta + 1) * U[0]

    dU = [du1, du2]
    return dU


def vdp_jacobian(U, t, alpha, beta, delta):
    '''
    Jacobian of the above system.
    J[i, j] = d(dU[i]) / d(U[j])
    '''
    J = [[0, 1],
         [-(delta + 1), -(beta - 2 * alpha)]]

    return J
