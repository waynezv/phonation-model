# -*- coding: utf-8 -*-
import pdb


def vdp_coupled(Z, t, alpha, beta, delta):
    '''
    Coupled van der Pol oscillator.
    Second order, nonlinear, constant coefficients, inhomogeneous

    Parameters
    ----------
    t: time
    Z: variables [u1(t), u2(t), v1(t), v2(t)]
    alpha: force related parameter
    beta: system related parameter
    delta: asymmetry parameter

    Returns
    -------
    dZdt: system derivative [du1, du2, dv1, dv2]
    '''
    du1 = Z[1]

    dv1 = Z[3]

    du2 = -beta * (1 + Z[0] ** 2) * Z[1] - (1 - delta / 2) * Z[0] +\
        alpha * (Z[1] + Z[3])

    dv2 = -beta * (1 + Z[2] ** 2) * Z[3] - (1 + delta / 2) * Z[2] +\
        alpha * (Z[1] + Z[3])

    dZdt = [du1, du2, dv1, dv2]
    return dZdt


def vdp_jacobian(Z, t, alpha, beta, delta):
    '''
    Jacobian of the above system.
    J[i, j] = df[i] / dZ[j]
    '''
    J = [
        [0, 1, 0, 0],

        [-2 * beta * Z[1] * Z[0] - (1 - delta / 2),
         -beta * (1 + Z[0] ** 2) + alpha,
         0,
         alpha],

        [0, 0, 0, 1],

        [0,
         alpha,
         -2 * beta * Z[3] * Z[2] - (1 + delta / 2),
         -beta * (1 + Z[2] ** 2) + alpha]
    ]

    return J
