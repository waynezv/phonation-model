# -*- coding: utf-8 -*-
import pdb


def vdp_coupled(t, Z, alpha, beta, delta):
    '''
    Coupled van der Pol oscillator of the explicit form
        dZ = f(Z).
    Second order, nonlinear, constant coefficients, inhomogeneous.

    Parameters
    ----------
    t: time
    Z: state variables [u1(t), u2(t), v1(t), v2(t)], u right, v left.
    alpha: force related parameter
    beta: system related parameter
    delta: asymmetry parameter

    Returns
    -------
    dZ: system derivative [du1, du2, dv1, dv2]
    '''
    du1 = Z[1]

    dv1 = Z[3]

    du2 = -beta * (1 + Z[0] ** 2) * Z[1] - (1 - delta / 2) * Z[0] +\
        alpha * (Z[1] + Z[3])

    dv2 = -beta * (1 + Z[2] ** 2) * Z[3] - (1 + delta / 2) * Z[2] +\
        alpha * (Z[1] + Z[3])

    dZ = [du1, du2, dv1, dv2]
    return dZ


def vdp_jacobian(t, Z, alpha, beta, delta):
    '''
    Jacobian of the above system of the form
        J[i, j] = df[i] / dZ[j].
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
