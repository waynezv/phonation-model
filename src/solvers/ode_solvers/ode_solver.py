# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode
import pdb


def ode_solver(model, model_jacobian, model_params, init_state, init_t,
               solver='lsoda', ixpr=1,
               dt=0.1, tmax=1000):
    '''
    ODE solver.

    Parameters
    ----------
    model: ODE model dy = f(t, y)
    model_jacobian: Jacobian of ODE model
    model_params: model parameters, List[float]
    init_state: initial model state, List[float]
    init_t: initial simulation time, float
    solver: ODE solver, string
        Options: vode, dopri5, dop853, lsoda; depends on stiffness and precision.
    ixpr: whether to generate extra printing at method switches, int
    dt: time step increment, float
    tmax: maximum simulation time, float

    Returns
    -------
    sol: solution [time, model states], np.array[float]
    '''
    sol = []

    r = ode(model, model_jacobian)
    r.set_f_params(*model_params)
    r.set_jac_params(*model_params)

    r.set_initial_value(init_state, init_t)

    r.set_integrator(solver,
                     with_jacobian=True,
                     ixpr=ixpr)

    while r.successful() and r.t < tmax:
        r.integrate(r.t + dt)
        sol.append([r.t, *list(r.y)])

    return np.array(sol)  # (t, [p, dp]) tangent bundle
