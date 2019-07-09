# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from assimulo.problem import Implicit_Problem
import pdb


def dae_solver(residual, y0, yd0, t0,
               p0=None, jac=None, name='DAE',
               solver='IDA', algvar=None, atol=1e-6, backward=False,
               display_progress=True, pbar=None, report_continuously=True,
               rtol=1e-6, sensmethod='STAGGERED', suppress_alg=False,
               suppress_sens=False, usejac=False, usesens=False, verbosity=30,
               tfinal=10., ncp=500):
    '''
    DAE solver.

    Parameters
    ----------
    residual: function
        Implicit DAE model.
    y0: List[float]
        Initial model state.
    yd0: List[float]
        Initial model state derivatives.
    t0: float
        Initial simulation time.
    p0: List[float]
        Parameters for which sensitivites are to be calculated.
    jac: function
        Model jacobian.
    name: string
        Model name.
    solver: string
        DAE solver.
    algvar: List[bool]
        A list for defining which variables are differential and which are algebraic.
        The value True(1.0) indicates a differential variable and the value False(0.0) indicates an algebraic variable.
    atol: float
        Absolute tolerance.
    backward: bool
        Specifies if the simulation is done in reverse time.
    display_progress: bool
        Actives output during the integration in terms of that the current integration is periodically printed to the stdout.
        Report_continuously needs to be activated.
    pbar: List[float]
        An array of positive floats equal to the number of parameters. Default absolute values of the parameters.
        Specifies the order of magnitude for the parameters. Useful if IDAS is to estimate tolerances for the sensitivity solution vectors.
    report_continuously: bool
        Specifies if the solver should report the solution continuously after steps.
    rtol: float
        Relative tolerance.
    sensmethod: string
        Specifies the sensitivity solution method.
        Can be either ‘SIMULTANEOUS’ or ‘STAGGERED’. Default is 'STAGGERED'.
    suppress_alg: bool
        Indicates that the error-tests are suppressed on algebraic variables.
    suppress_sens: bool
        Indicates that the error-tests are suppressed on the sensitivity variables.
    usejac: bool
        Sets the option to use the user defined jacobian.
    usesens: bool
        Aactivates or deactivates the sensitivity calculations.
    verbosity: int
        Determines the level of the output.
        QUIET = 50 WHISPER = 40 NORMAL = 30 LOUD = 20 SCREAM = 10
    tfinal: float
        Simulation final time.
    ncp: int
        Number of communication points (number of return points).

    Returns
    -------
    sol: solution [time, model states], List[float]
    '''
    if usesens is True:  # parameter sensitivity
        model = Implicit_Problem(residual, y0, yd0, t0, p0=p0)
    else:
        model = Implicit_Problem(residual, y0, yd0, t0)

    model.name = name

    if usejac is True:  # jacobian
        model.jac = jac

    if algvar is not None:  # differential or algebraic variables
        model.algvar = algvar

    if solver == 'IDA':  # solver
        from assimulo.solvers import IDA
        sim = IDA(model)

    sim.atol = atol
    sim.rtol = rtol
    sim.backward = backward  # backward in time
    sim.report_continuously = report_continuously
    sim.display_progress = display_progress
    sim.suppress_alg = suppress_alg
    sim.verbosity = verbosity

    if usesens is True:  # sensitivity
        sim.sensmethod = sensmethod
        sim.pbar = np.abs(p0)
        sim.suppress_sens = suppress_sens

    # Simulation
    t, y, yd = sim.simulate(tfinal, ncp)

    # Plot
    # sim.plot()

    # plt.figure()
    # plt.subplot(221)
    # plt.plot(t, y[:, 0], 'b.-')
    # plt.legend([r'$\lambda$'])
    # plt.subplot(222)
    # plt.plot(t, y[:, 1], 'r.-')
    # plt.legend([r'$\dot{\lambda}$'])
    # plt.subplot(223)
    # plt.plot(t, y[:, 2], 'k.-')
    # plt.legend([r'$\eta$'])
    # plt.subplot(224)
    # plt.plot(t, y[:, 3], 'm.-')
    # plt.legend([r'$\dot{\eta}$'])
    # plt.show()

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(y[:, 0], y[:, 2])
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$\eta$')
    # plt.subplot(122)
    # plt.plot(y[:, 1], y[:, 3])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\dot{\eta}$')
    # plt.show()

    sol = [t, y, yd]
    return sol
