# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pdb
for path in [
        'models/vocal_fold',
        'solvers/ode_solvers'
]:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))),
            path))
from vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from adjoint_model_displacement import adjoint_model
from ode_solver import ode_solver
from dae_solver import dae_solver

# Initial conditions
beta = 0.20
alpha_low = 0
alpha_high = 1.5
delta_low = 0
delta_high = 2

grid_size = 200
t_max = 500

vdp_init_t = 0.
vdp_init_state = [0., 0.1, 0., 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

ns = []  # storing number of intersections for right oscillator
ms = []  # storing number of intersections for left oscillator

for alpha in np.linspace(alpha_low, alpha_high, grid_size):
    for delta in np.linspace(delta_low, delta_high, grid_size):

        vdp_params = [alpha, beta, delta]
        # vdp_params = [0.64, 0.32, 0.16]  # normal
        # vdp_params = [0.64, 0.32, 1.6]  # torus
        # vdp_params = [0.7, 0.32, 1.6]  # two cycle
        # vdp_params = [0.8, 0.32, 1.6]  # one cycle

        # Solve vocal fold displacement model
        sol = ode_solver(vdp_coupled, vdp_jacobian, vdp_params,
                        vdp_init_state, vdp_init_t,
                        solver='lsoda', ixpr=0,
                        dt=1,
                        tmax=t_max
                        )

        # Get steady state
        Sr = sol[int(t_max / 2):, [1, 2]]  # right states, (xr, dxr)
        Sl = sol[int(t_max / 2):, [3, 4]]  # left states, (xl, dxl)

        # Plot states
        # plt.figure()
        # plt.subplot(121)
        # plt.plot(Sr[:, 0], Sr[:, 1], 'b.-')
        # # plt.xlabel(r'$\xi$')
        # # plt.ylabel(r'$\dot{\xi}$')
        # plt.xlabel(r'$\xi_r$')
        # plt.ylabel(r'$\dot{\xi}_r$')
        # plt.subplot(122)
        # plt.plot(Sl[:, 0], Sl[:, 1], 'b.-')
        # plt.xlabel(r'$\xi_l$')
        # plt.ylabel(r'$\dot{\xi}_l$')
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('2_cycle_07-032-16.png')

        # Poincare section at dx = 0
        n = 0
        m = 0

        # px_r = []  # storing crossing pts coord x for right oscillator
        # py_r = []  # storing crossing pts coord y for right oscillator
        # px_l = []
        # py_l = []

        for k in range(len(Sr) - 1):  # loop over time
            if Sr[k, 1] >= 0 and Sr[k + 1, 1] <= 0:  # in direction decreasing dx
                n += 1  # add 1 crossing

                # s = -Sr[k, 1] / (Sr[k + 1, 1] - Sr[k, 1])  # 0 = S[k] + s * (S[k+1] - S[k])
                # rx = (1 - s) * Sr[k, 0] + s * Sr[k + 1, 0]  # interpolation
                # ry = (1 - s) * Sr[k, 1] + s * Sr[k + 1, 1]
                # px_r.append(rx)
                # py_r.append(ry)
                # ns.append()

            if Sl[k, 1] >= 0 and Sl[k + 1, 1] <= 0:  # in direction decreasing dx
                m += 1  # add 1 crossing

                # s = -Sl[k, 1] / (Sl[k + 1, 1] - Sl[k, 1])  # 0 = S[k] + s * (S[k+1] - S[k])
                # lx = (1 - s) * Sl[k, 0] + s * Sl[k + 1, 0]  # interpolation
                # ly = (1 - s) * Sl[k, 1] + s * Sl[k + 1, 1]
                # px_l.append(lx)
                # py_l.append(ly)

        ns.append(n)
        ms.append(m)

        # plt.figure()
        # plt.scatter(px, py, s=2, c='b', marker='o')
        # plt.xlabel(r'$\xi_r$')
        # plt.ylabel(r'$\dot{\xi}_r$')
        # plt.tight_layout()
        # plt.show()

# Entrainment plot
entrainment_ratio = []
for n, m in zip(ns, ms):
    if m == 0:
        entrainment_ratio.append(0)
    else:
        entrainment_ratio.append(float(n) / (float(m) + 1e-9))
entrainment_ratio = np.array(entrainment_ratio).reshape(grid_size, grid_size).transpose()

plt.figure()
X = np.linspace(alpha_low, alpha_high, grid_size)
Y = np.linspace(delta_low, delta_high, grid_size)
XX, YY = np.meshgrid(X, Y)
plt.pcolormesh(YY, XX, entrainment_ratio, cmap='Greys')
plt.xlabel(r'$\Delta$')
plt.ylabel(r'$\alpha$')
plt.tight_layout()
# plt.show()
plt.savefig('bifurcation_plot_T{}_grid{}.png'.format(t_max, grid_size))
