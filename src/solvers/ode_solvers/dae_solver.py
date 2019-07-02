# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from ode_model import vdp_coupled, vdp_jacobian
import pdb


y0 = [0.1, 0, 0.1, 0]
t0 = 0
tmax = 1000
dt = 0.1
sol = []

r = ode(vdp_coupled, vdp_jacobian)
r.set_integrator('lsoda',
                 with_jacobian=True,
                 ixpr=1)
r.set_f_params(0.8, 0.32, 1)
r.set_jac_params(0.8, 0.32, 1)

r.set_initial_value(y0, t0)

while r.successful() and r.t < tmax:
    r.integrate(r.t + dt)
    sol.append([r.t, *list(r.y)])

sol = np.array(sol)  # dim: t, xr, dxr, xl, dxl

plt.figure()
plt.plot(sol[:1000, 0], sol[:1000, 1], 'b.-')
plt.plot(sol[:1000, 0], sol[:1000, 3], 'r.-')
plt.xlabel('t')
plt.ylabel('x')
plt.show()

plt.figure()
plt.plot(sol[:, 2], sol[:, 4], 'b.-')
plt.xlabel('du')
plt.ylabel('dv')
plt.show()
