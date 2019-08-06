"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

import fenics as F
import mshr
import numpy as np
from matplotlib import pyplot as plt
import sympy
import pdb


# def solver(f, u_D, Nx, Ny, degree=1):


T = 1.0            # final time
num_steps = 500   # number of time steps
dt = T / num_steps  # time step size

tol = 1E-14

alpha = 1.
beta = 5.
c = 1.
length = 1.


# Create mesh
def UnitHyperCube(divisions):
    mesh_classes = [F.UnitIntervalMesh, F.UnitSquareMesh, F.UnitCubeMesh]
    d = len(divisions)
    mesh = mesh_classes[d - 1](*divisions)
    return mesh


# mesh = UnitHyperCube((8,))
mesh = UnitHyperCube((64,))

# Define function spaces
V = F.FunctionSpace(mesh, 'P', 1)


#
# Use SymPy to compute f from the manufactured solution u
# x, t = sympy.symbols('x[0], x[1]')
# u = 1 + alpha * (x ** 2) + beta * (t ** 2)
x_coord = np.linspace(0, length, num_steps)
t_coord = np.linspace(0, T, num_steps)
xv, tv = np.meshgrid(x_coord, t_coord)
z = 1 + alpha * (xv ** 2) + beta * (tv ** 2)
h = plt.contourf(x_coord, t_coord, z)
# plt.show()
# pdb.set_trace()
# f = sympy.diff(sympy.diff(u, t), t) - (c ** 2) * sympy.diff(sympy.diff(u, x), x)
# f = sympy.simplify(f)
# u_0 = u.subs(x, 0)
# u_L = u.subs(x, length)

# u_code = sympy.printing.ccode(u)
# f_code = sympy.printing.ccode(f)
# u0_code = sympy.printing.ccode(u_0)
# uL_code = sympy.printing.ccode(u_L)
# print('u =', u_code)
# print('f =', f_code)
# print('u_0 =', u0_code)
# print('u_L =', uL_code)
#


# Define boundary conditions
def boundary(x, on_boundary):
    return on_boundary and ((F.near(x[0], 0., tol)) or (F.near(x[0], length, tol)))


g_expr = 'alpha*x[0]*x[0] + beta*t*t + 1'  # NOTE: x is a vector
g = F.Expression(g_expr, alpha=alpha, beta=beta, t=0, degree=2)  # NOTE: at initial time t=0

bcu = F.DirichletBC(V, g, boundary)  # NOTE: vary with t

# Define initial conditions
# Define functions for solutions at previous and current time steps
u_nm2 = F.project(g, V)  # NOTE: g^0
# u_nm2 = F.interpolate(g, V)
# u_nm2 = F.Function(V)  # NOTE: initial 0?
# u_nm2 = F.Constant(0)
# u_nm1 = u_nm2  # NOTE: by du(x, 0)/dt = 0
u_nm1 = F.project(u_nm2, V)  # BUG

# Define trial and test functions
u = F.TrialFunction(V)
v = F.TestFunction(V)

# Define expressions used in variational forms
DT = F.Constant(dt)  # NOTE: make a Constant
C = F.Constant(c)

f = F.Expression('2*beta - 2*alpha*c*c', beta=beta, alpha=alpha, c=c, degree=2)  # NOTE: not vary with t
# f = F.Constant(0)  # NOTE: faster?

# Define variational problem
# a = u * v * F.dx + (DT ** 2) * (C ** 2) * F.dot(F.grad(u), F.grad(v)) * F.dx
a = F.dot(u, v) * F.dx + (DT ** 2) * (C ** 2) * F.dot(F.grad(u), F.grad(v)) * F.dx
L = ((DT ** 2) * f + 2 * u_nm1 - u_nm2) * v * F.dx

# Solve
u_ = F.Function(V)  # NOTE: solution at current t, not trial function

t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    g.t = t  # NOTE: necessary, update bcs

    # Solve
    F.solve(a == L, u_,  bcu)

    # Plot solution
    F.plot(u_, title='Velocity')

    # Compute error at vertices
    # print('u max:', np.array(u_.vector()).max())
    u_e = F.interpolate(g, V)
    # F.plot(u_e)
    error = np.abs(np.array(u_e.vector()) - np.array(u_.vector())).max()
    print('t = {:.2f}: error = {:.3g}'.format(t, error))

    # Update previous solution
    u_nm2.assign(u_nm1)
    u_nm1.assign(u_)


# TODO: below
# Assemble matrices
# A = F.assemble(a)

# Apply boundary conditions to matrices
# [bc.apply(A) for bc in bcu]

# Create XDMF files for visualization output
# xdmffile_u = F.XDMFFile('vocal_tract_test/velocity.xdmf')

# Create time series (for use in reaction_system.py)
# timeseries_u = F.TimeSeries('vocal_tract_test/velocity_series')

# Save mesh to file (for use in reaction_system.py)
# F.File('vocal_tract_test/mesh.xml.gz') << mesh

# Create progress bar
# progress = F.Progress('Time-stepping')

# F.set_log_level(F.LogLevel.PROGRESS)  # PROGRESS = 16

# Time-stepping

# t = 0
# for n in range(num_steps):

    # # Update current time
    # t += dt
    # u_expr.t = t

    # # Solve
    # b = F.assemble(L)
    # [bc.apply(b) for bc in bcu]
    # # F.solve(A, u_.vector(), b, 'bicgstab', 'hypre_amg')
    # F.solve(A, u_.vector(), b, 'bicgstab', 'hypre_amg')

    # # Plot solution
    # F.plot(u_, title='Velocity')

    # # Save solution to file (XDMF/HDF5)
    # xdmffile_u.write(u_, t)

    # # Save nodal values to file
    # timeseries_u.store(u_.vector(), t)

    # # Update previous solution
    # u_nm1.assign(u_)

    # # Update progress bar
    # # progress.update(t / T)  # BUG: not working
    # # progress += (t / T)
    # print('u max:', np.array(u_.vector()).max())

# Hold plot
plt.show()


def test_solver():
    pass


def run_solver():
    pass


if __name__ == '__main__':
    run_solver()
