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


def UnitHyperCube(divisions):
    '''
    Make unit hyper-cube mesh with uniformly spaced cells.
    Currently only support maximum 3D.

    Parameters
    ----------
    divisions: List[int]
        List of number of vertices along each dimension.

    Returns
    -------
    mesh
    '''
    mesh_classes = [F.UnitIntervalMesh, F.UnitSquareMesh, F.UnitCubeMesh]
    d = len(divisions)
    mesh = mesh_classes[d - 1](*divisions)
    return mesh


def pde_solver(f, u_D, u_I, c,
               f_boundary, divisions,
               T, dt, num_steps,
               initial_method='project', degree=1,
               g=None):
    '''
    Solve

        d^u/dt^2 = c^2 d^u/dx^2 + f(x, t)
        s.t.    u(t) = u_D  @ DirichletBC
                du/dn = 0   @ NeumannBC
                u(x, 0) = u_I
                du(x, 0)/dt = 0

    with Backward Euler in time.

    Parameters
    ----------
    f: FEniCS.Expression
        Constitute linear form L(v), which is on the right hand side of
        the variational form. Could be time-dependent.
    u_D: FEniCS.Expression
        Expression for Dirichlet boundary condition.
        Could be time-dependent.
    u_I: FEniCS.Expression
        Expression for initial condition.
    c: float
        Speed of sound in the medium defined by the problem.
    f_boundary: function
        Function that returns boolean value indicating on boundary or not.
    divisions: List[int]
        List of number of vertices along each dimension.
    T: float
        Total time span.
    dt: float
        Time step size.
    num_steps: int
        Number of time steps.
    initial_method: string
        Method to determine initial values.
        Can be either 'project' or 'interpolate'.
    degree: int
        Degree of the (Lagrange) polynomial element.
    g: FEniCS.Expression
        Exact solution.

    Returns
    -------
    '''

    # Create mesh
    mesh = UnitHyperCube(divisions)

    # Define function spaces
    V = F.FunctionSpace(mesh, 'P', degree)

    # Define boundary conditions
    bcu = F.DirichletBC(V, u_D, f_boundary)  # NOTE: vary with t

    # Define initial conditions
    # define functions for solutions at previous and current time steps
    if initial_method == 'project':
        # by u^0 = u_I
        # solving <u_nm2, v> = <u_I, v> for u_nm2
        u_nm2 = F.project(u_I, V)
        # by du^0/dt = 0 => u_nm1 = u_nm2
        # solving <u_nm1, v> = <u_nm2, v> for u_nm1
        u_nm1 = F.project(u_nm2, V)

    elif initial_method == 'interpolate':
        # by u^0 = U^j phi_j = u_I
        # setting U^j = u_I[j]
        u_nm2 = F.interpolate(u_I, V)
        # by du^0/dt = 0 => u_nm1 = U^j phi_j = u_nm2
        # setting U^j = u_nm2[j]
        u_nm1 = F.interpolate(u_nm2, V)

    # Define expressions used in variational forms
    DT = F.Constant(dt)  # NOTE: make a Constant
    C = F.Constant(c)

    def _make_variational_problem(V):
        '''
        Parameters
        ----------
        V: FEniCS.FunctionSpace

        Returns
        -------
        a, L: FEniCS.Expression
            Variational forms.
        '''
        # Define trial and test functions
        u = F.TrialFunction(V)
        v = F.TestFunction(V)

        # Define variational problem
        a = F.dot(u, v) * F.dx +\
            (DT ** 2) * (C ** 2) * F.dot(F.grad(u), F.grad(v)) * F.dx
        L = ((DT ** 2) * f + 2 * u_nm1 - u_nm2) * v * F.dx
        return a, L

    # a, L = _make_variational_problem(V)

    # TODO: log
    # F.set_log_level(F.LogLevel.PROGRESS)  # PROGRESS = 16

    # TODO: save?
    # Create XDMF files for visualization output
    # xdmffile_u = F.XDMFFile('vocal_tract_test/velocity.xdmf')
    # Create time series (for use in reaction_system.py)
    # timeseries_u = F.TimeSeries('vocal_tract_test/velocity_series')
    # Save mesh to file (for use in reaction_system.py)
    # F.File('vocal_tract_test/mesh.xml.gz') << mesh

    def _assembly(V):
        # Define trial and test functions
        u = F.TrialFunction(V)
        v = F.TestFunction(V)

        # Assemble
        a_M = F.dot(u, v) * F.dx
        a_K = F.dot(F.grad(u), F.grad(v)) * F.dx

        M = F.assemble(a_M)
        K = F.assemble(a_K)

        A = M + (DT ** 2) * (C ** 2) * K
        return M, K, A

    M, K, A = _assembly(V)

    #
    # Solve in time
    u_ = F.Function(V)  # NOTE: solution at current t, not trial function

    t = 0
    # TODO: Create progress bar?
    # progress = F.Progress('Time-stepping')
    # A = F.assemble(a)
    plt.figure()
    for n in range(num_steps):

        # Update current time
        t += dt
        u_D.t = t  # NOTE: necessary if bc vary with time
        # f.t = t  # NOTE: necessary if f vary with time

        # Solve
        # F.solve(a == L, u_,  bcu)

        # or equivalently
        # b = F.assemble(L)
        # bcu.apply(A, b)
        # F.solve(A, u_.vector(), b)

        # or
        F_n = F.interpolate(f, V).vector()

        b = 2 * M * u_nm1.vector() - M * u_nm2.vector() + (DT ** 2) * M * F_n
        bcu.apply(A, b)
        F.solve(A, u_.vector(), b)

        # Save solution to file (XDMF/HDF5)
        # xdmffile_u.write(u_, t)
        # Save nodal values to file
        # timeseries_u.store(u_.vector(), t)

        # Plot solution
        plt.subplot(121)
        F.plot(u_)
        # vertex_values = u_.compute_vertex_values(mesh)
        # plt.plot(np.linspace(0, T, len(vertex_values)), vertex_values)
        plt.xlabel('x')
        plt.ylabel('u')

        # Compute error at vertices
        if g is not None:
            # u_e = F.project(g, V)
            u_e = F.interpolate(g, V)

            error = np.abs(
                np.array(u_e.vector()) - np.array(u_.vector())
            ).max()
            print('[{:.2f}/{:.2f}]  error = {:.3g}'.format(t, T, error))

            plt.subplot(122)
            F.plot(u_e)
            plt.xlabel('x')
            plt.ylabel('u_e')

        else:
            # TODO: define residual?
            pass

        # Update previous solution
        u_nm2.assign(u_nm1)
        u_nm1.assign(u_)

    plt.show()


def test_probelm_1():

    c = 1.  # speed of sound in medium
    alpha = 1.
    beta = 5.

    T = 2.0  # final time
    num_steps = 500  # number of time steps
    dt = T / num_steps  # time step size

    length = 1.  # spatial dimension
    divisions = (64,)  # mesh size

    f = F.Expression('2*beta - 2*alpha*c*c',
                     beta=beta, alpha=alpha, c=c, degree=2)

    # Boundary expression
    uD_expr = 'alpha*x[0]*x[0] + beta*t*t + 1'  # NOTE: x is a vector
    u_D = F.Expression(uD_expr,  # NOTE: at initial time t=0
                       alpha=alpha, beta=beta, t=0, degree=2)
    # Initial expression
    u_I = u_D
    # u_I = F.Constant(0)
    # Exact solution
    g = u_D

    def boundary(x, on_boundary):
        '''
        Test if x is on boundary.
        '''
        tol = 1E-14
        return on_boundary and ((F.near(x[0], 0., tol)) or
                                (F.near(x[0], length, tol)))

    pde_solver(f, u_D, u_I, c,
               boundary, divisions,
               T, dt, num_steps,
               initial_method='project', degree=1, g=g)


def test_solver():
    test_probelm_1()


def run_solver():
    pass

#
# Use SymPy to compute f from the manufactured solution u
# x, t = sympy.symbols('x[0], x[1]')
# u = 1 + alpha * (x ** 2) + beta * (t ** 2)
# x_coord = np.linspace(0, length, 64 + 1)
# t_coord = np.linspace(0, T, num_steps)
# # xv, tv = np.meshgrid(x_coord, t_coord)
# # z = 1 + alpha * (xv ** 2) + beta * (tv ** 2)
# z = lambda x, t: 1 + alpha * (x ** 2) + beta * (t ** 2)
# plt.figure()
# # h = plt.contourf(x_coord, t_coord, z)
# for t in t_coord:
    # zv = list(map(z, x_coord, [t] * len(x_coord)))
    # plt.plot(x_coord, zv)

# plt.xlim([0, 1])
# plt.ylim([1, 2])

# plt.show()
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


if __name__ == '__main__':
    test_solver()
    # run_solver()
